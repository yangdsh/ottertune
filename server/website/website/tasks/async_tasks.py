#
# OtterTune - async_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import random
import queue
import numpy as np
import tensorflow as tf
import gpflow
from pyDOE import lhs
from scipy.stats import uniform

from celery.task import task, Task
from celery.utils.log import get_task_logger
from djcelery.models import TaskMeta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from analysis.ddpg.ddpg import DDPG
from analysis.gp import GPRNP
from analysis.gp_tf import GPRGD
from analysis.nn_tf import NeuralNet
from analysis.gpr import gpr_models
from analysis.gpr import ucb
from analysis.gpr.optimize import tf_optimize
from analysis.preprocessing import Bin, DummyEncoder
from analysis.constraints import ParamConstraintHelper
from website.models import PipelineData, PipelineRun, Result, Workload, KnobCatalog, SessionKnob
from website import db
from website.types import PipelineTaskType, AlgorithmType
from website.utils import DataUtil, JSONUtil
from website.settings import IMPORTANT_KNOB_NUMBER, NUM_SAMPLES, TOP_NUM_CONFIG  # pylint: disable=no-name-in-module
from website.settings import (USE_GPFLOW, DEFAULT_LENGTH_SCALE, DEFAULT_MAGNITUDE,
                              MAX_TRAIN_SIZE, BATCH_SIZE, NUM_THREADS,
                              DEFAULT_RIDGE, DEFAULT_LEARNING_RATE,
                              DEFAULT_EPSILON, MAX_ITER, GPR_EPS,
                              DEFAULT_SIGMA_MULTIPLIER, DEFAULT_MU_MULTIPLIER,
                              DEFAULT_UCB_SCALE, HP_LEARNING_RATE, HP_MAX_ITER,
                              DDPG_SIMPLE_REWARD, DDPG_GAMMA, USE_DEFAULT,
                              DDPG_BATCH_SIZE, ACTOR_LEARNING_RATE,
                              CRITIC_LEARNING_RATE, UPDATE_EPOCHS,
                              ACTOR_HIDDEN_SIZES, CRITIC_HIDDEN_SIZES,
                              DNN_TRAIN_ITER, DNN_EXPLORE, DNN_EXPLORE_ITER,
                              DNN_NOISE_SCALE_BEGIN, DNN_NOISE_SCALE_END,
                              DNN_DEBUG, DNN_DEBUG_INTERVAL, GPR_DEBUG)

from website.settings import INIT_FLIP_PROB, FLIP_PROB_DECAY
from website.types import VarType


LOG = get_task_logger(__name__)


class UpdateTask(Task):  # pylint: disable=abstract-method

    def __init__(self):
        self.rate_limit = '50/m'
        self.max_retries = 3
        self.default_retry_delay = 60


class TrainDDPG(UpdateTask):  # pylint: disable=abstract-method
    def on_success(self, retval, task_id, args, kwargs):
        super(TrainDDPG, self).on_success(retval, task_id, args, kwargs)

        # Completely delete this result because it's huge and not
        # interesting
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = None
        task_meta.save()


class AggregateTargetResults(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(AggregateTargetResults, self).on_success(retval, task_id, args, kwargs)

        # Completely delete this result because it's huge and not
        # interesting
        task_meta = TaskMeta.objects.get(task_id=task_id)
        task_meta.result = None
        task_meta.save()


class MapWorkload(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(MapWorkload, self).on_success(retval, task_id, args, kwargs)

        # Replace result with formatted result
        if not args[0][0]['bad']:
            new_res = {
                'scores': sorted(args[0][0]['scores'].items()),
                'mapped_workload_id': args[0][0]['mapped_workload'],
            }
            task_meta = TaskMeta.objects.get(task_id=task_id)
            task_meta.result = new_res  # Only store scores
            task_meta.save()
        else:
            task_meta = TaskMeta.objects.get(task_id=task_id)
            task_meta.result = None
            task_meta.save()


class ConfigurationRecommendation(UpdateTask):  # pylint: disable=abstract-method

    def on_success(self, retval, task_id, args, kwargs):
        super(ConfigurationRecommendation, self).on_success(retval, task_id, args, kwargs)

        result_id = retval['result_id']
        result = Result.objects.get(pk=result_id)

        # Replace result with formatted result
        formatted_params = db.parser.format_dbms_knobs(result.dbms.pk, retval['recommendation'])
        task_meta = TaskMeta.objects.get(task_id=task_id)
        retval['recommendation'] = formatted_params
        task_meta.result = retval
        task_meta.save()

        # Create next configuration to try
        config = db.parser.create_knob_configuration(result.dbms.pk, retval['recommendation'])
        retval['recommendation'] = config
        result.next_configuration = JSONUtil.dumps(retval)
        result.save()


def clean_knob_data(knob_matrix, knob_labels, session):
    # Makes sure that all knobs in the dbms are included in the knob_matrix and knob_labels
    knob_cat = SessionKnob.objects.get_knobs_for_session(session)
    knob_cat = [knob["name"] for knob in knob_cat if knob["tunable"]]
    matrix = np.array(knob_matrix)
    missing_columns = set(knob_cat) - set(knob_labels)
    unused_columns = set(knob_labels) - set(knob_cat)
    LOG.debug("clean_knob_data added %d knobs and removed %d knobs.", len(missing_columns),
              len(unused_columns))
    # If columns are missing from the matrix
    if missing_columns:
        for knob in missing_columns:
            knob_object = KnobCatalog.objects.get(dbms=session.dbms, name=knob, tunable=True)
            index = knob_cat.index(knob)
            matrix = np.insert(matrix, index, knob_object.default, axis=1)
            knob_labels.insert(index, knob)
    # If they are useless columns in the matrix
    if unused_columns:
        indexes = [i for i, n in enumerate(knob_labels) if n in unused_columns]
        # Delete unused columns
        matrix = np.delete(matrix, indexes, 1)
        for i in sorted(indexes, reverse=True):
            del knob_labels[i]
    return matrix, knob_labels


@task(base=AggregateTargetResults, name='aggregate_target_results')
def aggregate_target_results(result_id, algorithm):
    # Check that we've completed the background tasks at least once. We need
    # this data in order to make a configuration recommendation (until we
    # implement a sampling technique to generate new training data).
    newest_result = Result.objects.get(pk=result_id)
    has_pipeline_data = PipelineData.objects.filter(workload=newest_result.workload).exists()
    if newest_result.session.tuning_session == 'lhs':
        all_samples = JSONUtil.loads(newest_result.session.lhs_samples)
        if len(all_samples) == 0:
            knobs = SessionKnob.objects.get_knobs_for_session(newest_result.session)
            all_samples = gen_lhs_samples(knobs, 100)
            LOG.debug('%s: Generated LHS.\n\ndata=%s\n',
                      AlgorithmType.name(algorithm), JSONUtil.dumps(all_samples[:5], pprint=True))
        samples = all_samples.pop()
        result = Result.objects.filter(pk=result_id)
        agg_data = DataUtil.aggregate_data(result)
        agg_data['newest_result_id'] = result_id
        agg_data['bad'] = True
        agg_data['config_recommend'] = samples
        newest_result.session.lhs_samples = JSONUtil.dumps(all_samples)
        newest_result.session.save()
        LOG.debug('%s: Got LHS config.\n\ndata=%s\n',
                  AlgorithmType.name(algorithm), JSONUtil.dumps(agg_data, pprint=True))

    elif not has_pipeline_data or newest_result.session.tuning_session == 'randomly_generate':
        if not has_pipeline_data and newest_result.session.tuning_session == 'tuning_session':
            LOG.debug("Background tasks haven't ran for this workload yet, picking random data.")

        result = Result.objects.filter(pk=result_id)
        knobs = SessionKnob.objects.get_knobs_for_session(newest_result.session)

        # generate a config randomly
        random_knob_result = gen_random_data(knobs)
        agg_data = DataUtil.aggregate_data(result)
        agg_data['newest_result_id'] = result_id
        agg_data['bad'] = True
        agg_data['config_recommend'] = random_knob_result
        LOG.debug('%s: Finished generating a random config.\n\ndata=%s\n',
                  AlgorithmType.name(algorithm), JSONUtil.dumps(agg_data, pprint=True))

    else:
        # Aggregate all knob config results tried by the target so far in this
        # tuning session and this tuning workload.
        target_results = Result.objects.filter(session=newest_result.session,
                                               dbms=newest_result.dbms,
                                               workload=newest_result.workload)
        if len(target_results) == 0:
            raise Exception('Cannot find any results for session_id={}, dbms_id={}'
                            .format(newest_result.session, newest_result.dbms))
        agg_data = DataUtil.aggregate_data(target_results)
        agg_data['newest_result_id'] = result_id
        agg_data['bad'] = False

        # Clean knob data
        cleaned_agg_data = clean_knob_data(agg_data['X_matrix'], agg_data['X_columnlabels'],
                                           newest_result.session)
        agg_data['X_matrix'] = np.array(cleaned_agg_data[0])
        agg_data['X_columnlabels'] = np.array(cleaned_agg_data[1])

        LOG.debug('%s: Finished aggregating target results.\n\ndata=%s\n',
                  AlgorithmType.name(algorithm), JSONUtil.dumps(agg_data, pprint=True))

    return agg_data, algorithm


def gen_random_data(knobs):
    random_knob_result = {}
    for knob in knobs:
        name = knob["name"]
        if knob["vartype"] == VarType.BOOL:
            flag = random.randint(0, 1)
            if flag == 0:
                random_knob_result[name] = False
            else:
                random_knob_result[name] = True
        elif knob["vartype"] == VarType.ENUM:
            enumvals = knob["enumvals"].split(',')
            enumvals_len = len(enumvals)
            rand_idx = random.randint(0, enumvals_len - 1)
            random_knob_result[name] = rand_idx
        elif knob["vartype"] == VarType.INTEGER:
            random_knob_result[name] = random.randint(int(knob["minval"]), int(knob["maxval"]))
        elif knob["vartype"] == VarType.REAL:
            random_knob_result[name] = random.uniform(
                float(knob["minval"]), float(knob["maxval"]))
        elif knob["vartype"] == VarType.STRING:
            random_knob_result[name] = "None"
        elif knob["vartype"] == VarType.TIMESTAMP:
            random_knob_result[name] = "None"
        else:
            raise Exception(
                'Unknown variable type: {}'.format(knob["vartype"]))

    return random_knob_result


def gen_lhs_samples(knobs, nsamples):
    names = []
    maxvals = []
    minvals = []
    types = []

    for knob in knobs:
        names.append(knob['name'])
        maxvals.append(float(knob['maxval']))
        minvals.append(float(knob['minval']))
        types.append(knob['vartype'])

    nfeats = len(knobs)
    samples = lhs(nfeats, samples=nsamples, criterion='maximin')
    maxvals = np.array(maxvals)
    minvals = np.array(minvals)
    scales = maxvals - minvals
    for fidx in range(nfeats):
        samples[:, fidx] = uniform(loc=minvals[fidx], scale=scales[fidx]).ppf(samples[:, fidx])
    lhs_samples = []
    for sidx in range(nsamples):
        lhs_samples.append(dict())
        for fidx in range(nfeats):
            if types[fidx] == VarType.INTEGER:
                lhs_samples[-1][names[fidx]] = int(round(samples[sidx][fidx]))
            elif types[fidx] == VarType.REAL:
                lhs_samples[-1][names[fidx]] = float(samples[sidx][fidx])
            else:
                LOG.debug("LHS type not supported: %s", types[fidx])

    return lhs_samples


@task(base=TrainDDPG, name='train_ddpg')
def train_ddpg(result_id):
    LOG.info('Add training data to ddpg and train ddpg')
    result = Result.objects.get(pk=result_id)
    session = Result.objects.get(pk=result_id).session
    session_results = Result.objects.filter(session=session,
                                            creation_time__lt=result.creation_time)
    result_info = {}
    result_info['newest_result_id'] = result_id

    # Extract data from result and previous results
    result = Result.objects.filter(pk=result_id)
    if len(session_results) == 0:
        base_result_id = result_id
        prev_result_id = result_id
    else:
        base_result_id = session_results[0].pk
        prev_result_id = session_results[len(session_results)-1].pk
    base_result = Result.objects.filter(pk=base_result_id)
    prev_result = Result.objects.filter(pk=prev_result_id)

    agg_data = DataUtil.aggregate_data(result)
    metric_data = agg_data['y_matrix'].flatten()
    base_metric_data = (DataUtil.aggregate_data(base_result))['y_matrix'].flatten()
    prev_metric_data = (DataUtil.aggregate_data(prev_result))['y_matrix'].flatten()
    metric_scalar = MinMaxScaler().fit(metric_data.reshape(1, -1))
    normalized_metric_data = metric_scalar.transform(metric_data.reshape(1, -1))[0]

    # Clean knob data
    cleaned_knob_data = clean_knob_data(agg_data['X_matrix'], agg_data['X_columnlabels'], session)
    knob_data = np.array(cleaned_knob_data[0])
    knob_labels = np.array(cleaned_knob_data[1])
    knob_bounds = np.vstack(DataUtil.get_knob_bounds(knob_labels.flatten(), session))
    knob_data = MinMaxScaler().fit(knob_bounds).transform(knob_data)[0]
    knob_num = len(knob_data)
    metric_num = len(metric_data)
    LOG.info('knob_num: %d, metric_num: %d', knob_num, metric_num)

    # Filter ys by current target objective metric
    result = Result.objects.get(pk=result_id)
    target_objective = result.session.target_objective
    target_obj_idx = [i for i, n in enumerate(agg_data['y_columnlabels']) if n == target_objective]
    if len(target_obj_idx) == 0:
        raise Exception(('Could not find target objective in metrics '
                         '(target_obj={})').format(target_objective))
    elif len(target_obj_idx) > 1:
        raise Exception(('Found {} instances of target objective in '
                         'metrics (target_obj={})').format(len(target_obj_idx),
                                                           target_objective))
    objective = metric_data[target_obj_idx]
    base_objective = base_metric_data[target_obj_idx]
    prev_objective = prev_metric_data[target_obj_idx]
    metric_meta = db.target_objectives.get_metric_metadata(
        result.session.dbms.pk, result.session.target_objective)

    # Calculate the reward
    if DDPG_SIMPLE_REWARD:
        objective = objective / base_objective
        if metric_meta[target_objective].improvement == '(less is better)':
            reward = -objective
        else:
            reward = objective
    else:
        if metric_meta[target_objective].improvement == '(less is better)':
            if objective - base_objective <= 0:  # positive reward
                reward = (np.square((2 * base_objective - objective) / base_objective) - 1)\
                    * abs(2 * prev_objective - objective) / prev_objective
            else:  # negative reward
                reward = -(np.square(objective / base_objective) - 1) * objective / prev_objective
        else:
            if objective - base_objective > 0:  # positive reward
                reward = (np.square(objective / base_objective) - 1) * objective / prev_objective
            else:  # negative reward
                reward = -(np.square((2 * base_objective - objective) / base_objective) - 1)\
                    * abs(2 * prev_objective - objective) / prev_objective
    LOG.info('reward: %f', reward)

    # Update ddpg
    ddpg = DDPG(n_actions=knob_num, n_states=metric_num, alr=ACTOR_LEARNING_RATE,
                clr=CRITIC_LEARNING_RATE, gamma=DDPG_GAMMA, batch_size=DDPG_BATCH_SIZE,
                a_hidden_sizes=ACTOR_HIDDEN_SIZES, c_hidden_sizes=CRITIC_HIDDEN_SIZES,
                use_default=USE_DEFAULT)
    if session.ddpg_actor_model and session.ddpg_critic_model:
        ddpg.set_model(session.ddpg_actor_model, session.ddpg_critic_model)
    if session.ddpg_reply_memory:
        ddpg.replay_memory.set(session.ddpg_reply_memory)
    ddpg.add_sample(normalized_metric_data, knob_data, reward, normalized_metric_data)
    for _ in range(UPDATE_EPOCHS):
        ddpg.update()
    session.ddpg_actor_model, session.ddpg_critic_model = ddpg.get_model()
    session.ddpg_reply_memory = ddpg.replay_memory.get()
    session.save()
    return result_info


@task(base=ConfigurationRecommendation, name='configuration_recommendation_ddpg')
def configuration_recommendation_ddpg(result_info):  # pylint: disable=invalid-name
    LOG.info('Use ddpg to recommend configuration')
    result_id = result_info['newest_result_id']
    result = Result.objects.filter(pk=result_id)
    session = Result.objects.get(pk=result_id).session
    agg_data = DataUtil.aggregate_data(result)
    metric_data = agg_data['y_matrix'].flatten()
    metric_scalar = MinMaxScaler().fit(metric_data.reshape(1, -1))
    normalized_metric_data = metric_scalar.transform(metric_data.reshape(1, -1))[0]
    cleaned_knob_data = clean_knob_data(agg_data['X_matrix'], agg_data['X_columnlabels'],
                                        session)
    knob_labels = np.array(cleaned_knob_data[1]).flatten()
    knob_num = len(knob_labels)
    metric_num = len(metric_data)

    ddpg = DDPG(n_actions=knob_num, n_states=metric_num, a_hidden_sizes=ACTOR_HIDDEN_SIZES,
                c_hidden_sizes=CRITIC_HIDDEN_SIZES, use_default=USE_DEFAULT)
    if session.ddpg_actor_model is not None and session.ddpg_critic_model is not None:
        ddpg.set_model(session.ddpg_actor_model, session.ddpg_critic_model)
    if session.ddpg_reply_memory is not None:
        ddpg.replay_memory.set(session.ddpg_reply_memory)
    knob_data = ddpg.choose_action(normalized_metric_data)

    knob_bounds = np.vstack(DataUtil.get_knob_bounds(knob_labels, session))
    knob_data = MinMaxScaler().fit(knob_bounds).inverse_transform(knob_data.reshape(1, -1))[0]
    conf_map = {k: knob_data[i] for i, k in enumerate(knob_labels)}
    conf_map_res = {}
    conf_map_res['status'] = 'good'
    conf_map_res['result_id'] = result_id
    conf_map_res['recommendation'] = conf_map
    conf_map_res['info'] = 'INFO: ddpg'
    return conf_map_res


@task(base=ConfigurationRecommendation, name='configuration_recommendation')
def configuration_recommendation(recommendation_input):
    target_data, algorithm = recommendation_input
    LOG.info('configuration_recommendation called')

    if target_data['bad'] is True:
        target_data_res = dict(
            status='bad',
            result_id=target_data['newest_result_id'],
            info='WARNING: no training data, the config is generated randomly',
            recommendation=target_data['config_recommend'],
            pipeline_run=target_data['pipeline_run'])
        LOG.debug('%s: Skipping configuration recommendation.\n\ndata=%s\n',
                  AlgorithmType.name(algorithm), JSONUtil.dumps(target_data, pprint=True))
        return target_data_res

    # Load mapped workload data
    mapped_workload_id = target_data['mapped_workload'][0]

    latest_pipeline_run = PipelineRun.objects.get(pk=target_data['pipeline_run'])
    mapped_workload = Workload.objects.get(pk=mapped_workload_id)
    workload_knob_data = PipelineData.objects.get(
        pipeline_run=latest_pipeline_run,
        workload=mapped_workload,
        task_type=PipelineTaskType.KNOB_DATA)
    workload_knob_data = JSONUtil.loads(workload_knob_data.data)
    workload_metric_data = PipelineData.objects.get(
        pipeline_run=latest_pipeline_run,
        workload=mapped_workload,
        task_type=PipelineTaskType.METRIC_DATA)
    workload_metric_data = JSONUtil.loads(workload_metric_data.data)

    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    cleaned_workload_knob_data = clean_knob_data(workload_knob_data["data"],
                                                 workload_knob_data["columnlabels"],
                                                 newest_result.session)

    X_workload = np.array(cleaned_workload_knob_data[0])
    X_columnlabels = np.array(cleaned_workload_knob_data[1])
    y_workload = np.array(workload_metric_data['data'])
    y_columnlabels = np.array(workload_metric_data['columnlabels'])
    rowlabels_workload = np.array(workload_metric_data['rowlabels'])

    # Target workload data
    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    X_target = target_data['X_matrix']
    y_target = target_data['y_matrix']
    rowlabels_target = np.array(target_data['rowlabels'])

    if not np.array_equal(X_columnlabels, target_data['X_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'))
    if not np.array_equal(y_columnlabels, target_data['y_columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'))

    # Filter Xs by top 10 ranked knobs
    ranked_knobs = PipelineData.objects.get(
        pipeline_run=latest_pipeline_run,
        workload=mapped_workload,
        task_type=PipelineTaskType.RANKED_KNOBS)
    ranked_knobs = JSONUtil.loads(ranked_knobs.data)[:IMPORTANT_KNOB_NUMBER]
    ranked_knob_idxs = [i for i, cl in enumerate(X_columnlabels) if cl in ranked_knobs]
    X_workload = X_workload[:, ranked_knob_idxs]
    X_target = X_target[:, ranked_knob_idxs]
    X_columnlabels = X_columnlabels[ranked_knob_idxs]

    # Filter ys by current target objective metric
    target_objective = newest_result.session.target_objective
    target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]
    if len(target_obj_idx) == 0:
        raise Exception(('Could not find target objective in metrics '
                         '(target_obj={})').format(target_objective))
    elif len(target_obj_idx) > 1:
        raise Exception(('Found {} instances of target objective in '
                         'metrics (target_obj={})').format(len(target_obj_idx),
                                                           target_objective))

    metric_meta = db.target_objectives.get_metric_metadata(
        newest_result.session.dbms.pk, newest_result.session.target_objective)
    lessisbetter = metric_meta[target_objective].improvement == db.target_objectives.LESS_IS_BETTER

    y_workload = y_workload[:, target_obj_idx]
    y_target = y_target[:, target_obj_idx]
    y_columnlabels = y_columnlabels[target_obj_idx]

    # Combine duplicate rows in the target/workload data (separately)
    X_workload, y_workload, rowlabels_workload = DataUtil.combine_duplicate_rows(
        X_workload, y_workload, rowlabels_workload)
    X_target, y_target, rowlabels_target = DataUtil.combine_duplicate_rows(
        X_target, y_target, rowlabels_target)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    target_row_tups = [tuple(row) for row in X_target]
    for i, row in enumerate(X_workload):
        if tuple(row) in target_row_tups:
            dups_filter[i] = False
    X_workload = X_workload[dups_filter, :]
    y_workload = y_workload[dups_filter, :]
    rowlabels_workload = rowlabels_workload[dups_filter]

    # Combine target & workload Xs for preprocessing
    X_matrix = np.vstack([X_target, X_workload])

    # Dummy encode categorial variables
    categorical_info = DataUtil.dummy_encoder_helper(X_columnlabels,
                                                     mapped_workload.dbms)
    dummy_encoder = DummyEncoder(categorical_info['n_values'],
                                 categorical_info['categorical_features'],
                                 categorical_info['cat_columnlabels'],
                                 categorical_info['noncat_columnlabels'])
    X_matrix = dummy_encoder.fit_transform(X_matrix)

    # below two variables are needed for correctly determing max/min on dummies
    binary_index_set = set(categorical_info['binary_vars'])
    total_dummies = dummy_encoder.total_dummies()

    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)

    # Set up constraint helper
    constraint_helper = ParamConstraintHelper(scaler=X_scaler,
                                              encoder=dummy_encoder,
                                              binary_vars=categorical_info['binary_vars'],
                                              init_flip_prob=INIT_FLIP_PROB,
                                              flip_prob_decay=FLIP_PROB_DECAY)

    # FIXME (dva): check if these are good values for the ridge
    # ridge = np.empty(X_scaled.shape[0])
    # ridge[:X_target.shape[0]] = 0.01
    # ridge[X_target.shape[0]:] = 0.1

    # FIXME: we should generate more samples and use a smarter sampling
    # technique
    num_samples = NUM_SAMPLES
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])
    X_scaler_matrix = np.zeros([1, X_scaled.shape[1]])

    session_knobs = SessionKnob.objects.get_knobs_for_session(newest_result.session)

    # Set min/max for knob values
    for i in range(X_scaled.shape[1]):
        if i < total_dummies or i in binary_index_set:
            col_min = 0
            col_max = 1
        else:
            col_min = X_scaled[:, i].min()
            col_max = X_scaled[:, i].max()
            for knob in session_knobs:
                if X_columnlabels[i] == knob["name"]:
                    X_scaler_matrix[0][i] = knob["minval"]
                    col_min = X_scaler.transform(X_scaler_matrix)[0][i]
                    X_scaler_matrix[0][i] = knob["maxval"]
                    col_max = X_scaler.transform(X_scaler_matrix)[0][i]
        X_min[i] = col_min
        X_max[i] = col_max
        X_samples[:, i] = np.random.rand(num_samples) * (col_max - col_min) + col_min

    # Maximize the throughput, moreisbetter
    # Use gradient descent to minimize -throughput
    if not lessisbetter:
        y_scaled = -y_scaled

    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))

    i = 0
    while i < TOP_NUM_CONFIG:
        try:
            item = q.get_nowait()
            # Tensorflow get broken if we use the training data points as
            # starting points for GPRGD. We add a small bias for the
            # starting points. GPR_EPS default value is 0.001
            # if the starting point is X_max, we minus a small bias to
            # make sure it is within the range.
            dist = sum(np.square(X_max - X_scaled[item[1]]))
            if dist < 0.001:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] - abs(GPR_EPS)))
            else:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] + abs(GPR_EPS)))
            i = i + 1
        except queue.Empty:
            break

    session = newest_result.session
    res = None

    if algorithm == AlgorithmType.DNN:
        # neural network model
        model_nn = NeuralNet(n_input=X_samples.shape[1],
                             batch_size=X_samples.shape[0],
                             explore_iters=DNN_EXPLORE_ITER,
                             noise_scale_begin=DNN_NOISE_SCALE_BEGIN,
                             noise_scale_end=DNN_NOISE_SCALE_END,
                             debug=DNN_DEBUG,
                             debug_interval=DNN_DEBUG_INTERVAL)
        if session.dnn_model is not None:
            model_nn.set_weights_bin(session.dnn_model)
        model_nn.fit(X_scaled, y_scaled, fit_epochs=DNN_TRAIN_ITER)
        res = model_nn.recommend(X_samples, X_min, X_max,
                                 explore=DNN_EXPLORE, recommend_epochs=MAX_ITER)
        session.dnn_model = model_nn.get_weights_bin()
        session.save()

    elif algorithm == AlgorithmType.GPR:
        # default gpr model
        if USE_GPFLOW:
            model_kwargs = {}
            model_kwargs['model_learning_rate'] = HP_LEARNING_RATE
            model_kwargs['model_maxiter'] = HP_MAX_ITER
            opt_kwargs = {}
            opt_kwargs['learning_rate'] = DEFAULT_LEARNING_RATE
            opt_kwargs['maxiter'] = MAX_ITER
            opt_kwargs['bounds'] = [X_min, X_max]
            opt_kwargs['debug'] = GPR_DEBUG
            ucb_beta = 'get_beta_td'
            opt_kwargs['ucb_beta'] = ucb.get_ucb_beta(ucb_beta, scale=DEFAULT_UCB_SCALE,
                                                      t=i + 1., ndim=X_scaled.shape[1])
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            gpflow.reset_default_session(graph=graph)
            m = gpr_models.create_model('BasicGP', X=X_scaled, y=y_scaled, **model_kwargs)
            res = tf_optimize(m.model, X_samples, **opt_kwargs)
        else:
            model = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
                          magnitude=DEFAULT_MAGNITUDE,
                          max_train_size=MAX_TRAIN_SIZE,
                          batch_size=BATCH_SIZE,
                          num_threads=NUM_THREADS,
                          learning_rate=DEFAULT_LEARNING_RATE,
                          epsilon=DEFAULT_EPSILON,
                          max_iter=MAX_ITER,
                          sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER,
                          mu_multiplier=DEFAULT_MU_MULTIPLIER)
            model.fit(X_scaled, y_scaled, X_min, X_max, ridge=DEFAULT_RIDGE)
            res = model.predict(X_samples, constraint_helper=constraint_helper)

    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    best_config = X_scaler.inverse_transform(best_config)
    # Decode one-hot encoding into categorical knobs
    best_config = dummy_encoder.inverse_transform(best_config)

    # Although we have max/min limits in the GPRGD training session, it may
    # lose some precisions. e.g. 0.99..99 >= 1.0 may be True on the scaled data,
    # when we inversely transform the scaled data, the different becomes much larger
    # and cannot be ignored. Here we check the range on the original data
    # directly, and make sure the recommended config lies within the range
    X_min_inv = X_scaler.inverse_transform(X_min)
    X_max_inv = X_scaler.inverse_transform(X_max)
    best_config = np.minimum(best_config, X_max_inv)
    best_config = np.maximum(best_config, X_min_inv)

    conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
    conf_map_res = dict(
        status='good',
        result_id=target_data['newest_result_id'],
        recommendation=conf_map,
        info='INFO: training data size is {}'.format(X_scaled.shape[0]),
        pipeline_run=latest_pipeline_run.pk)
    LOG.debug('%s: Finished selecting the next config.\n\ndata=%s\n',
              AlgorithmType.name(algorithm), JSONUtil.dumps(conf_map_res, pprint=True))

    return conf_map_res


def load_data_helper(filtered_pipeline_data, workload, task_type):
    pipeline_data = filtered_pipeline_data.get(workload=workload,
                                               task_type=task_type)
    LOG.debug("PIPELINE DATA: %s", str(pipeline_data.data))
    return JSONUtil.loads(pipeline_data.data)


@task(base=MapWorkload, name='map_workload')
def map_workload(map_workload_input):
    target_data, algorithm = map_workload_input

    if target_data['bad']:
        assert target_data is not None
        target_data['pipeline_run'] = None
        LOG.debug('%s: Skipping workload mapping.\n\ndata=%s\n',
                  AlgorithmType.name(algorithm), JSONUtil.dumps(target_data, pprint=True))

        return target_data, algorithm

    # Get the latest version of pipeline data that's been computed so far.
    latest_pipeline_run = PipelineRun.objects.get_latest()
    assert latest_pipeline_run is not None
    target_data['pipeline_run'] = latest_pipeline_run.pk

    newest_result = Result.objects.get(pk=target_data['newest_result_id'])
    target_workload = newest_result.workload
    X_columnlabels = np.array(target_data['X_columnlabels'])
    y_columnlabels = np.array(target_data['y_columnlabels'])

    # Find all pipeline data belonging to the latest version with the same
    # DBMS and hardware as the target
    pipeline_data = PipelineData.objects.filter(
        pipeline_run=latest_pipeline_run,
        workload__dbms=target_workload.dbms,
        workload__hardware=target_workload.hardware)

    # FIXME (dva): we should also compute the global (i.e., overall) ranked_knobs
    # and pruned metrics but we just use those from the first workload for now
    initialized = False
    global_ranked_knobs = None
    global_pruned_metrics = None
    ranked_knob_idxs = None
    pruned_metric_idxs = None

    # Compute workload mapping data for each unique workload
    unique_workloads = pipeline_data.values_list('workload', flat=True).distinct()
    assert len(unique_workloads) > 0
    workload_data = {}
    for unique_workload in unique_workloads:

        workload_obj = Workload.objects.get(pk=unique_workload)
        wkld_results = Result.objects.filter(workload=workload_obj)
        if wkld_results.exists() is False:
            # delete the workload
            workload_obj.delete()
            continue

        # Load knob & metric data for this workload
        knob_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.KNOB_DATA)
        knob_data["data"], knob_data["columnlabels"] = clean_knob_data(knob_data["data"],
                                                                       knob_data["columnlabels"],
                                                                       newest_result.session)

        metric_data = load_data_helper(pipeline_data, unique_workload, PipelineTaskType.METRIC_DATA)
        X_matrix = np.array(knob_data["data"])
        y_matrix = np.array(metric_data["data"])
        rowlabels = np.array(knob_data["rowlabels"])
        assert np.array_equal(rowlabels, metric_data["rowlabels"])

        if not initialized:
            # For now set ranked knobs & pruned metrics to be those computed
            # for the first workload
            global_ranked_knobs = load_data_helper(
                pipeline_data, unique_workload,
                PipelineTaskType.RANKED_KNOBS)[:IMPORTANT_KNOB_NUMBER]
            global_pruned_metrics = load_data_helper(
                pipeline_data, unique_workload, PipelineTaskType.PRUNED_METRICS)
            ranked_knob_idxs = [i for i in range(X_matrix.shape[1]) if X_columnlabels[
                i] in global_ranked_knobs]
            pruned_metric_idxs = [i for i in range(y_matrix.shape[1]) if y_columnlabels[
                i] in global_pruned_metrics]

            # Filter X & y columnlabels by top ranked_knobs & pruned_metrics
            X_columnlabels = X_columnlabels[ranked_knob_idxs]
            y_columnlabels = y_columnlabels[pruned_metric_idxs]
            initialized = True

        # Filter X & y matrices by top ranked_knobs & pruned_metrics
        X_matrix = X_matrix[:, ranked_knob_idxs]
        y_matrix = y_matrix[:, pruned_metric_idxs]

        # Combine duplicate rows (rows with same knob settings)
        X_matrix, y_matrix, rowlabels = DataUtil.combine_duplicate_rows(
            X_matrix, y_matrix, rowlabels)

        workload_data[unique_workload] = {
            'X_matrix': X_matrix,
            'y_matrix': y_matrix,
            'rowlabels': rowlabels,
        }

    assert len(workload_data) > 0

    # Stack all X & y matrices for preprocessing
    Xs = np.vstack([entry['X_matrix'] for entry in list(workload_data.values())])
    ys = np.vstack([entry['y_matrix'] for entry in list(workload_data.values())])

    # Scale the X & y values, then compute the deciles for each column in y
    X_scaler = StandardScaler(copy=False)
    X_scaler.fit(Xs)
    y_scaler = StandardScaler(copy=False)
    y_scaler.fit_transform(ys)
    y_binner = Bin(bin_start=1, axis=0)
    y_binner.fit(ys)
    del Xs
    del ys

    # Filter the target's X & y data by the ranked knobs & pruned metrics.
    X_target = target_data['X_matrix'][:, ranked_knob_idxs]
    y_target = target_data['y_matrix'][:, pruned_metric_idxs]

    # Now standardize the target's data and bin it by the deciles we just
    # calculated
    X_target = X_scaler.transform(X_target)
    y_target = y_scaler.transform(y_target)
    y_target = y_binner.transform(y_target)

    scores = {}
    for workload_id, workload_entry in list(workload_data.items()):
        predictions = np.empty_like(y_target)
        X_workload = workload_entry['X_matrix']
        X_scaled = X_scaler.transform(X_workload)
        y_workload = workload_entry['y_matrix']
        y_scaled = y_scaler.transform(y_workload)
        for j, y_col in enumerate(y_scaled.T):
            # Using this workload's data, train a Gaussian process model
            # and then predict the performance of each metric for each of
            # the knob configurations attempted so far by the target.
            y_col = y_col.reshape(-1, 1)
            model = GPRNP(length_scale=DEFAULT_LENGTH_SCALE,
                          magnitude=DEFAULT_MAGNITUDE,
                          max_train_size=MAX_TRAIN_SIZE,
                          batch_size=BATCH_SIZE)
            model.fit(X_scaled, y_col, ridge=DEFAULT_RIDGE)
            predictions[:, j] = model.predict(X_target).ypreds.ravel()
        # Bin each of the predicted metric columns by deciles and then
        # compute the score (i.e., distance) between the target workload
        # and each of the known workloads
        predictions = y_binner.transform(predictions)
        dists = np.sqrt(np.sum(np.square(
            np.subtract(predictions, y_target)), axis=1))
        scores[workload_id] = np.mean(dists)

    # Find the best (minimum) score
    best_score = np.inf
    best_workload_id = None
    best_workload_name = None
    scores_info = {}
    for workload_id, similarity_score in list(scores.items()):
        workload_name = Workload.objects.get(pk=workload_id).name
        if similarity_score < best_score:
            best_score = similarity_score
            best_workload_id = workload_id
            best_workload_name = workload_name
        scores_info[workload_id] = (workload_name, similarity_score)
    target_data.update(mapped_workload=(best_workload_id, best_workload_name, best_score),
                       scores=scores_info)
    LOG.debug('%s: Finished mapping the workload.\n\ndata=%s\n',
              AlgorithmType.name(algorithm), JSONUtil.dumps(target_data, pprint=True))

    return target_data, algorithm
