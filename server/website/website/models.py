from collections import namedtuple, OrderedDict

from django.contrib.auth.models import User
from django.core.validators import (validate_comma_separated_integer_list,
                                    MinValueValidator)
from django.db import models
from django.utils.timezone import now

from .types import (DBMSType, LabelStyleType, MetricType, HardwareType,
                    KnobUnitType, PipelineTaskType, VarType)


class BaseModel(object):

    @classmethod
    def get_labels(cls, style=LabelStyleType.DEFAULT_STYLE):
        from .utils import LabelUtil

        labels = {}
        fields = cls._meta.get_fields()
        for field in fields:
            try:
                verbose_name = field.verbose_name
                if field.name == 'id':
                    verbose_name = cls._model_name() + ' id'
                labels[field.name] = verbose_name
            except:
                pass
        return LabelUtil.style_labels(labels, style)

    @classmethod
    def _model_name(cls):
        return cls.__name__


class DBMSCatalog(models.Model):
    type = models.IntegerField(choices=DBMSType.choices())
    version = models.CharField(max_length=16)

    @property
    def name(self):
        return DBMSType.name(self.type)

    @property
    def key(self):
        return '{}_{}'.format(self.name, self.version)

    @property
    def full_name(self):
        return '{} v{}'.format(self.name, self.version)

    def __unicode__(self):
        return self.full_name


class KnobCatalog(models.Model, BaseModel):
    dbms = models.ForeignKey(DBMSCatalog)
    name = models.CharField(max_length=64)
    vartype = models.IntegerField(choices=VarType.choices(), verbose_name="variable type")
    unit = models.IntegerField(choices=KnobUnitType.choices())
    category = models.TextField(null=True)
    summary = models.TextField(null=True, verbose_name='description')
    description = models.TextField(null=True)
    scope = models.CharField(max_length=16)
    minval = models.CharField(max_length=32, null=True, verbose_name="minimum value")
    maxval = models.CharField(max_length=32, null=True, verbose_name="maximum value")
    default = models.TextField(verbose_name="default value")
    enumvals = models.TextField(null=True, verbose_name="valid values")
    context = models.CharField(max_length=32)
    tunable = models.BooleanField(verbose_name="tunable")


MetricMeta = namedtuple('MetricMeta', ['name', 'pprint', 'unit', 'short_unit', 'scale', 'improvement'])


class MetricManager(models.Manager):

    # Direction of performance improvement
    LESS_IS_BETTER = '(less is better)'
    MORE_IS_BETTER = '(more is better)'

    # Possible objective functions
    THROUGHPUT = 'throughput_txn_per_sec'
    THROUGHPUT_META = (THROUGHPUT, 'Throughput',
                       'transactions / second',
                       'txn/sec', 1, MORE_IS_BETTER)

    # Objective function metric metadata
    OBJ_META = { THROUGHPUT: THROUGHPUT_META }

    @staticmethod
    def get_default_metrics(target_objective=None):
        default_metrics = list(MetricManager.OBJ_META.keys())
        if target_objective is not None and target_objective not in default_metrics:
            default_metrics = [target_objective] + default_metrics
        return default_metrics

    @staticmethod
    def get_default_objective_function():
        return MetricManager.THROUGHPUT

    @staticmethod
    def get_metric_meta(dbms, include_target_objectives=True):
        numeric_metric_names = MetricCatalog.objects.filter(
            dbms=dbms, metric_type=MetricType.COUNTER).values_list('name', flat=True)
        numeric_metrics = {}
        for metname in numeric_metric_names:
            numeric_metrics[metname] = MetricMeta(metname, metname, 'events / second', 'events/sec', 1, '')
        sorted_metrics = [(mname, mmeta) for mname, mmeta in
                          sorted(numeric_metrics.iteritems())]
        if include_target_objectives:
            for mname, mmeta in sorted(MetricManager.OBJ_META.iteritems())[::-1]:
                sorted_metrics.insert(0, (mname, MetricMeta(*mmeta)))
        return OrderedDict(sorted_metrics)


class MetricCatalog(models.Model, BaseModel):
    objects = MetricManager()

    dbms = models.ForeignKey(DBMSCatalog)
    name = models.CharField(max_length=64)
    vartype = models.IntegerField(choices=VarType.choices())
    summary = models.TextField(null=True, verbose_name='description')
    scope = models.CharField(max_length=16)
    metric_type = models.IntegerField(choices=MetricType.choices())


class Project(models.Model, BaseModel):
    user = models.ForeignKey(User)
    name = models.CharField(max_length=64, verbose_name="project name")
    description = models.TextField(null=True, blank=True)
    creation_time = models.DateTimeField()
    last_update = models.DateTimeField()

    def delete(self, using=None):
        apps = Application.objects.filter(project=self)
        for x in apps:
            x.delete()
        super(Project, self).delete(using)

    def __unicode__(self):
        return self.name


class Hardware(models.Model):
    type = models.IntegerField(choices=HardwareType.choices())
    name = models.CharField(max_length=32)
    cpu = models.IntegerField()
    memory = models.FloatField()
    storage = models.CharField(max_length=64,
            validators=[validate_comma_separated_integer_list])
    storage_type = models.CharField(max_length=16)
    additional_specs = models.TextField(null=True)

    def __unicode__(self):
        return HardwareType.TYPE_NAMES[self.type]


class Application(models.Model, BaseModel):
    user = models.ForeignKey(User)
    name = models.CharField(max_length=64, verbose_name="application name")
    description = models.TextField(null=True, blank=True)
    dbms = models.ForeignKey(DBMSCatalog)
    hardware = models.ForeignKey(Hardware)

    project = models.ForeignKey(Project)
    creation_time = models.DateTimeField()
    last_update = models.DateTimeField()

    upload_code = models.CharField(max_length=30, unique=True)
    tuning_session = models.BooleanField()
    target_objective = models.CharField(max_length=64)
    nondefault_settings = models.TextField(null=True)

    def clean(self):
        if self.tuning_session:
            if self.target_objective is None:
                self.target_objective = MetricManager.get_default_objective_function()
        else:
            self.target_objective = None

    def delete(self, using=None):
        targets = DBConf.objects.filter(application=self)
        results = Result.objects.filter(application=self)
        for t in targets:
            t.delete()
        for r in results:
            r.delete()
        super(Application, self).delete(using)

    def __unicode__(self):
        return self.name 


class ExpManager(models.Manager):

    def create_name(self, config, key):
        ts = config.creation_time.strftime("%m-%d-%y")
        return (key + '@' + ts + '#' + str(config.pk))


class ExpModel(models.Model, BaseModel):
    application = models.ForeignKey(Application)
    name = models.CharField(max_length=50, verbose_name="configuration name")
    description = models.CharField(max_length=512, null=True, blank=True)
    creation_time = models.DateTimeField()
    configuration = models.TextField()

    def __unicode__(self):
        return self.name


class DBModel(ExpModel):
    dbms = models.ForeignKey(DBMSCatalog, verbose_name="dbms")
    orig_config_diffs = models.TextField()


class DBConfManager(ExpManager):

    def create_dbconf(self, app, config, orig_config_diffs,
                      dbms, desc=None):
        try:
            return DBConf.objects.get(application=app,
                                      configuration=config)
        except DBConf.DoesNotExist:
            conf = self.create(application=app,
                               configuration=config,
                               orig_config_diffs=orig_config_diffs,
                               dbms=dbms,
                               description=desc,
                               creation_time=now())
            conf.name = self.create_name(conf, dbms.key)
            conf.save()
            return conf


class DBConf(DBModel):
    objects = DBConfManager()


class DBMSMetricsManager(ExpManager):

    def create_dbms_metrics(self, app, config, orig_config_diffs,
                            exec_time, dbms, desc=None):
        metrics = self.create(application=app,
                              configuration=config,
                              orig_config_diffs=orig_config_diffs,
                              dbms=dbms,
                              execution_time=exec_time,
                              description=desc,
                              creation_time=now())
        metrics.name = self.create_name(metrics, dbms.key)
        metrics.save()
        return metrics


class DBMSMetrics(DBModel):
    objects = DBMSMetricsManager()

    execution_time = models.IntegerField(
        validators=[MinValueValidator(0)])


class WorkloadManager(models.Manager):

    def create_workload(self, dbms, hardware, name):
        try:
            return Workload.objects.get(name=name)
        except Workload.DoesNotExist:
            return self.create(dbms=dbms,
                               hardware=hardware,
                               name=name)


class Workload(models.Model, BaseModel):
#     __DEFAULT_FMT = '{db}_{hw}_UNASSIGNED'.format

    objects = WorkloadManager()

    dbms = models.ForeignKey(DBMSCatalog)
    hardware = models.ForeignKey(Hardware)
    name = models.CharField(max_length=128, unique=True, verbose_name='workload name')

#     @property
#     def isdefault(self):
#         return self.cluster_name == self.default
#
#     @property
#     def default(self):
#         return self.__DEFAULT_FMT(db=self.dbms.pk,
#                                   hw=self.hardware.pk)
#
#     @staticmethod
#     def get_default(dbms_id, hw_id):
#         return Workload.__DEFAULT_FMT(db=dbms_id,
#                                       hw=hw_id)

    def __unicode__(self):
        return self.name


class ResultManager(models.Manager):

    def create_result(self, app, dbms, workload,
                      dbms_config, dbms_metrics,
                      start_timestamp, end_timestamp,
                      execution_time, task_ids=None,
                      most_similar=None):
        return self.create(application=app,
                           dbms=dbms,
                           workload=workload,
                           dbms_config=dbms_config,
                           dbms_metrics=dbms_metrics,
                           start_timestamp=start_timestamp,
                           end_timestamp=end_timestamp,
                           execution_time=execution_time,
                           task_ids=task_ids,
                           most_similar=most_similar,
                           creation_time=now())


class Result(models.Model, BaseModel):
    objects = ResultManager()

    application = models.ForeignKey(Application, verbose_name='application name')
    dbms = models.ForeignKey(DBMSCatalog)
    workload = models.ForeignKey(Workload)
    dbms_config = models.ForeignKey(DBConf)
    dbms_metrics = models.ForeignKey(DBMSMetrics)

    creation_time = models.DateTimeField()
    start_timestamp = models.DateTimeField()
    end_timestamp = models.DateTimeField()
    execution_time = models.FloatField()
    task_ids = models.CharField(max_length=180, null=True)
    most_similar = models.CharField(max_length=100, validators=[
                                    validate_comma_separated_integer_list],
                                    null=True)

    def __unicode__(self):
        return unicode(self.pk)


class ResultData(models.Model):
    result = models.ForeignKey(Result)
    workload_cluster = models.ForeignKey(Workload)
    param_data = models.TextField()
    metric_data = models.TextField()

    class Meta:
        ordering = ('workload_cluster',)

    def clean_fields(self, exclude=None):
        super(ResultData, self).clean_fields(exclude=exclude)


class PipelineResult(models.Model):
    dbms = models.ForeignKey(DBMSCatalog)
    hardware = models.ForeignKey(Hardware)
    creation_timestamp = models.DateTimeField()
    task_type = models.IntegerField(choices=PipelineTaskType.choices())
    value = models.TextField()

    @staticmethod
    def get_latest(dbms, hardware, task_type):
        results = PipelineResult.objects.filter(
            dbms=dbms, hardware=hardware, task_type=task_type)
        return None if len(results) == 0 else results.latest()

    class Meta:
        unique_together = ("dbms", "hardware",
                           "creation_timestamp", "task_type")
        get_latest_by = ('creation_timestamp')


# class Statistics(models.Model):
#     objects = StatsManager()
# 
#     data_result = models.ForeignKey(Result)
#     type = models.IntegerField(choices = StatsType.choices())
#     time = models.IntegerField()
#     throughput = models.FloatField()
#     avg_latency = models.FloatField()
#     min_latency = models.FloatField()
#     p25_latency = models.FloatField()
#     p50_latency = models.FloatField()
#     p75_latency = models.FloatField()
#     p90_latency = models.FloatField()
#     p95_latency = models.FloatField()
#     p99_latency = models.FloatField()
#     max_latency = models.FloatField()
