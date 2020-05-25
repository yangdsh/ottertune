#
# OtterTune - fabfile.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018

@author: bohan
'''
import glob
import json
import os
import re
import time
from collections import OrderedDict
from multiprocessing import Process

import logging
from logging.handlers import RotatingFileHandler

import requests
from fabric.api import env, lcd, local, settings, show, task, hide
from fabric.state import output as fabric_output

from utils import (file_exists, get, get_content, load_driver_conf, parse_bool,
                   put, run, run_sql_script, sudo, FabricException)

# Loads the driver config file (defaults to driver_config.py)
dconf = load_driver_conf()  # pylint: disable=invalid-name

# Fabric settings
fabric_output.update({
    'running': True,
    'stdout': True,
})
env.abort_exception = FabricException
env.hosts = [dconf.LOGIN]
env.password = dconf.LOGIN_PASSWORD

# Create local directories
for _d in (dconf.RESULT_DIR, dconf.LOG_DIR, dconf.TEMP_DIR):
    os.makedirs(_d, exist_ok=True)


# Configure logging
LOG = logging.getLogger(__name__)
LOG.setLevel(getattr(logging, dconf.LOG_LEVEL, logging.DEBUG))
Formatter = logging.Formatter(  # pylint: disable=invalid-name
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')
ConsoleHandler = logging.StreamHandler()  # pylint: disable=invalid-name
ConsoleHandler.setFormatter(Formatter)
LOG.addHandler(ConsoleHandler)
FileHandler = RotatingFileHandler(  # pylint: disable=invalid-name
    dconf.DRIVER_LOG, maxBytes=50000, backupCount=2)
FileHandler.setFormatter(Formatter)
LOG.addHandler(FileHandler)


@task
def check_disk_usage():
    partition = dconf.DATABASE_DISK
    disk_use = 0
    if partition:
        cmd = "df -h {}".format(partition)
        out = run(cmd).splitlines()[1]
        m = re.search(r'\d+(?=%)', out)
        if m:
            disk_use = int(m.group(0))
        LOG.info("Current Disk Usage: %s%s", disk_use, '%')
    return disk_use


@task
def check_memory_usage():
    run('free -m -h')


@task
def create_controller_config():
    if dconf.DB_TYPE == 'postgres':
        dburl_fmt = 'jdbc:postgresql://{host}:{port}/{db}'.format
    elif dconf.DB_TYPE == 'oracle':
        dburl_fmt = 'jdbc:oracle:thin:@{host}:{port}:{db}'.format
    elif dconf.DB_TYPE == 'mysql':
        if dconf.DB_VERSION in ['5.6', '5.7']:
            dburl_fmt = 'jdbc:mysql://{host}:{port}/{db}?useSSL=false'.format
        elif dconf.DB_VERSION == '8.0':
            dburl_fmt = ('jdbc:mysql://{host}:{port}/{db}?'
                         'allowPublicKeyRetrieval=true&useSSL=false').format
        else:
            raise Exception("MySQL Database Version {} Not Implemented !".format(dconf.DB_VERSION))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))

    config = dict(
        database_type=dconf.DB_TYPE,
        database_url=dburl_fmt(host=dconf.DB_HOST, port=dconf.DB_PORT, db=dconf.DB_NAME),
        username=dconf.DB_USER,
        password=dconf.DB_PASSWORD,
        upload_code='DEPRECATED',
        upload_url='DEPRECATED',
        workload_name=dconf.OLTPBENCH_BENCH
    )

    with open(dconf.CONTROLLER_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)


@task
def restart_database():
    if dconf.DB_TYPE == 'postgres':
        if dconf.HOST_CONN == 'docker':
            # Restarting the docker container here is the cleanest way to do it
            # becaues there's no init system running and the only process running
            # in the container is postgres itself
            local('docker restart {}'.format(dconf.CONTAINER_NAME))
        elif dconf.HOST_CONN == 'remote_docker':
            run('docker restart {}'.format(dconf.CONTAINER_NAME), remote_only=True)
        else:
            sudo('pg_ctl -D {} -w -t 600 restart -m fast'.format(
                dconf.PG_DATADIR), user=dconf.ADMIN_USER, capture=False)
    elif dconf.DB_TYPE == 'mysql':
        if dconf.HOST_CONN == 'docker':
            local('docker restart {}'.format(dconf.CONTAINER_NAME))
        elif dconf.HOST_CONN == 'remote_docker':
            run('docker restart {}'.format(dconf.CONTAINER_NAME), remote_only=True)
        else:
            sudo('service mysql restart')
    elif dconf.DB_TYPE == 'oracle':
        db_log_path = os.path.join(os.path.split(dconf.DB_CONF)[0], 'startup.log')
        local_log_path = os.path.join(dconf.LOG_DIR, 'startup.log')
        local_logs_path = os.path.join(dconf.LOG_DIR, 'startups.log')
        run_sql_script('restartOracle.sh', db_log_path)
        get(db_log_path, local_log_path)
        with open(local_log_path, 'r') as fin, open(local_logs_path, 'a') as fout:
            lines = fin.readlines()
            for line in lines:
                if line.startswith('ORACLE instance started.'):
                    return True
                if not line.startswith('SQL>'):
                    fout.write(line)
            fout.write('\n')
        return False
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    return True


@task
def drop_database():
    if dconf.DB_TYPE == 'postgres':
        run("PGPASSWORD={} dropdb -e --if-exists {} -U {} -h {}".format(
            dconf.DB_PASSWORD, dconf.DB_NAME, dconf.DB_USER, dconf.DB_HOST))
    elif dconf.DB_TYPE == 'mysql':
        run("mysql --user={} --password={} -e 'drop database if exists {}'".format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def create_database():
    if dconf.DB_TYPE == 'postgres':
        run("PGPASSWORD={} createdb -e {} -U {} -h {}".format(
            dconf.DB_PASSWORD, dconf.DB_NAME, dconf.DB_USER, dconf.DB_HOST))
    elif dconf.DB_TYPE == 'mysql':
        run("mysql --user={} --password={} -e 'create database {}'".format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def create_user():
    if dconf.DB_TYPE == 'postgres':
        sql = "CREATE USER {} SUPERUSER PASSWORD '{}';".format(dconf.DB_USER, dconf.DB_PASSWORD)
        run("PGPASSWORD={} psql -c \\\"{}\\\" -U postgres -h {}".format(
            dconf.DB_PASSWORD, sql, dconf.DB_HOST))
    elif dconf.DB_TYPE == 'oracle':
        run_sql_script('createUser.sh', dconf.DB_USER, dconf.DB_PASSWORD)
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def drop_user():
    if dconf.DB_TYPE == 'postgres':
        sql = "DROP USER IF EXISTS {};".format(dconf.DB_USER)
        run("PGPASSWORD={} psql -c \\\"{}\\\" -U postgres -h {}".format(
            dconf.DB_PASSWORD, sql, dconf.DB_HOST))
    elif dconf.DB_TYPE == 'oracle':
        run_sql_script('dropUser.sh', dconf.DB_USER)
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))

@task
def reset_conf(always=True):
    if always:
        change_conf()
        return

    # reset the config only if it has not been changed by Ottertune,
    # i.e. OtterTune signal line is not in the config file.
    signal = "# configurations recommended by ottertune:\n"
    tmp_conf_in = os.path.join(dconf.TEMP_DIR, os.path.basename(dconf.DB_CONF) + '.in')
    get(dconf.DB_CONF, tmp_conf_in)
    with open(tmp_conf_in, 'r') as f:
        lines = f.readlines()
    if signal not in lines:
        change_conf()

@task
def change_conf(next_conf=None):
    signal = "# configurations recommended by ottertune:\n"
    next_conf = next_conf or {}

    tmp_conf_in = os.path.join(dconf.TEMP_DIR, os.path.basename(dconf.DB_CONF) + '.in')
    get(dconf.DB_CONF, tmp_conf_in)
    with open(tmp_conf_in, 'r') as f:
        lines = f.readlines()

    if signal not in lines:
        lines += ['\n', signal]

    signal_idx = lines.index(signal)
    lines = lines[0:signal_idx + 1]

    if dconf.DB_TYPE == 'mysql':
        lines.append('[mysqld]\n')

    if dconf.BASE_DB_CONF:
        assert isinstance(dconf.BASE_DB_CONF, dict), \
            (type(dconf.BASE_DB_CONF), dconf.BASE_DB_CONF)
        for name, value in sorted(dconf.BASE_DB_CONF.items()):
            if value is None:
                lines.append('{}\n'.format(name))
            else:
                lines.append('{} = {}\n'.format(name, value))

    if isinstance(next_conf, str):
        with open(next_conf, 'r') as f:
            recommendation = json.load(
                f, encoding="UTF-8", object_pairs_hook=OrderedDict)['recommendation']
    else:
        recommendation = next_conf

    assert isinstance(recommendation, dict)

    for name, value in recommendation.items():
        if dconf.DB_TYPE == 'oracle' and isinstance(value, str):
            value = value.strip('B')
        # If innodb_flush_method is set to NULL on a Unix-like system,
        # the fsync option is used by default.
        if name == 'innodb_flush_method' and value == '':
            value = "fsync"
        lines.append('{} = {}\n'.format(name, value))
    lines.append('\n')

    tmp_conf_out = os.path.join(dconf.TEMP_DIR, os.path.basename(dconf.DB_CONF) + '.out')
    with open(tmp_conf_out, 'w') as f:
        f.write(''.join(lines))

    sudo('cp {0} {0}.ottertune.bak'.format(dconf.DB_CONF), remote_only=True)
    put(tmp_conf_out, dconf.DB_CONF, use_sudo=True)
    local('rm -f {} {}'.format(tmp_conf_in, tmp_conf_out))


@task
def load_oltpbench():
    if os.path.exists(dconf.OLTPBENCH_CONFIG) is False:
        msg = 'oltpbench config {} does not exist, '.format(dconf.OLTPBENCH_CONFIG)
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    set_oltpbench_config()
    cmd = "./oltpbenchmark -b {} -c {} --create=true --load=true".\
          format(dconf.OLTPBENCH_BENCH, dconf.OLTPBENCH_CONFIG)
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench():
    if os.path.exists(dconf.OLTPBENCH_CONFIG) is False:
        msg = 'oltpbench config {} does not exist, '.format(dconf.OLTPBENCH_CONFIG)
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    set_oltpbench_config()
    cmd = "./oltpbenchmark -b {} -c {} --execute=true -s 5 -o outputfile".\
          format(dconf.OLTPBENCH_BENCH, dconf.OLTPBENCH_CONFIG)
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench_bg():
    if os.path.exists(dconf.OLTPBENCH_CONFIG) is False:
        msg = 'oltpbench config {} does not exist, '.format(dconf.OLTPBENCH_CONFIG)
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    # set oltpbench config, including db username, password, url
    set_oltpbench_config()
    cmd = "./oltpbenchmark -b {} -c {} --execute=true -s 5 -o outputfile > {} 2>&1 &".\
          format(dconf.OLTPBENCH_BENCH, dconf.OLTPBENCH_CONFIG, dconf.OLTPBENCH_LOG)
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_controller():
    LOG.info('Controller config path: %s', dconf.CONTROLLER_CONFIG)
    create_controller_config()
    cmd = 'gradle run -PappArgs="-c {} -d output/" --no-daemon > {}'.\
          format(dconf.CONTROLLER_CONFIG, dconf.CONTROLLER_LOG)
    with lcd(dconf.CONTROLLER_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def signal_controller():
    pidfile = os.path.join(dconf.CONTROLLER_HOME, 'pid.txt')
    with open(pidfile, 'r') as f:
        pid = int(f.read())
    cmd = 'kill -2 {}'.format(pid)
    with lcd(dconf.CONTROLLER_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def save_dbms_result():
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    if dconf.ENABLE_UDM:
        files.append('user_defined_metrics.json')
    for f_ in files:
        srcfile = os.path.join(dconf.CONTROLLER_HOME, 'output', f_)
        dstfile = os.path.join(dconf.RESULT_DIR, '{}__{}'.format(t, f_))
        local('cp {} {}'.format(srcfile, dstfile))
    return t


@task
def save_next_config(next_config, t=None):
    if not t:
        t = int(time.time())
    with open(os.path.join(dconf.RESULT_DIR, '{}__next_config.json'.format(t)), 'w') as f:
        json.dump(next_config, f, indent=2)
    return t


@task
def free_cache():
    if dconf.HOST_CONN not in ['docker', 'remote_docker']:
        with show('everything'), settings(warn_only=True):  # pylint: disable=not-context-manager
            res = sudo("sh -c \"echo 3 > /proc/sys/vm/drop_caches\"")
            if res.failed:
                LOG.error('%s (return code %s)', res.stderr.strip(), res.return_code)
    else:
        res = sudo("sh -c \"echo 3 > /proc/sys/vm/drop_caches\"", remote_only=True)


@task
def upload_result(result_dir=None, prefix=None, upload_code=None):
    result_dir = result_dir or os.path.join(dconf.CONTROLLER_HOME, 'output')
    prefix = prefix or ''
    upload_code = upload_code or dconf.UPLOAD_CODE
    files = {}
    bases = ['summary', 'knobs', 'metrics_before', 'metrics_after']
    if dconf.ENABLE_UDM:
        bases.append('user_defined_metrics')
    for base in bases:
        fpath = os.path.join(result_dir, prefix + base + '.json')

        # Replaces the true db version with the specified version to allow for
        # testing versions not officially supported by OtterTune
        if base == 'summary' and dconf.OVERRIDE_DB_VERSION:
            with open(fpath, 'r') as f:
                summary = json.load(f)
            summary['real_database_version'] = summary['database_version']
            summary['database_version'] = dconf.OVERRIDE_DB_VERSION
            with open(fpath, 'w') as f:
                json.dump(summary, f, indent=1)

        files[base] = open(fpath, 'rb')

    response = requests.post(dconf.WEBSITE_URL + '/new_result/', files=files,
                             data={'upload_code': upload_code})
    if response.status_code != 200:
        raise Exception('Error uploading result.\nStatus: {}\nMessage: {}\n'.format(
            response.status_code, get_content(response)))

    for f in files.values():  # pylint: disable=not-an-iterable
        f.close()

    LOG.info(get_content(response))

    return response


@task
def get_result(max_time_sec=180, interval_sec=5, upload_code=None):
    max_time_sec = int(max_time_sec)
    interval_sec = int(interval_sec)
    upload_code = upload_code or dconf.UPLOAD_CODE
    url = dconf.WEBSITE_URL + '/query_and_get/' + upload_code
    elapsed = 0
    response_dict = None
    rout = ''

    while elapsed <= max_time_sec:
        rsp = requests.get(url)
        response = get_content(rsp)
        assert response != 'null'
        rout = json.dumps(response, indent=4) if isinstance(response, dict) else response

        LOG.debug('%s\n\n[status code: %d, type(response): %s, elapsed: %ds, %s]', rout,
                  rsp.status_code, type(response), elapsed,
                  ', '.join(['{}: {}'.format(k, v) for k, v in rsp.headers.items()]))

        if rsp.status_code == 200:
            # Success
            response_dict = response
            break

        elif rsp.status_code == 202:
            # Not ready
            time.sleep(interval_sec)
            elapsed += interval_sec

        elif rsp.status_code == 400:
            # Failure
            raise Exception(
                "Failed to download the next config.\nStatus code: {}\nMessage: {}\n".format(
                    rsp.status_code, rout))

        elif rsp.status_code == 500:
            # Failure
            msg = rout
            if isinstance(response, str):
                savepath = os.path.join(dconf.LOG_DIR, 'error.html')
                with open(savepath, 'w') as f:
                    f.write(response)
                msg = "Saved HTML error to '{}'.".format(os.path.relpath(savepath))
            raise Exception(
                "Failed to download the next config.\nStatus code: {}\nMessage: {}\n".format(
                    rsp.status_code, msg))

        else:
            raise NotImplementedError(
                "Unhandled status code: '{}'.\nMessage: {}".format(rsp.status_code, rout))

    if not response_dict:
        assert elapsed > max_time_sec, \
            'response={} but elapsed={}s <= max_time={}s'.format(
                rout, elapsed, max_time_sec)
        raise Exception(
            'Failed to download the next config in {}s: {} (elapsed: {}s)'.format(
                max_time_sec, rout, elapsed))

    LOG.info('Downloaded the next config in %ds', elapsed)

    return response_dict


@task
def download_debug_info(pprint=False):
    pprint = parse_bool(pprint)
    url = '{}/dump/{}'.format(dconf.WEBSITE_URL, dconf.UPLOAD_CODE)
    params = {'pp': int(True)} if pprint else {}
    rsp = requests.get(url, params=params)

    if rsp.status_code != 200:
        raise Exception('Error downloading debug info.')

    filename = rsp.headers.get('Content-Disposition').split('=')[-1]
    file_len, exp_len = len(rsp.content), int(rsp.headers.get('Content-Length'))
    assert file_len == exp_len, 'File {}: content length != expected length: {} != {}'.format(
        filename, file_len, exp_len)

    with open(filename, 'wb') as f:
        f.write(rsp.content)
    LOG.info('Downloaded debug info to %s', filename)

    return filename


@task
def add_udm(result_dir=None):
    result_dir = result_dir or os.path.join(dconf.CONTROLLER_HOME, 'output')
    with lcd(dconf.UDM_DIR):  # pylint: disable=not-context-manager
        local('python3 user_defined_metrics.py {}'.format(result_dir))


@task
def upload_batch(result_dir=None, sort=True, upload_code=None):
    result_dir = result_dir or dconf.RESULT_DIR
    sort = parse_bool(sort)
    results = glob.glob(os.path.join(result_dir, '*__summary.json'))
    if sort:
        results = sorted(results)
    count = len(results)

    LOG.info('Uploading %d samples from %s...', count, result_dir)
    for i, result in enumerate(results):
        prefix = os.path.basename(result)
        prefix_len = os.path.basename(result).find('_') + 2
        prefix = prefix[:prefix_len]
        upload_result(result_dir=result_dir, prefix=prefix, upload_code=upload_code)
        LOG.info('Uploaded result %d/%d: %s__*.json', i + 1, count, prefix)

@task
def dump_database():
    dumpfile = os.path.join(dconf.DB_DUMP_DIR, dconf.DB_NAME + '.dump')
    if dconf.DB_TYPE == 'oracle':
        if not dconf.ORACLE_FLASH_BACK and file_exists(dumpfile):
            LOG.info('%s already exists ! ', dumpfile)
            return False
    else:
        if file_exists(dumpfile):
            LOG.info('%s already exists ! ', dumpfile)
            return False

    if dconf.DB_TYPE == 'oracle' and dconf.ORACLE_FLASH_BACK:
        LOG.info('create restore point %s for database %s in %s', dconf.RESTORE_POINT,
                 dconf.DB_NAME, dconf.RECOVERY_FILE_DEST)
    else:
        LOG.info('Dump database %s to %s', dconf.DB_NAME, dumpfile)

    if dconf.DB_TYPE == 'oracle':
        if dconf.ORACLE_FLASH_BACK:
            run_sql_script('createRestore.sh', dconf.RESTORE_POINT,
                           dconf.RECOVERY_FILE_DEST_SIZE, dconf.RECOVERY_FILE_DEST)
        else:
            run_sql_script('dumpOracle.sh', dconf.DB_USER, dconf.DB_PASSWORD,
                           dconf.DB_NAME, dconf.DB_DUMP_DIR)

    elif dconf.DB_TYPE == 'postgres':
        run('PGPASSWORD={} pg_dump -U {} -h {} -F c -d {} > {}'.format(
            dconf.DB_PASSWORD, dconf.DB_USER, dconf.DB_HOST, dconf.DB_NAME,
            dumpfile))
    elif dconf.DB_TYPE == 'mysql':
        sudo('mysqldump --user={} --password={} --databases {} > {}'.format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME, dumpfile))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    return True


@task
def clean_recovery():
    run_sql_script('removeRestore.sh', dconf.RESTORE_POINT)
    cmds = ("""rman TARGET / <<EOF\nDELETE ARCHIVELOG ALL;\nexit\nEOF""")
    run(cmds)


@task
def restore_database():
    dumpfile = os.path.join(dconf.DB_DUMP_DIR, dconf.DB_NAME + '.dump')
    if not dconf.ORACLE_FLASH_BACK and not file_exists(dumpfile):
        raise FileNotFoundError("Database dumpfile '{}' does not exist!".format(dumpfile))

    LOG.info('Start restoring database')
    if dconf.DB_TYPE == 'oracle':
        if dconf.ORACLE_FLASH_BACK:
            run_sql_script('flashBack.sh', dconf.RESTORE_POINT)
            clean_recovery()
        else:
            drop_user()
            create_user()
            run_sql_script('restoreOracle.sh', dconf.DB_USER, dconf.DB_NAME)
    elif dconf.DB_TYPE == 'postgres':
        drop_database()
        create_database()
        run('PGPASSWORD={} pg_restore -U {} -h {} -n public -j 8 -F c -d {} {}'.format(
            dconf.DB_PASSWORD, dconf.DB_USER, dconf.DB_HOST, dconf.DB_NAME, dumpfile))
    elif dconf.DB_TYPE == 'mysql':
        run('mysql --user={} --password={} < {}'.format(dconf.DB_USER, dconf.DB_PASSWORD, dumpfile))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    LOG.info('Finish restoring database')


@task
def is_ready_db(interval_sec=10):
    if dconf.DB_TYPE == 'mysql':
        cmd_fmt = "mysql --user={} --password={} -e 'exit'".format
    else:
        LOG.info('database %s connecting function is not implemented, sleep %s seconds and return',
                 dconf.DB_TYPE, dconf.RESTART_SLEEP_SEC)
        return

    with hide('everything'), settings(warn_only=True):  # pylint: disable=not-context-manager
        while True:
            res = run(cmd_fmt(dconf.DB_USER, dconf.DB_PASSWORD))
            if res.failed:
                LOG.info('Database %s is not ready, wait for %s seconds',
                         dconf.DB_TYPE, interval_sec)
                time.sleep(interval_sec)
            else:
                LOG.info('Database %s is ready.', dconf.DB_TYPE)
                return

def _ready_to_start_oltpbench():
    ready = False
    if os.path.exists(dconf.CONTROLLER_LOG):
        with open(dconf.CONTROLLER_LOG, 'r') as f:
            content = f.read()
        ready = 'Output the process pid to' in content
    return ready


def _ready_to_start_controller():
    ready = False
    if os.path.exists(dconf.OLTPBENCH_LOG):
        with open(dconf.OLTPBENCH_LOG, 'r') as f:
            content = f.read()
        ready = 'Warmup complete, starting measurements' in content
    return ready


def _ready_to_shut_down_controller():
    pidfile = os.path.join(dconf.CONTROLLER_HOME, 'pid.txt')
    ready = False
    if os.path.exists(pidfile) and os.path.exists(dconf.OLTPBENCH_LOG):
        with open(dconf.OLTPBENCH_LOG, 'r') as f:
            content = f.read()
        if 'Failed' in content:
            m = re.search('\n.*Failed.*\n', content)
            error_msg = m.group(0)
            LOG.error('OLTPBench Failed!')
            return True, error_msg
        ready = 'Output throughput samples into file' in content
    return ready, None


def clean_logs():
    # remove oltpbench and controller log files
    local('rm -f {} {}'.format(dconf.OLTPBENCH_LOG, dconf.CONTROLLER_LOG))


@task
def clean_oltpbench_results():
    # remove oltpbench result files
    local('rm -f {}/results/*'.format(dconf.OLTPBENCH_HOME))


def _set_oltpbench_property(name, line):
    if name == 'username':
        ss = line.split('username')
        new_line = ss[0] + 'username>{}</username'.format(dconf.DB_USER) + ss[-1]
    elif name == 'password':
        ss = line.split('password')
        new_line = ss[0] + 'password>{}</password'.format(dconf.DB_PASSWORD) + ss[-1]
    elif name == 'DBUrl':
        ss = line.split('DBUrl')
        if dconf.DB_TYPE == 'postgres':
            dburl_fmt = 'jdbc:postgresql://{host}:{port}/{db}'.format
        elif dconf.DB_TYPE == 'oracle':
            dburl_fmt = 'jdbc:oracle:thin:@{host}:{port}:{db}'.format
        elif dconf.DB_TYPE == 'mysql':
            if dconf.DB_VERSION in ['5.6', '5.7']:
                dburl_fmt = 'jdbc:mysql://{host}:{port}/{db}?useSSL=false'.format
            elif dconf.DB_VERSION == '8.0':
                dburl_fmt = ('jdbc:mysql://{host}:{port}/{db}?'
                             'allowPublicKeyRetrieval=true&amp;useSSL=false').format
            else:
                raise Exception("MySQL Database Version {} "
                                "Not Implemented !".format(dconf.DB_VERSION))
        else:
            raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
        database_url = dburl_fmt(host=dconf.DB_HOST, port=dconf.DB_PORT, db=dconf.DB_NAME)
        new_line = ss[0] + 'DBUrl>{}</DBUrl'.format(database_url) + ss[-1]
    else:
        raise Exception("OLTPBench Config Property {} Not Implemented !".format(name))
    return new_line


@task
def set_oltpbench_config():
    # set database user, password, and connection url in oltpbench config
    lines = None
    text = None
    with open(dconf.OLTPBENCH_CONFIG, 'r') as f:
        lines = f.readlines()
        text = ''.join(lines)
    lid = 10
    for i, line in enumerate(lines):
        if 'dbtype' in line:
            dbtype = line.split('dbtype')[1][1:-2].strip()
            if dbtype != dconf.DB_TYPE:
                raise Exception("dbtype {} in OLTPBench config != DB_TYPE {}"
                                "in driver config !".format(dbtype, dconf.DB_TYPE))
        if 'username' in line:
            lines[i] = _set_oltpbench_property('username', line)
        elif 'password' in line:
            lines[i] = _set_oltpbench_property('password', line)
            lid = i + 1
        elif 'DBUrl' in line:
            lines[i] = _set_oltpbench_property('DBUrl', line)
    if dconf.ENABLE_UDM:
        # add the empty uploadCode and uploadUrl so that OLTPBench will output the summary file,
        # which contains throughput, 99latency, 95latency, etc.
        if 'uploadUrl' not in text:
            line = '    <uploadUrl></uploadUrl>\n'
            lines.insert(lid, line)
        if 'uploadCode' not in text:
            line = '    <uploadCode></uploadCode>\n'
            lines.insert(lid, line)
    text = ''.join(lines)
    with open(dconf.OLTPBENCH_CONFIG, 'w') as f:
        f.write(text)
    LOG.info('oltpbench config is set: %s', dconf.OLTPBENCH_CONFIG)


@task
def loop(i):
    i = int(i)

    # free cache
    free_cache()

    # remove oltpbench log and controller log
    clean_logs()

    if dconf.ENABLE_UDM is True:
        clean_oltpbench_results()

    # check disk usage
    if check_disk_usage() > dconf.MAX_DISK_USAGE:
        LOG.warning('Exceeds max disk usage %s', dconf.MAX_DISK_USAGE)

    # run controller from another process
    p = Process(target=run_controller, args=())
    p.start()
    LOG.info('Run the controller')

    # run oltpbench as a background job
    while not _ready_to_start_oltpbench():
        time.sleep(1)
    run_oltpbench_bg()
    LOG.info('Run OLTP-Bench')

    # the controller starts the first collection
    while not _ready_to_start_controller():
        time.sleep(1)
    signal_controller()
    LOG.info('Start the first collection')

    # stop the experiment
    ready_to_shut_down = False
    error_msg = None
    while not ready_to_shut_down:
        ready_to_shut_down, error_msg = _ready_to_shut_down_controller()
        time.sleep(1)

    signal_controller()
    LOG.info('Start the second collection, shut down the controller')

    p.join()
    if error_msg:
        raise Exception('OLTPBench Failed: ' + error_msg)
    # add user defined metrics
    if dconf.ENABLE_UDM is True:
        add_udm()

    # save result
    result_timestamp = save_dbms_result()

    if i >= dconf.WARMUP_ITERATIONS:
        # upload result
        upload_result()

        # get result
        response = get_result()

        # save next config
        save_next_config(response, t=result_timestamp)

        # change config
        change_conf(response['recommendation'])

@task
def run_loops(max_iter=10):
    # dump database if it's not done before.
    dump = dump_database()
    # put the BASE_DB_CONF in the config file
    # e.g., mysql needs to set innodb_monitor_enable to track innodb metrics
    reset_conf(False)
    for i in range(int(max_iter)):
        # restart database
        restart_succeeded = restart_database()
        if not restart_succeeded:
            files = {'summary': b'{"error": "DB_RESTART_ERROR"}',
                     'knobs': b'{}',
                     'metrics_before': b'{}',
                     'metrics_after': b'{}'}
            if dconf.ENABLE_UDM:
                files['user_defined_metrics'] = b'{}'
            response = requests.post(dconf.WEBSITE_URL + '/new_result/', files=files,
                                     data={'upload_code': dconf.UPLOAD_CODE})
            response = get_result()
            result_timestamp = int(time.time())
            save_next_config(response, t=result_timestamp)
            change_conf(response['recommendation'])
            continue

        # reload database periodically
        if dconf.RELOAD_INTERVAL > 0:
            if i % dconf.RELOAD_INTERVAL == 0:
                is_ready_db(interval_sec=10)
                if i == 0 and dump is False:
                    restore_database()
                elif i > 0:
                    restore_database()
        LOG.info('Wait %s seconds after restarting database', dconf.RESTART_SLEEP_SEC)
        is_ready_db(interval_sec=10)
        LOG.info('The %s-th Loop Starts / Total Loops %s', i + 1, max_iter)
        loop(i % dconf.RELOAD_INTERVAL if dconf.RELOAD_INTERVAL > 0 else i)
        LOG.info('The %s-th Loop Ends / Total Loops %s', i + 1, max_iter)


@task
def rename_batch(result_dir=None):
    result_dir = result_dir or dconf.RESULT_DIR
    results = glob.glob(os.path.join(result_dir, '*__summary.json'))
    results = sorted(results)
    for i, result in enumerate(results):
        prefix = os.path.basename(result)
        prefix_len = os.path.basename(result).find('_') + 2
        prefix = prefix[:prefix_len]
        new_prefix = str(i) + '__'
        bases = ['summary', 'knobs', 'metrics_before', 'metrics_after']
        if dconf.ENABLE_UDM:
            bases.append('user_defined_metrics')
        for base in bases:
            fpath = os.path.join(result_dir, prefix + base + '.json')
            rename_path = os.path.join(result_dir, new_prefix + base + '.json')
            os.rename(fpath, rename_path)


def _http_content_to_json(content):
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    try:
        json_content = json.loads(content)
        decoded = True
    except (TypeError, json.decoder.JSONDecodeError):
        json_content = None
        decoded = False

    return json_content, decoded


def _modify_website_object(obj_name, action, verbose=False, **kwargs):
    verbose = parse_bool(verbose)
    if obj_name == 'project':
        valid_actions = ('create', 'edit')
    elif obj_name == 'session':
        valid_actions = ('create', 'edit')
    elif obj_name == 'user':
        valid_actions = ('create', 'delete')
    else:
        raise ValueError('Invalid object: {}. Valid objects: project, session'.format(obj_name))

    if action not in valid_actions:
        raise ValueError('Invalid action: {}. Valid actions: {}'.format(
            action, ', '.join(valid_actions)))

    data = {}
    for k, v in kwargs.items():
        if isinstance(v, (dict, list, tuple)):
            v = json.dumps(v)
        data[k] = v

    url_path = '/{}/{}/'.format(action, obj_name)
    response = requests.post(dconf.WEBSITE_URL + url_path, data=data)

    content = response.content.decode('utf-8')
    if response.status_code != 200:
        raise Exception("Failed to {} {}.\nStatus: {}\nMessage: {}\n".format(
            action, obj_name, response.status_code, content))

    json_content, decoded = _http_content_to_json(content)
    if verbose:
        if decoded:
            LOG.info('\n%s_%s = %s', action.upper(), obj_name.upper(),
                     json.dumps(json_content, indent=4))
        else:
            LOG.warning("Content could not be decoded.\n\n%s\n", content)

    return response, json_content, decoded


@task
def create_website_user(**kwargs):
    return _modify_website_object('user', 'create', **kwargs)


@task
def delete_website_user(**kwargs):
    return _modify_website_object('user', 'delete', **kwargs)


@task
def create_website_project(**kwargs):
    return _modify_website_object('project', 'create', **kwargs)


@task
def edit_website_project(**kwargs):
    return _modify_website_object('project', 'edit', **kwargs)


@task
def create_website_session(**kwargs):
    return _modify_website_object('session', 'create', **kwargs)


@task
def edit_website_session(**kwargs):
    return _modify_website_object('session', 'edit', **kwargs)


def wait_pipeline_data_ready(max_time_sec=800, interval_sec=10):
    max_time_sec = int(max_time_sec)
    interval_sec = int(interval_sec)
    elapsed = 0
    ready = False

    while elapsed <= max_time_sec:
        response = requests.get(dconf.WEBSITE_URL + '/test/pipeline/')
        content = get_content(response)
        LOG.info("%s (elapsed: %ss)", content, elapsed)
        if 'False' in content:
            time.sleep(interval_sec)
            elapsed += interval_sec
        else:
            ready = True
            break

    return ready


@task
def integration_tests_simple():

    # Create test website
    response = requests.get(dconf.WEBSITE_URL + '/test/create/')
    LOG.info(get_content(response))

    # Upload training data
    LOG.info('Upload training data to no tuning session')
    upload_batch(result_dir='./integrationTests/data/', upload_code='ottertuneTestNoTuning')

    # periodic tasks haven't ran, lhs result returns.
    LOG.info('Test no pipeline data, LHS returned')
    upload_result(result_dir='./integrationTests/data/', prefix='0__',
                  upload_code='ottertuneTestTuningGPR')
    response = get_result(upload_code='ottertuneTestTuningGPR')
    assert response['status'] == 'lhs'

    # wait celery periodic task finishes
    assert wait_pipeline_data_ready(), "Pipeline data failed"

    # Test DNN
    LOG.info('Test DNN (deep neural network)')
    upload_result(result_dir='./integrationTests/data/', prefix='0__',
                  upload_code='ottertuneTestTuningDNN')
    response = get_result(upload_code='ottertuneTestTuningDNN')
    assert response['status'] == 'good'

    # Test GPR
    LOG.info('Test GPR (gaussian process regression)')
    upload_result(result_dir='./integrationTests/data/', prefix='0__',
                  upload_code='ottertuneTestTuningGPR')
    response = get_result(upload_code='ottertuneTestTuningGPR')
    assert response['status'] == 'good'

    # Test DDPG
    LOG.info('Test DDPG (deep deterministic policy gradient)')
    upload_result(result_dir='./integrationTests/data/', prefix='0__',
                  upload_code='ottertuneTestTuningDDPG')
    response = get_result(upload_code='ottertuneTestTuningDDPG')
    assert response['status'] == 'good'

    # Test DNN: 2rd iteration
    upload_result(result_dir='./integrationTests/data/', prefix='1__',
                  upload_code='ottertuneTestTuningDNN')
    response = get_result(upload_code='ottertuneTestTuningDNN')
    assert response['status'] == 'good'

    # Test GPR: 2rd iteration
    upload_result(result_dir='./integrationTests/data/', prefix='1__',
                  upload_code='ottertuneTestTuningGPR')
    response = get_result(upload_code='ottertuneTestTuningGPR')
    assert response['status'] == 'good'

    # Test DDPG: 2rd iteration
    upload_result(result_dir='./integrationTests/data/', prefix='1__',
                  upload_code='ottertuneTestTuningDDPG')
    response = get_result(upload_code='ottertuneTestTuningDDPG')
    assert response['status'] == 'good'

    LOG.info("\n\nIntegration Tests: PASSED!!\n")

    # Test task status UI
    task_status_ui_test()


@task
def task_status_ui_test():
    # Test GPR
    upload_code = 'ottertuneTestTuningGPR'
    response = requests.get(dconf.WEBSITE_URL + '/test/task_status/' + upload_code)
    LOG.info(get_content(response))
    assert 'Success:' in get_content(response)

    # Test DNN:
    upload_code = 'ottertuneTestTuningDNN'
    response = requests.get(dconf.WEBSITE_URL + '/test/task_status/' + upload_code)
    LOG.info(get_content(response))
    assert 'Success:' in get_content(response)

    # Test DDPG:
    upload_code = 'ottertuneTestTuningDDPG'
    response = requests.get(dconf.WEBSITE_URL + '/test/task_status/' + upload_code)
    LOG.info(get_content(response))
    assert 'Success:' in get_content(response)

    LOG.info("\n\nTask Status UI Tests: PASSED!!\n")


def simulate_db_run(i, next_conf):
    # Using 1 knob to decide performance; simple but effective
    gain = int(next_conf['effective_cache_size'].replace('kB', ''))

    with open('./integrationTests/data/x__metrics_after.json', 'r') as fin:
        metrics_after = json.load(fin)
        metrics_after['local']['database']['pg_stat_database']['tpcc']['xact_commit'] = gain
    with open('./integrationTests/data/x__metrics_after.json', 'w') as fout:
        json.dump(metrics_after, fout)
    with open('./integrationTests/data/x__knobs.json', 'r') as fin:
        knobs = json.load(fin)
        for knob in next_conf:
            knobs['global']['global'][knob] = next_conf[knob]
    with open('./integrationTests/data/x__knobs.json', 'w') as fout:
        json.dump(knobs, fout)
    with open('./integrationTests/data/x__summary.json', 'r') as fin:
        summary = json.load(fin)
        summary['start_time'] = i * 20000000
        summary['observation_time'] = 100
        summary['end_time'] = (i + 1) * 20000000
    with open('./integrationTests/data/x__summary.json', 'w') as fout:
        json.dump(summary, fout)
    return gain


@task
def integration_tests():
    # Create test website
    response = requests.get(dconf.WEBSITE_URL + '/test/create/')
    LOG.info(get_content(response))

    # Upload training data
    LOG.info('Upload training data to no tuning session')
    upload_batch(result_dir='./integrationTests/data/', upload_code='ottertuneTestNoTuning')

    # periodic tasks haven't ran, lhs result returns.
    LOG.info('Test no pipeline data, LHS returned')
    upload_result(result_dir='./integrationTests/data/', prefix='0__',
                  upload_code='ottertuneTestTuningGPR')
    response = get_result(upload_code='ottertuneTestTuningGPR')
    assert response['status'] == 'lhs'

    # wait celery periodic task finishes
    assert wait_pipeline_data_ready(), "Pipeline data failed"

    total_n = 30
    first_n = 5
    last_n = 10
    average = 0
    simulate_db_run(1, {'effective_cache_size': '0kB'})
    for i in range(2, total_n + 2):
        LOG.info('Test GPR (gaussian process regression)')
        upload_result(result_dir='./integrationTests/data/', prefix='x__',
                      upload_code='ottertuneTestTuningGPR')
        response = get_result(upload_code='ottertuneTestTuningGPR')
        assert response['status'] == 'good'
        gain = simulate_db_run(i, response['recommendation'])
        if i < first_n + 2:
            average += gain / first_n
        elif i > total_n - last_n + 2:
            assert gain > average

    average = 0
    simulate_db_run(1, {'effective_cache_size': '0kB'})
    for i in range(2, total_n + 2):
        LOG.info('Test DNN (deep neural network)')
        upload_result(result_dir='./integrationTests/data/', prefix='x__',
                      upload_code='ottertuneTestTuningDNN')
        response = get_result(upload_code='ottertuneTestTuningDNN')
        assert response['status'] == 'good'
        gain = simulate_db_run(i, response['recommendation'])
        if i < first_n + 2:
            average += gain / first_n
        elif i > total_n - last_n + 2:
            assert gain > average

    average = 0
    simulate_db_run(1, {'effective_cache_size': '0kB'})
    for i in range(2, total_n + 2):
        upload_result(result_dir='./integrationTests/data/', prefix='x__',
                      upload_code='ottertuneTestTuningDDPG')
        response = get_result(upload_code='ottertuneTestTuningDDPG')
        assert response['status'] == 'good'
        gain = simulate_db_run(i, response['recommendation'])
        if i < first_n + 2:
            average += gain / first_n
        elif i > total_n - last_n + 2:
            assert gain > average
