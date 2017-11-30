import itertools
import numpy as np
import os.path
from collections import OrderedDict

from celery.task import task, Task
from django.utils.timezone import now
from djcelery.models import TaskMeta
from sklearn.preprocessing import StandardScaler

from analysis.gp_tf import GPR, GPR_GD
from analysis.preprocessing import bin_by_decile, Bin
from website.models import (DBMSCatalog, Hardware, KnobCatalog, PipelineResult,
                            Result, ResultData, WorkloadCluster)
from website.settings import PIPELINE_DIR
from website.types import KnobUnitType, PipelineTaskType, VarType
from website.utils import (ConversionUtil, DataUtil, DBMSUtil, JSONUtil,
                           MediaUtil, PostgresUtilImpl)


from celery.task.schedules import crontab
from celery.decorators import periodic_task
from datetime import datetime
from celery.utils.log import get_task_logger




## periodic task example 
logger = get_task_logger(__name__)

#every 10 seconds
@periodic_task(run_every=10,name="periodic_task_example")

#every 1 min 
#@periodic_task(run_every=(crontab(hour=0,minute=1,day_of_week=0)),name="periodic_task_example")
def periodic_task_example():
    
   logger.info("This is a periodic task example. CHANGE ME !! " )

