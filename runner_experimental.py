import platform

import gin
import torch

import metric_logging
from policy_train_eagle import train_policy_eagle


@gin.configurable()
def run_job(job_class):
    metric_logging.log_object('host_name', platform.node())
    metric_logging.log_object('n_gpus', str(torch.cuda.device_count()))
    job_class().execute()

def run_function(func):
    metric_logging.log_object('host_name', platform.node())
    metric_logging.log_object('n_gpus', str(torch.cuda.device_count()))
    func()

if __name__ == '__main__':
    run_function(train_policy_eagle)