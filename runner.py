import argparse
import platform

import torch


import gin
# This makes gin configurable classes picklable
# gin.configs._OPERATIVE_CONFIG_LOCK = dask.SerializableLock()

import metric_logging

import configs.gin_configurable_classes #keep this import
from configs.global_config import NEPTUNE_PROJECT


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', action='append', default=[],
        help='Gin configs files.'
    )
    parser.add_argument(
        '--config', action='append', default=[],
        help='Gin configs overrides.'
    )
    parser.add_argument(
        '--mrunner', action='store_true',
        help='Add mrunner spec to gin-configs overrides and Neptune to loggers.'
        '\nNOTE: It assumes that the last configs override (--configs argument) '
        'is a path to a pickled experiment configs created by the mrunner CLI or'
        'a mrunner specification file.'
    )
    return parser.parse_args()

@gin.configurable()
def run(job_class):
    metric_logging.log_object('host_name', platform.node())
    metric_logging.log_object('n_gpus', str(torch.cuda.device_count()))

    job = job_class()
    return job.execute()

if __name__ == '__main__':
    args = _parse_args()

    gin_bindings = args.config
    if args.mrunner:
        from mrunner_utils import mrunner_client
        spec_path = gin_bindings.pop()
        specification, overrides = mrunner_client.get_configuration(spec_path)
        gin_bindings.extend(overrides)

        print(f'specification: {specification}')
        if 'use_neptune' in specification['parameters']:
            if specification['parameters']['use_neptune']:
                print(f'Creating neptune logger with project {NEPTUNE_PROJECT}')
                try:
                    neptune_logger = mrunner_client.configure_neptune(specification)
                    metric_logging.register_logger(neptune_logger)
                    metric_logging.register_pytorch_callback_logger(neptune_logger)

                except mrunner_client.NeptuneAPITokenException:
                    print('HINT: To run with Neptune logging please set your '
                          'NEPTUNE_API_TOKEN environment variable')

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    run()
