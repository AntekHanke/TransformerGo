"""Metric logging."""
import numpy as np
from mrunner_utils.source_files_register import SourceFilesRegister

source_files_register = SourceFilesRegister()


class StdoutLogger:
    """Logs to standard output."""

    @staticmethod
    def log_value(name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        #   2137 | loss:                      1.0e-5
        if 0 < value < 1e-2:
            print("{:>6} | {:64}{:>9.1e}".format(step, name + ":", value))
        else:
            print("{:>6} | {:64}{:>9.3f}".format(step, name + ":", value))

    @staticmethod
    def log_value_without_step(name, value):
        if 0 < value < 1e-2:
            print("{:64}{:>9.1e}".format(name + ":", value))
        else:
            print("{:64}{:>9.3f}".format(name + ":", value))

    @staticmethod
    def log_object(name, value):
        # Not supported in this logger.
        pass

    @staticmethod
    def log_param(name, value):
        print(f"{name} = {value}")


_loggers = [StdoutLogger]


def register_logger(logger):
    """Adds a logger to log to."""
    _loggers.append(logger)


def log_value(name, step, value):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_value(name, step, value)


def log_value_without_step(name, value):
    """Logs a scalar to the loggers."""
    for logger in _loggers:
        logger.log_value_without_step(name, value)


def log_object(name, object):
    for logger in _loggers:
        logger.log_object(name, object)


def log_param(name, value):
    for logger in _loggers:
        logger.log_param(name, value)


def compute_scalar_statistics(x, prefix=None, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x (np.ndarray): Samples of the scalar to produce statistics for.
        prefix (str): Prefix to put before a statistic name, separated with
            an underscore.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.

    Returns:
        Dictionary with statistic names as keys (can be prefixed, see the prefix
        argument) and statistic values.
    """
    prefix = prefix + "_" if prefix else ""
    stats = {}

    stats[prefix + "mean"] = np.mean(x)
    stats[prefix + "std"] = np.std(x)
    if with_min_and_max:
        stats[prefix + "min"] = np.min(x)
        stats[prefix + "max"] = np.max(x)

    return stats


class MetricsAccumulator:
    def __init__(self):
        self._metrics = {}
        self._data_to_average = {}
        self._data_to_accumulate = {}

    def log_metric_to_average(self, name, value):
        self._data_to_average.setdefault(name, []).append(value)
        self._metrics[name] = np.mean(self._data_to_average[name])

    def log_metric_to_accumulate(self, name, value):
        self._data_to_average.setdefault(name, []).append(value)
        self._metrics[name] = np.sum(self._data_to_average[name])

    def return_scalars(self):
        return self._metrics

    def get_value(self, name):
        return self._metrics[name]
