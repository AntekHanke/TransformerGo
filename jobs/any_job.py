import gin

from generator_train_entropy import train_generator_entropy
from jobs.core import Job
from policy_train_eagle import train_policy_eagle
from generator_train_eagle import train_generator_eagle
# from quality_subgoals_data_eagle import generate_quality_database


class AnyJob(Job):
    """Job used for quick prototyping. You can put anything you want here."""
    def __init__(self, learning_rate, k, n_datapoints):
        self.learning_rate = learning_rate
        self.k = k
        self.n_datapoints = n_datapoints

    def execute(self):
        pass
        # generate_quality_database()
        # train_generator_entropy(learning_rate=self.learning_rate, k=self.k, n_datapoints=self.n_datapoints)
        train_generator_eagle(learning_rate=self.learning_rate, k=self.k)