import gin

from jobs.core import Job
from policy_train_eagle import train_policy_eagle
from generator_train_eagle import train_generator_eagle


class AnyJob(Job):
    """Job used for quick prototyping. You can put anything you want here."""
    def __init__(self, learning_rate, k):
        self.learning_rate = learning_rate
        self.k = k

    def execute(self):
        train_generator_eagle(learning_rate=self.learning_rate, k=self.k)