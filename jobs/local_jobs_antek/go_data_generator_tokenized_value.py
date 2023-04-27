from typing import Type

from jobs.core import Job
from data_processing.go_data_generator import GoValueTokenized

class GoTokenizedValueGenerator(Job):
    def __init__(self, GoTokenizedDataGenerator: Type[GoValueTokenized]):
        self.GoTokenizedDataGenerator = GoTokenizedDataGenerator()

    def execute(self):
        self.GoTokenizedDataGenerator.create_data()
