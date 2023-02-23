from typing import Type

from jobs.core import Job
from data_processing.go_data_generator import GoSimpleGamesDataGeneratorTokenizedAlwaysBlack

class GoTokenizedPolicyGeneratorAlwaysBlack(Job):
    def __init__(self, GoTokenizedDataGenerator: Type[GoSimpleGamesDataGeneratorTokenizedAlwaysBlack]):
        self.GoTokenizedDataGenerator = GoTokenizedDataGenerator()

    def execute(self):
        self.GoTokenizedDataGenerator.create_data()
