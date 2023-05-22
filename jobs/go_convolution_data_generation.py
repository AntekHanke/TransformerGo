from typing import Type

from jobs.core import Job
from data_processing.go_data_generator import SimpleGamesDataGenerator

class GoConvolutionDataGeneration(Job):
    def __init__(self, GoConvolutionDataGenerator: Type[SimpleGamesDataGenerator]):
        self.GoConvolutionDataGenerator = GoConvolutionDataGenerator()

    def execute(self):
        self.GoConvolutionDataGenerator.create_data()
