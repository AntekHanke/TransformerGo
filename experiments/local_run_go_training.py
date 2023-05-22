from local_runner import local_run
import os

local_bindings = {
'TrainModelFromScratch.out_dir = "/plgantekhanke/models/ultra_small_generator_from_scratch"': 'TrainModelFromScratch.out_dir = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/models/ultra_small_generator_from_scratch"',
'TrainModelFromScratch.path_to_training_data = "/plgantekhanke/tokenized_data/val/train/"': 'TrainModelFromScratch.path_to_training_data = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/data_processing/tokenized_data/subgoal/test/train/info_2005-11_train_part_0.pkl"',
'TrainModelFromScratch.path_to_eval_data = "/plgantekhanke/tokenized_data/val/eval/"': 'TrainModelFromScratch.path_to_eval_data = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/data_processing/tokenized_data/subgoal/test/train/info_2005-11_train_part_0.pkl"'
}

print(os.environ)

local_run("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/experiments/train/generator/go_ultra_small_generator_from_scratch.py", use_neptune=True
          ,local_path_bindings = local_bindings)
