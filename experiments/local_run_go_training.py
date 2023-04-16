from local_runner import local_run
import os

local_bindings = {
'TrainModelFromScratch.out_dir = "/plgantekhanke/models/ultra_small_policy_from_scratch"': 'TrainModelFromScratch.out_dir = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/models/ultra_small_policy_from_scratch"',
'TrainModelFromScratch.path_to_training_data = "/plgantekhanke/tokenized_data/val/train/"': 'TrainModelFromScratch.path_to_training_data = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/tokenized_data/job_run/trainparsed_data_train_part_0.pkl"',
'TrainModelFromScratch.path_to_eval_data = "/plgantekhanke/tokenized_data/val/eval/"': 'TrainModelFromScratch.path_to_eval_data = "/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/tokenized_data/job_run/evalparsed_data_eval_part_0.pkl"'
}

print(os.environ)

local_run("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/experiments/train/policy/go_ultra_small_policy_from_scratch.py", use_neptune=True
          ,local_path_bindings = local_bindings)
