import os
from typing import Union, List, Optional, Tuple

import gin


@gin.configurable
class GlobalParamsHandler:
    """Some parameters affect many parts of the code. This class is a
    centralized place to store and access those parameters. This is also used to perform simple grid-search by mrunner.
    Parameters which are set here, override each class gin specification.
    """

    def __init__(
        self,
        k: Union[int, List[int]] = None,
        learning_rate: float = None,
        data_location: str = None,
        out_dir: str = None,
        path_type: Optional[str] = None,
        path_format: List[str] = None,
    ):

        self.k = k
        self.learning_rate = learning_rate
        self.data_location = data_location
        self.out_dir = out_dir
        self.path_type = path_type
        self.path_format = path_format

        self.out_dir_generated = False

    def get_data_path(self) -> Optional[Tuple[str, str]]:
        if self.data_location is None or self.path_type is None:
            return None
        if self.path_type == "generator":
            assert self.k is not None, (
                "Please choose kind of subgolas You want to use." "Available options: 1, 2, 3, 4, 5, 6"
            )

            path_to_train_dataset: Optional[str] = None
            path_to_eval_dataset: Optional[str] = None

            for folder_name in os.listdir(self.data_location):
                if folder_name == "subgoals_data_train":
                    path_to_train_dataset = self.data_location + "/" + folder_name + "/" + "subgoals_k=" + str(self.k)
                elif folder_name == "subgoals_data_eval":
                    path_to_eval_dataset = self.data_location + "/" + folder_name + "/" + "subgoals_k=" + str(self.k)
                else:
                    continue

            assert path_to_train_dataset and path_to_eval_dataset is not None, (
                "No folders:" "subgoals_data_train and" "subgoals_data_eval "
            )
            return path_to_train_dataset, path_to_eval_dataset

        elif self.path_type == "policy":
            # TODO return paths for both train and eval
            return self.data_location

    def get_out_dir(self):
        if not self.out_dir_generated:
            if self.path_format is not None:
                for param_name in self.path_format:
                    self.out_dir += f"/{self.insert_path_element(param_name)}"
            self.out_dir_generated = True
        return self.out_dir

    def insert_path_element(self, param_name):
        if param_name == "k":
            return f"_k={self.k}"
        elif param_name == "learning_rate":
            return f"_lr={self.learning_rate}"
