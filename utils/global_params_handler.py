from typing import Union, List

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
        path_type: str = "full_info",
        path_format: List[str] = None,
    ):
        """All parameters can be set here by gin"""
        assert path_type in ["full_info", "raw_path"], "Path type must be either full_info or raw_path"

        self.k = k
        self.learning_rate = learning_rate
        self.data_location = data_location
        self.out_dir = out_dir
        self.path_type = path_type
        self.path_format = path_format

    def get_data_path(self):
        if self.path_type == "full_info":
            """This is the path to the pickle file containing the data"""
            if self.k is not None:
                if isinstance(self.k, list):
                    return [self.data_location + f"_k={i}.pkl" for i in self.k]
                else:
                    return self.data_location + f"_k={self.k}.pkl"
        if self.path_type == "raw_path":
            return self.data_location

    def get_out_dir(self):
        if self.path_format is not None:
            for param_name in self.path_format:
                self.out_dir += f"/{self.insert_path_element(param_name)}"
        return self.out_dir

    def insert_path_element(self, param_name):
        if param_name == "k":
            return f"_k={self.k}"
        elif param_name == "learning_rate":
            return f"_lr={self.learning_rate}"
