from typing import Union, List


class GlobalParamsHandler:
    """Some parameters affect many parts of the code. This class is a
    centralized place to store and access those parameters. This is also used to perform simple grid-search by mrunner.
    """

    def __init__(self, k: Union[int, List[int]] = None, data_location: str = None, out_dir: str = None, data_type: str = "leela"):
        """All parameters can be set here by gin"""
        assert data_type in ["leela", "pgn"], "data_type must be either leela or pgn"

        self.k = k
        self.data_location = data_location
        self.out_dir = out_dir
        self.data_type = data_type

    def get_data_path(self):
        if self.data_type == "leela":
            """This is the path to the pickle file containing the data"""
            if self.k is not None:
                if isinstance(self.k, list):
                    return [self.data_location + f"_k={i}.pkl" for i in self.k]
            else:
                return self.data_location + f"_k={self.k}.pkl"

    def get_out_dir(self):
        if self.k is not None:
            if isinstance(self.k, list):
                return self.out_dir
        else:
            return self.out_dir + f"_k={self.k}"

