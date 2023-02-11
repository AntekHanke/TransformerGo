from typing import Tuple, Dict
import numpy as np
from scipy.stats import norm


def normal_density(x: float, mean: float = 0.0, standard_dev: float = 1.0) -> float:
    return norm.pdf(x, loc=mean, scale=standard_dev)


def TO_314_distribution(number_of_samples: int) -> np.ndarray:
    if number_of_samples > (4.0 / np.sqrt(np.pi)) * np.sqrt(np.log(100.0)):
        top: float = (4.0 / np.sqrt(np.pi)) * np.sqrt(np.log(100.0)) / number_of_samples
        mean: float = number_of_samples / 4.0
        standard_dev: float = number_of_samples / (4.0 * np.sqrt(2 * np.log(100.0)))
        prob: np.ndarray = np.array([])

        for x in range(1, number_of_samples + 1):
            if x <= np.ceil(mean):
                prob = np.append(prob, normal_density(x, mean=mean, standard_dev=standard_dev))
            else:
                prob = np.append(prob, top)
        prob = (1.0 / sum(prob)) * prob
    else:
        prob = np.array([1.0 / number_of_samples for _ in range(number_of_samples)])
    return prob


def prob_table_for_diff_n(n_range: Tuple[int, int]) -> Dict[int, np.ndarray]:
    low, hight = n_range
    assert low < hight
    prob_for_diff_n: Dict[int, np.ndarray] = {n: TO_314_distribution(n) for n in range(low, hight + 1)}
    return prob_for_diff_n
