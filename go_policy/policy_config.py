from transformers import PretrainedConfig
from typing import Tuple


class AlphaZeroPolicyConfig(PretrainedConfig):

    model_type = "AlphaZeroPolicy"

    def __init__(
        self,
        num_residual_blocks : int = 19, 
        num_in_channels : int = 4,
        num_out_channels : int = 256, 
        kernel_size : int = 3,
        stride : int = 1,
        board_size : Tuple[int, int] = (19,19),
        **kwargs,
    ):
        
        
        self.num_residual_blocks = num_residual_blocks
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.board_size = board_size

        super().__init__(**kwargs)



    

            