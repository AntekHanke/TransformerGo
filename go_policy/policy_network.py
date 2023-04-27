import torch
import torch.nn as nn
from typing import Tuple

class AlphaZeroPolicyNetwork(nn.Module):
    "AlphaZero dual-res policy network architecture, a convolutional block followed by a tower of blocks with resuidual connections"
    "As input it expects a batch of boards of shape (B, 4, BOARD_SIZEH, BOARD_SIZEW)"
    "outputs logits of move probability of shape (B, BOARD_SIZEH*BOARD_SIZEW + 1)"

    def __init__(
        self, 
        num_residual_blocks : int = 19, 
        num_in_channels : int = 4,
        num_out_channels : int = 256, #Channels used in the inner convolutions
        kernel_size : int = 3,
        stride : int = 1,
        board_size : Tuple[int, int] = (19,19),
        *args, 
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_residual_blocks = num_residual_blocks
        self.kernel_size = kernel_size
        self.stride = stride
        self.board_size = board_size

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels = self.num_in_channels, out_channels= self.num_out_channels, \
                                                  kernel_size= self.kernel_size, stride = self.stride, padding = 'same'),
                                          nn.BatchNorm2d(self.num_out_channels),
                                          nn.ReLU())
        
        self.residual_tower = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels = self.num_out_channels, out_channels=self.num_out_channels, kernel_size = self.kernel_size, \
                            stride = 1, padding = 'same'),
                            nn.BatchNorm2d(self.num_out_channels),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=self.num_out_channels, out_channels=self.num_out_channels, \
                                      kernel_size= self.kernel_size, stride = self.stride, padding = 'same'),
                            nn.BatchNorm2d(self.num_out_channels)) for i in range(self.num_residual_blocks)])
        
        self.conv_final = nn.Sequential(nn.Conv2d(in_channels = self.num_out_channels, out_channels = 2, kernel_size = 1, stride = 1),
                                          nn.BatchNorm2d(2),
                                          nn.ReLU())

        self.policy_head = nn.Linear(self.board_size[0]* self.board_size[1] * 2, self.board_size[0] * self.board_size[1] + 1)

        
    def forward(self, x : torch.Tensor):
        
        x = self.conv_block(x)

        prev_input = x
        for res in self.residual_tower:
            x = res(x)
            x = x + prev_input 
            x = nn.functional.relu(x)
            prev_input = x

        x = self.conv_final(x)
        

        return self.policy_head(x.flatten(start_dim=1))




