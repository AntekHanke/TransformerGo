from go_policy.policy_network import AlphaZeroPolicyNetwork
from go_policy.policy_config  import AlphaZeroPolicyConfig
from transformers import PreTrainedModel
import torch

class   AlphaZeroPolicyModel(PreTrainedModel):
    config_class = AlphaZeroPolicyConfig


    def __init__(self, config):
        super().__init__(config)
        
        self.model = AlphaZeroPolicyNetwork(
            num_residual_blocks = config.num_residual_blocks,
            num_in_channels = config.num_in_channels,
            num_out_channels = config.num_out_channels,
            kernel_size = config.kernel_size,
            stride = config.stride,
            board_size = config.board_size
        )

    def forward(self, input_ids : torch.Tensor, labels : torch.Tensor = None):
        logits = self.model(input_ids.permute(0,3,1,2).float()) # batch, channels, board_size, board_size 
        if labels is not None:
            labels[:, 0] *= 19
            labels = torch.sum(labels, dim = -1).flatten()
            loss = torch.nn.functional.cross_entropy(logits, torch.nn.functional.one_hot(
                                                labels, num_classes = 19*19 + 20).float())
            return {"loss" : loss, "logits" : logits}
        return {"logits" : logits   }
