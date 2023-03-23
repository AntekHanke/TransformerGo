from policy_network import AlphaZeroPolicyNetwork
from policy_config  import AlphaZeroPolicyConfig
from transformers   import PreTrainedModel
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

    def forward(self, tensor : torch.Tensor, labels : torch.Tensor = None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss(tensor, labels)
            return {"loss" : loss, "logits" : logits}
        return {"logits" : logits}
