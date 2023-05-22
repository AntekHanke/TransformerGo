import sente
from data_processing.goPlay.goban import ConvolutionPolicy, play_bots_match, TransformerPolicy
    
convolution_model = ConvolutionPolicy("./model_checkpoints/conv_checkpoints/checkpoint-365500")
transformer = TransformerPolicy("./model_checkpoints/transformer_checkpoints/checkpoint-192500")

save_plot_path = "data_processing/goPlay/plots"

play_bots_match(convolution_model, transformer, save_plot_path)