import sente
from data_processing.goPlay.goban import ConvolutionPolicy, play_bots_match, TransformerPolicy
    
convolution_model = ConvolutionPolicy("./model_checkpoints/conv_checkpoints/checkpoint-351500", history = True)
#convolution_model_basic = ConvolutionPolicy("./model_checkpoints/conv_checkpoints/checkpoint-365500")
transformer = TransformerPolicy("./model_checkpoints/transformer_checkpoints/checkpoint-192500")

save_plot_path = "data_processing/goPlay/plots/conv_history_white_transformer_black"
save_sgf_path = "data_processing/goPlay/sgfs/conv_history_white_transformer_black/CH_W_T_B.sgf"
play_bots_match(transformer, convolution_model, save_plot_path = save_plot_path, save_sgf_path = save_sgf_path)