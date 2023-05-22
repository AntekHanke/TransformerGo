from data_processing.goPlay.goban import playAgainstModel, ConvolutionPolicy, Board, TransformerPolicy
import pygame
import sente

BACKGROUND = './data_processing/goPlay/images/ramin.jpg'
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
model = TransformerPolicy("./model_checkpoints/transformer_checkpoints/checkpoint-192500")
pygame.init()
pygame.display.set_caption('Goban')
screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
background = pygame.image.load(BACKGROUND).convert()
board = Board(background, screen)
path = "data_processing/goPlay/plots"
playAgainstModel(sgf = "./data_processing/goPlay/boards/ladder_wtm_discont.sgf", 
                 youPlayAs = sente.stone.BLACK,
                 model = model,
                  Board = board, 
                 Screen = screen, 
                 background = background,
                 save_plot_path=path)