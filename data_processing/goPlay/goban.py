#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import pygame
from transformers import BartForConditionalGeneration

import go
from sys import exit
#from load_essentials import *
import torch
import numpy as np

from data_structures.go_data_structures import GoImmutableBoard

BACKGROUND = 'images/ramin.jpg'
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



class Stone(go.Stone):
    def __init__(self, board, point, color):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.coords = (5 + self.point[0] * 40, 5 + self.point[1] * 40)
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(screen, self.color, self.coords, 20, 0)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (self.coords[0] - 20, self.coords[1] - 20)
        area_rect = pygame.Rect(blit_coords, (40, 40))
        screen.blit(background, blit_coords, area_rect)
        pygame.display.update()
        super(Stone, self).remove()


class Board(go.Board):
    def __init__(self):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__()
        self.outline = pygame.Rect(45, 45, 720, 720)
        self.draw()

    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        self.outline.inflate_ip(20, 20)
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(45 + (40 * i), 45 + (40 * j), 40, 40)
                pygame.draw.rect(background, BLACK, rect, 1)
        for i in range(3):
            for j in range(3):
                coords = (165 + (240 * i), 165 + (240 * j))
                pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        if added_stone:
            added_stone.group.update_liberties()


def main():
    while True:
        pygame.time.wait(250)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and board.outline.collidepoint(event.pos):
                    x = int(round(((event.pos[0] - 5) / 40.0), 0))
                    y = int(round(((event.pos[1] - 5) / 40.0), 0))
                    stone = board.search(point=(x, y))

                    if stone:
                        stone.remove()
                    else:
                        added_stone = Stone(board, (x, y), board.turn())
                    board.update_liberties(added_stone)

# def loadModel(directory: str):
#     model =

from data_processing.go_tokenizer import GoTokenizer


class TransformerPolicy:
    def __init__(self, checkpoint_path_or_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used for policy:", device)

        if isinstance(checkpoint_path_or_model, str):
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path_or_model)

        else:
            self.model = checkpoint_path_or_model

        self.model.to(device)
        self.model.eval()
        self.tokenizer = GoTokenizer()

    def get_best_moves(
        self,
        immutable_board: GoImmutableBoard,
        num_return_sequences: int = 8,
        num_beams: int = 16
    ):
        input_tensor: torch.Tensor = torch.IntTensor(
            [self.tokenizer.encode_immutable_board(immutable_board)]
        ).to(self.model.device)
        outputs = self.model.generate(
            inputs=input_tensor,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=2,
            output_scores=True,
            return_dict_in_generate=True
        )

        sequence = outputs.sequences.tolist()
        scores = outputs.scores[0]

        moves = []
        moves_ids = []


        for output in sequence:
            #output = [x for x in output if x not in ChessTokenizer.special_vocab_to_tokens.values()]
            real_move = [output[1]]
            moves.append(self.tokenizer.decode_move(real_move))
            moves_ids.append(output)


        print(f"Moves = {[str(move) for move in moves]}")

        #board = immutable_board.to_board()
        #converted_moves = [chess960_to_standard(move, board) for move in moves]

        return moves


def playAgainstModel(model: TransformerPolicy, youPlayAs = WHITE):

    curr_game = sente.Game()

    while True:
        pygame.time.wait(250)

        if(board.next == youPlayAs):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    moves = curr_game.get_current_sequence()
                    print(moves)
                    sente.sgf.dump(curr_game, "my game.sgf")
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0] - 5) / 40.0), 0))
                        y = int(round(((event.pos[1] - 5) / 40.0), 0))
                        stone = board.search(point=(x, y))
                        if stone:
                            pass
                        else:
                            curr_game.play(x,y)
                            added_stone = Stone(board, (x, y), board.turn())
                        board.update_liberties(added_stone)
        else:
            goImmut = GoImmutableBoard.from_game(curr_game)

            moves = model.get_best_moves(goImmut)
            print(f"My moves are: {moves}")
            whichChoice = 0
            bestmove = moves[whichChoice]
            x = bestmove[0]+1
            y = bestmove[1]+1
            print("Playing ", x, y)
            stone = board.search(point=(x, y))

            while stone:
                print("Oops I tried illegal, going next choice")
                whichChoice+=1
                bestmove = moves[whichChoice]
                x = bestmove[0] + 1
                y = bestmove[1] + 1
                print("Which is ", x, y)
                stone = board.search(point=(x, y))

            else:
                added_stone = Stone(board, (x, y), board.turn())
                board.update_liberties(added_stone)
                curr_game.play(x,y)

            print(curr_game)




if __name__ == '__main__':

    import sente
    #
    cos = TransformerPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-67000")
    #
    # game = sente.Game()
    #
    # goImmut = GoImmutableBoard.from_game(game)
    #
    # wynik = cos.get_best_moves(goImmut)
    #
    # print(wynik)


    pygame.init()
    pygame.display.set_caption('Goban')
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()
    board = Board()

    playAgainstModel(cos)
