#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import matplotlib.pyplot as plt
import pygame
from transformers import BartForConditionalGeneration
from data_processing.goGraphics import plot_go_game
from data_processing.goPlay import go
from data_processing.go_tokenizer import GoTokenizer
from sys import exit
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Type, Union, Tuple
from sys import path
from data_structures.go_data_structures import GoImmutableBoard
import sente
from go_policy.policy_model import AlphaZeroPolicyModel
from collections import deque
import os

BACKGROUND = 'images/ramin.jpg'
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



class Stone(go.Stone):
    def __init__(self, board, point, color, screen, background):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.coords = (5 + self.point[0] * 40, 5 + self.point[1] * 40)
        self.screen = screen
        self.background = background
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(self.screen, self.color, self.coords, 20, 0)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (self.coords[0] - 20, self.coords[1] - 20)
        area_rect = pygame.Rect(blit_coords, (40, 40))
        self.screen.blit(self.background, blit_coords, area_rect)
        pygame.display.update()
        super(Stone, self).remove()


class Board(go.Board):
    def __init__(self, background, screen):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__()
        self.outline = pygame.Rect(45, 45, 720, 720)
        self.draw(background, screen)

    def draw(self, background = None, screen = None):
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
                        stone.remove(screen)
                    else:
                        added_stone = Stone(board, (x, y), board.turn())
                    board.update_liberties(added_stone)



class playingGoModel:
    """Model which can play the game."""

    def play_move(self, position: GoImmutableBoard) -> Tuple[int, int]:
        """Returns x,y coordinates of move played (20,20 is pass)."""
        raise NotImplementedError

    def play_moves(self, position: GoImmutableBoard) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Returns x,y coordinates of mulitple moves it would play (20,20 is pass).
        Used so that in a case of illegal move, the next one can be chosen.
        Also returns the probabilities of each of the returned moves"""
        pass


class valueGoModel:
    """Model which can play the game."""

    def value(self, position: GoImmutableBoard) -> (float):
        """Returns value of given position. (Probability of black winning)"""
        raise NotImplementedError


class TransformerValue(valueGoModel):
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
        self.softmax = torch.nn.Softmax()

    def get_best_moves(
        self,
        immutable_board: GoImmutableBoard,
        num_return_sequences: int = 2,
        num_beams: int = 2
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

        scores = outputs.sequences_scores
        probs = self.softmax(scores)
        moves = []
        moves_ids = []


        for output in sequence:
            real_move = [output[1]]
            moves.append(real_move)
            moves_ids.append(output)

        print("My values i found: ", moves)
        return moves, probs


    def value(self, position: GoImmutableBoard) -> float:

        results, probs = self.get_best_moves(position)
        for res,prob in zip(results, probs):
            if res==[4]:
                prob_real = prob

        print(f"Black winning chance: {prob_real}")
        return prob_real




class TransformerPolicy(playingGoModel):
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
        self.softmax = torch.nn.Softmax()

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

        scores = outputs.sequences_scores
        probs = self.softmax(scores)
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

        return moves, probs
    def play_move(self, position: GoImmutableBoard) -> Tuple[int, int]:
        moves = self.get_best_moves(position)
        print(f"My moves are: {moves}")
        whichChoice = 0
        bestmove = moves[whichChoice]
        x = bestmove[0] + 1
        y = bestmove[1] + 1
        return(x,y)

    def play_moves(self, curr_game: sente.Game) -> List[Tuple[int, int]]:
        position = GoImmutableBoard.from_game(curr_game)
        if(position.active_player == sente.stone.WHITE):
            position = GoImmutableBoard.from_all_data(position.boards[:,:,[1,0,2,3]], position.legal_moves, position.active_player, position.metadata)

        moves, probs = self.get_best_moves(position)
        realmoves = [(x+1, y+1) for (x,y,b) in moves]
        print(f"My moves are: {realmoves}")
        return realmoves, probs.tolist()

class ConvolutionPolicy(playingGoModel):
    def __init__(self, checkpoint_path_or_model, history = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used for policy:", device)

        if isinstance(checkpoint_path_or_model, str):
            self.model = AlphaZeroPolicyModel.from_pretrained(checkpoint_path_or_model)

        else:
            self.model = checkpoint_path_or_model

        self.model.to(device)
        self.model.eval()
        self.softmax = torch.nn.Softmax()
        self.history = history
        if self.history:
            self.moves = deque(maxlen = 8)

    def get_best_moves(
        self,
        immutable_board: GoImmutableBoard,
        top_k_moves = 8
    ):                     

        board = immutable_board.boards
        active_player = immutable_board.active_player
        active_player = 1 if active_player == sente.BLACK else 0
        print(active_player)
        if not self.history:
            input_tensor = torch.Tensor(np.append(board , np.full((board.shape[0], board.shape[1], 1) 
                                            , active_player), axis = -1))
            outputs= self.model(input_tensor.view(1, 19, 19 ,5).to(self.model.device))

        else:
            input_tensor = torch.Tensor(np.concatenate(list(reversed([np.full((board.shape[0], board.shape[1], 1), active_player)]
                                                                +list(self.moves))), axis = -1))  
            outputs = self.model(input_tensor.view(-1, 19, 19, 17).to(self.model.device))

        scores = outputs["logits"].flatten()
        scores = self.softmax(scores)
        topk = torch.topk(scores, top_k_moves)
        probs = topk.values.flatten()
        outputs = topk.indices.flatten()
        print(probs, outputs)
        moves = []
        for move in outputs.tolist():
            position = (int(move/19), move % 19) if int(move/19) != 20 else (19,19)
            moves.append(position) 

        print(f"Moves = {[str(move) for move in moves]}")


        return moves, probs

    def play_move(self, curr_game: sente.Game) -> Tuple[int, int]:
        position = GoImmutableBoard.from_game(curr_game)
        moves = self.get_best_moves(position)
        print(f"My moves are: {moves}")
        whichChoice = 0
        bestmove = moves[whichChoice]
        x = bestmove[0] + 1 
        y = bestmove[1] + 1 
        return(x,y)

    def play_moves(self, curr_game: sente.Game) -> List[Tuple[int, int]]:
        position = GoImmutableBoard.from_game(curr_game)
        if(position.active_player == sente.stone.WHITE):
            position = GoImmutableBoard.from_all_data(position.boards[:,:,[1,0,2,3]], position.legal_moves, position.active_player, position.metadata)
        if self.history:
            if len(self.moves) == 0:
                seq = curr_game.get_default_sequence()
                for move in seq:
                    curr_game.play_move(move)
                    self.moves.append(curr_game.numpy()[:,:,:2])
                if len(self.moves) != self.moves.maxlen: #not enough moves in history pad with zeros
                   for i in range(self.moves.maxlen - len(self.moves)):
                      self.moves.appendleft(np.zeros((19,19,2)))
            else:
                self.moves.append(curr_game.numpy()[:,:,:2])    
        moves, probs = self.get_best_moves(position)
        realmoves = [(x + 1, y + 1) for (x,y) in moves]
        print(f"My moves are: {realmoves}")
        return realmoves, probs.tolist()



def playAgainstModel(model: TransformerPolicy, youPlayAs = sente.stone.WHITE, 
                     sgf = None, 
                     Board = None, 
                     Screen = None, 
                     background = None,
                     save_plot_path = None):
    if Board is not None:
        board = Board  
    if Screen is not None:
        screen = Screen
    curr_game = sente.Game()
    if sgf:
        curr_game = sente.sgf.load(sgf)
        seq = curr_game.get_default_sequence()
        for mov in seq:
            curr_game.play(mov)
            x = mov.get_x() + 1
            y = mov.get_y() + 1
            added_stone = Stone(board, (x, y), board.turn(), screen, background)
            board.update_liberties(added_stone)
    while True:
        pygame.time.wait(250)
        sente.sgf.dump(curr_game, "autosave.sgf")
        if(curr_game.get_active_player() == youPlayAs):
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
                            added_stone = Stone(board, (x, y), board.turn(), screen, background)
                        board.update_liberties(added_stone)
        else:

            moves, probs = model.play_moves(curr_game)
            whichChoice = 0
            bestmove = moves[whichChoice]

            x = bestmove[0] 
            y = bestmove[1] 
            print("Playing ", x, y)

            if(x==20 and y==20):
                print("Pass!")


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
                added_stone = Stone(board, (x, y), board.turn(), screen, background)
                board.update_liberties(added_stone)
                curr_game.play(x,y)
                fig, ax = plot_go_game(curr_game, lastmove=True, explore_move_possibs=(moves, probs))
                fig.show()
                plt.savefig(os.path.join(save_plot_path, "move.png"))
                plt.clf()

            print(curr_game)


def play_bots_match(black_player: playingGoModel = None,
                     white_player: playingGoModel = None, 
                    value_model: valueGoModel = None,
                    save_plot_path = None,
                    save_sgf_path = None):
    curr_game = sente.Game()
    move_num = 0
    model_dict = {
        sente.stone.WHITE: white_player,
        sente.stone.BLACK: black_player
    }

    while not curr_game.is_over():

        print(curr_game)


        next = curr_game.get_active_player()
        moves, probs = model_dict[next].play_moves(curr_game)
        goImmut = GoImmutableBoard.from_game(curr_game)
        if(value_model):
            value = value_model.value(goImmut)
        else:
            value = None

        try:
            fig, ax = plot_go_game(curr_game, lastmove=True, explore_move_possibs=(moves, probs), black_winning_prob=value)
            #fig.show()
            plt.savefig(os.path.join(save_plot_path, str(move_num) + ".png"))
            #plt.close()
            #plt.clf()
        except Exception as e:
            print(e)

        whichChoice = 0
        while True:
            x,y = moves[whichChoice]
            if(x==20 and y==20):
                curr_game.pss()
                print("I PASSED!!!!")
                break
            try:
                curr_game.play(x,y)
                break
            except:
                print(f"Oops illegal!{x},{y}")
                whichChoice+=1
        move_num+=1

    sente.sgf.dump(curr_game, save_sgf_path)
    print(curr_game)






if __name__ == '__main__':


    #t192k = TransformerPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-192500")
    # t67k = TransformerPolicy("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/checkpoint-67000")
    # v127k = TransformerValue("/mnt/c/Users/Antek/PycharmProjects/subgoal_search_chess/exclude/value_models/checkpoint-127000")
    model = ConvolutionPolicy("../../conv-checkpoints/conv-checkpoint-365500") #""" For playing against a model: """
    #pygame.init()
    #pygame.display.set_caption('Goban')
    #screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    #background = pygame.image.load(BACKGROUND).convert()
    #board = Board()
    ##playAgainstModel(model = t192k)
    #playAgainstModel(model = model, sgf="my game.sgf")
    """For games between 2 bots"""
    play_bots_match(model, model)



    # """testing"""
    # game = sente.sgf.loads("(;GM[1]FF[4]SZ[19];B[pp];W[dd];B[dq];W[pd];B[cn];W[qn];B[qo];W[pn];B[mq];W[pj];B[gq];W[jd];B[dk];W[jj];B[cc];W[dc];B[cd];W[ce];B[be];W[cf];B[db];W[eb];B[bb];W[da];B[cb];W[fc];B[hc];W[ic];B[hd];W[bf];B[bd];W[if];B[hf];W[hg];B[gf];W[gg];B[ef];W[fg];B[ff];W[df];B[eg];W[ci];B[ei];W[dj];B[fh];W[ih];B[ej];W[ck];B[ie];W[je];B[jf];W[ig];B[kf];W[ld];B[mf];W[of];B[lh];W[lj];B[nh];W[nj];B[ph];W[qi];B[qh];W[qf];B[rf];W[rg];B[ji];W[ii];B[ij];W[ki];B[jh];W[hj];B[ik];W[hk];B[jk];W[kj];B[hl];W[gl];B[hm];W[fk];B[gm];W[fm];B[gj];W[gi];B[fj];W[gk];B[fn];W[dl];B[ek];W[el];B[cl];W[cm];B[bl];W[bm];B[bk];W[cj];B[bj];W[bi];B[em];W[dm];B[fl];W[jg];B[kh];W[fm];B[mi];W[en];B[mj];W[mk];B[nk];W[ok];B[nl];W[nn];B[ml];W[rh];B[ee];W[de];B[ed];W[ec];B[kc];W[lc];B[ib];W[id];B[jb];W[he];B[ge];W[gb];B[ie];W[kd];B[hb];W[lb];B[qc];W[pc];B[qd];W[re];B[qe];W[pe];B[rd];W[sf];B[pb];W[rb];B[qb];W[ob];B[ra];W[sb];B[pa];W[sd];B[sc];W[rc];B[ne];W[nc];B[fo];W[ol];B[mn];W[no];B[mo];W[np];B[nq];W[mp];B[lp];W[oq];B[op];W[lq];B[lr];W[lo];B[kp];W[mm];B[ln];W[lm];B[kn];W[lk])")
    # sequence = game.get_default_sequence()
    # print(sequence)
    # black_win_probs = []
    # for move in sequence:
    #     goImmut = GoImmutableBoard.from_game(game)
    #     black_win_probs.append(v127k.value(goImmut))
    #     game.play(move)


    #plt.plot(black_win_probs)