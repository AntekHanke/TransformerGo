from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sente
import copy

# GoImmutableBoardData = namedtuple(
#     "GoImmutableBoard",
#     "game",
# )
#
# class GoImmutableBoard(GoImmutableBoardData):
#     @classmethod
#     def from_game(cls, game: sente.Game) -> "GoImmutableBoard":
#         return cls(game)
#     #
#     # @classmethod
#     # def from_fen_str(cls, fen: str) -> "ImmutableBoard":
#     #     return ImmutableBoard(*fen.split())
#     #
#     # def to_board(self) -> chess.Board:
#     #     return chess.Board(fen=" ".join(self))
#     #@classmethod
#     def to_np_array(self) -> (np.array, sente.stone):
#         a = self.game.numpy()
#         return (self.game.numpy(), self.game.get_active_player())
#
#
#     def act(self, move: (int, int)) -> "GoImmutableBoard":
#         newgame = GoImmutableBoard.from_game(self.game)
#
#         newgame.game.play(move[0], move[1])
#         return GoImmutableBoard.from_game(newgame.game)
#
#     def legal_moves(self) -> Tuple[sente.Move]:
#         return self.game.get_legal_moves()
#
#     # def fen(self) -> str:
#     #     return " ".join(self)
#     #
#     # def __hash__(self):
#     #     return hash(self.board + self.active_player + self.castles + self.en_passant_target)
#
#     def __eq__(self, other):
#         return all(
#             [
#                 self.game.to_np_array == other.game.to_np_array
#             ]
#         )

class GoMetadata:
    """Stores arbitrary metadata about a single game. Different games have different metadata fields."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


GoTransition = namedtuple("GoTransition", "immutable_board move move_number")

GoOneGameData = namedtuple("GoOneGameData", "metadata transitions")

GoImmutableBoardData = namedtuple(
    "GoImmutableBoard",
    "boards legal_moves active_player metadata",
)


class GoImmutableBoard(GoImmutableBoardData):
    @classmethod
    def from_game(cls, game: sente.Game) -> "GoImmutableBoard":
        return cls(game.numpy(), game.get_legal_moves(), game.get_active_player(), game.get_properties())
    # @classmethod
    # def from_sgf_string(cls, sgf_string: str) -> "GoImmutableBoard":
    #     game = sente.sgf.loads(sgf_string)
    #     return cls(game.numpy(), game.get_legal_moves(), game.get_active_player(), game.get_properties())
    @classmethod
    def from_all_data(cls, boards: np.array, legal_moves, active: sente.stone, metadata) -> "ImmutableBoard":
        return cls(boards, legal_moves, active, metadata)

    def to_game(self, moves=None) -> sente.Game:
        """ Creates a game object and apply moves from the list. """
        if(moves==None):
            moves = self.numpy_to_moves()

        game = sente.Game()
        if moves == []:
            return game
        if moves[0].get_stone() == sente.stone.WHITE:
            game.pss()
        for i in range(len(moves)):
            game.play(moves[i])
            if i + 1 == len(moves):
                return game
            if moves[i].get_stone() == moves[i + 1].get_stone():
                game.pss()

        return game


    def numpy_to_moves(self, array=None, active_player=None):
        """ Takes numpy array as an input and produces a sequence of moves. """
        if(array==None):
            array = self.boards
        if(active_player==None):
            active_player = self.active_player
        moves = []
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j][0] == 1:
                    moves.append(sente.Move(i, j, sente.BLACK))
                elif array[i][j][1] == 1:
                    moves.append(sente.Move(i, j, sente.WHITE))

        if not moves:
            return moves

        ### Makes last move a pass if neccesary
        if(moves[-1].get_stone() == active_player):
            if(active_player == sente.BLACK):
                opp = sente.WHITE
            else:
                opp = sente.BLACK
            moves.append(sente.Move(19, 19, opp))
        return moves

    def act(self, move: sente.Move) -> "GoImmutableBoard":
        game = self.to_game()
        game.play(move)
        return GoImmutableBoard.from_game(game)

    def __eq__(self, other):
        return all(
            [
                self.boards == other.boards,
                self.active_player == other.active_player
            ]
        )







if __name__ == '__main__':

    # np.set_printoptions(threshold=2000)
    # game = sente.sgf.load("00006007.sgf", disable_warnings=True)
    # print(sente.sgf.dumps(game))
    # costam = "(;SZ[19]BR[3d]RE[B+9.50]WR[7d]HA[4]KM[0.50];B[dd];W[];B[dp];W[];B[pd];W[];B[pp];W[qn];B[ql];W[qf];B[qh];W[qj];B[pn];W[of];B[nc];W[pi];B[rd];W[po];B[oo];W[qo];B[pm];W[qp];B[pq];W[qq];B[qr];W[rr];B[pr];W[rm];B[rl];W[no];B[nn];W[op];B[on];W[nq];B[mo];W[np];B[lq];W[or];B[ph];W[oh];B[oi];W[nh];B[oj];W[fc];B[cf];W[dg];B[cg];W[di];B[dh];W[eh];B[ch];W[fj];B[ec];W[fd];B[kc];W[fq];B[eq];W[fp];B[dn];W[lp];B[mp];W[mq];B[kq];W[mn];B[lo];W[kp];B[mr];W[nr];B[lr];W[jp];B[ln];W[iq];B[os];W[cl];B[cm];W[dl];B[fr];W[gr];B[er];W[cc];B[eb];W[cd];B[de];W[db];B[bb];W[fb];B[da];W[ca];B[ea];W[cb];B[be];W[bd];B[eg];W[ab];B[ac];W[ba];B[fg];W[mc];B[nd];W[md];B[me];W[ld];B[nf];W[qi];B[rh];W[pg];B[ri];W[rj];B[rf];W[re];B[qe];W[rg];B[se];W[qg];B[id];W[nb];B[ob];W[ib];B[ic];W[jb];B[kb];W[hb];B[kd];W[he];B[if];W[gg];B[fh];W[ie];B[jf];W[ps];B[qs];W[ns];B[rs];W[sr];B[ms];W[ps];B[hr];W[gq];B[os];W[oc];B[mb];W[ps];B[ng];W[jq];B[os];W[od];B[ne];W[ps];B[og];W[si];B[os];W[lb];B[na];W[ps];B[rq];W[kr];B[rp];W[ro];B[sn];W[sp];B[sm];W[mh];B[mj];W[hf];B[gh];W[hh];B[hi];W[ih];B[kh];W[li];B[ii];W[lj];B[lk];W[kk];B[ll];W[jj];B[ji];W[je];B[ke];W[kf];B[jg];W[ff];B[ef];W[fe];B[aa];W[ed];B[dc];W[ab];B[fa];W[ga];B[aa];W[co];B[do];W[ab];B[bl];W[bk];B[bm];W[ce];B[cj];W[ck];B[fl];W[ci];B[bi];W[bj];B[dj];W[ei];B[ej];W[fi];B[ek];W[bh];B[ai];W[ah];B[gi];W[em];B[gj];W[aj];B[bi];W[ai];B[fk];W[bi];B[bg];W[ae];B[ag];W[fm];B[gm];W[gn];B[hm];W[km];B[fn];W[en];B[fo];W[eo];B[go];W[ep];B[cq];W[jm];B[hp];W[hq];B[ik];W[kl];B[lm];W[ij];B[il];W[hj];B[hk];W[ho];B[hn];W[io];B[gs];W[pk];B[ok];W[pl];B[qm];W[ol];B[nl];W[oe];B[pf];W[pe];B[pc];W[ir];B[pf];W[sh];B[ka];W[mi];B[lg];W[mg];B[nj];W[hs];B[fs];W[al];B[dm];W[el];B[am];W[gp];B[gn];W[bf];B[df];W[af];B[ja];W[ia];B[hd];W[gd];B[hc];W[gc];B[mf];W[sf];B[sg];W[om];B[nm];W[sf];B[ki];W[re];B[kj];W[sd];B[rc];W[jk];B[sc];W[se];B[so];W[sq];B[rq];W[ls];B[rn];W[rp];B[jn];W[in];B[kn];W[im];B[sk];W[jd];B[jc];W[rk];B[jo];W[lh];B[jh];W[hg];B[ak];W[lf];B[kg];W[al];B[aa];W[bc];B[ak];W[sj];B[sl];W[al];B[ph];W[qh];B[ak];W[pe];B[oe];W[al];B[is];W[js];B[ak];W[qd];B[pe];W[al];B[ig];W[ak];B[ni];W[ph];B[gk];W[pj];B[qk];W[ip];B[ko];W[dk];B[ee];W[];B[])"
    # #costam = "(;SZ[19]BR[3d]RE[B+9.50]WR[7d]HA[4]KM[0.50];AB[dd];AB[dp];AB[pd];AB[pp];W[qn];B[ql];W[qf];B[qh];W[qj];B[pn];W[of];B[nc];W[pi];B[rd];W[po];B[oo];W[qo];B[pm];W[qp];B[pq];W[qq];B[qr];W[rr];B[pr];W[rm];B[rl];W[no];B[nn];W[op];B[on];W[nq];B[mo];W[np];B[lq];W[or];B[ph];W[oh];B[oi];W[nh];B[oj];W[fc];B[cf];W[dg];B[cg];W[di];B[dh];W[eh];B[ch];W[fj];B[ec];W[fd];B[kc];W[fq];B[eq];W[fp];B[dn];W[lp];B[mp];W[mq];B[kq];W[mn];B[lo];W[kp];B[mr];W[nr];B[lr];W[jp];B[ln];W[iq];B[os];W[cl];B[cm];W[dl];B[fr];W[gr];B[er];W[cc];B[eb];W[cd];B[de];W[db];B[bb];W[fb];B[da];W[ca];B[ea];W[cb];B[be];W[bd];B[eg];W[ab];B[ac];W[ba];B[fg];W[mc];B[nd];W[md];B[me];W[ld];B[nf];W[qi];B[rh];W[pg];B[ri];W[rj];B[rf];W[re];B[qe];W[rg];B[se];W[qg];B[id];W[nb];B[ob];W[ib];B[ic];W[jb];B[kb];W[hb];B[kd];W[he];B[if];W[gg];B[fh];W[ie];B[jf];W[ps];B[qs];W[ns];B[rs];W[sr];B[ms];W[ps];B[hr];W[gq];B[os];W[oc];B[mb];W[ps];B[ng];W[jq];B[os];W[od];B[ne];W[ps];B[og];W[si];B[os];W[lb];B[na];W[ps];B[rq];W[kr];B[rp];W[ro];B[sn];W[sp];B[sm];W[mh];B[mj];W[hf];B[gh];W[hh];B[hi];W[ih];B[kh];W[li];B[ii];W[lj];B[lk];W[kk];B[ll];W[jj];B[ji];W[je];B[ke];W[kf];B[jg];W[ff];B[ef];W[fe];B[aa];W[ed];B[dc];W[ab];B[fa];W[ga];B[aa];W[co];B[do];W[ab];B[bl];W[bk];B[bm];W[ce];B[cj];W[ck];B[fl];W[ci];B[bi];W[bj];B[dj];W[ei];B[ej];W[fi];B[ek];W[bh];B[ai];W[ah];B[gi];W[em];B[gj];W[aj];B[bi];W[ai];B[fk];W[bi];B[bg];W[ae];B[ag];W[fm];B[gm];W[gn];B[hm];W[km];B[fn];W[en];B[fo];W[eo];B[go];W[ep];B[cq];W[jm];B[hp];W[hq];B[ik];W[kl];B[lm];W[ij];B[il];W[hj];B[hk];W[ho];B[hn];W[io];B[gs];W[pk];B[ok];W[pl];B[qm];W[ol];B[nl];W[oe];B[pf];W[pe];B[pc];W[ir];B[pf];W[sh];B[ka];W[mi];B[lg];W[mg];B[nj];W[hs];B[fs];W[al];B[dm];W[el];B[am];W[gp];B[gn];W[bf];B[df];W[af];B[ja];W[ia];B[hd];W[gd];B[hc];W[gc];B[mf];W[sf];B[sg];W[om];B[nm];W[sf];B[ki];W[re];B[kj];W[sd];B[rc];W[jk];B[sc];W[se];B[so];W[sq];B[rq];W[ls];B[rn];W[rp];B[jn];W[in];B[kn];W[im];B[sk];W[jd];B[jc];W[rk];B[jo];W[lh];B[jh];W[hg];B[ak];W[lf];B[kg];W[al];B[aa];W[bc];B[ak];W[sj];B[sl];W[al];B[ph];W[qh];B[ak];W[pe];B[oe];W[al];B[is];W[js];B[ak];W[qd];B[pe];W[al];B[ig];W[ak];B[ni];W[ph];B[gk];W[pj];B[qk];W[ip];B[ko];W[dk];B[ee];W[];B[])"
    # #
    gicior = "(;BR[1p]RE[W+R]WR[1p]KM[5.5];B[pd];W[dp];B[pq];W[dd];B[qo];W[kq];B[cn];W[fp];B[bp];W[cq];B[ck];W[nq];B[fc];W[id];B[fe];W[cf];B[db];W[cc];B[ch];W[cb];B[ib];W[qf];B[qh];W[nc];B[nd];W[md];B[ne];W[gd];B[fd];W[hb];B[hc];W[ic];B[gb];W[jb];B[ha];W[pc];B[qc];W[oc];B[qd];W[lc];B[pg];W[dl];B[cl];W[dn];B[iq];W[co];B[dm];W[em];B[cm];W[jp];B[ip];W[jo];B[gq];W[fq];B[kr];W[jr];B[jq];W[lr];B[kp];W[ks];B[lq];W[kr];B[hn];W[ko];B[im];W[bo];B[no];W[mo];B[mn];W[np];B[lo];W[mp];B[oo];W[rj];B[rh];W[qm];B[qk];W[ro];B[qn];W[rn];B[rp];W[ql];B[rk];W[pk];B[rl];W[rm];B[qj];W[nm];B[nn];W[rq];B[sp];W[nk];B[lm];W[kn];B[km];W[mj];B[io];W[lp];B[kj];W[li];B[ki];W[lg];B[kh];W[ng];B[qq];W[od];B[oe];W[me];B[bg];W[bf];B[bn];W[pe];B[of];W[qb];B[rb];W[pb];B[ra];W[ao];B[el];W[fm];B[fl];W[dh];B[di];W[eh];B[ei];W[cg];B[bh];W[hd];B[ec];W[gc];B[hb];W[kg];B[fh];W[ih];B[af];W[ae];B[ag];W[bd];B[ef];W[dg];B[gg];W[hh];B[gm];W[fn];B[fr];W[er];B[or];W[jl];B[jm];W[lk];B[ij];W[hj];B[gi];W[hi];B[ik];W[gr];B[hr];W[fs];B[ca];W[ba];B[da];W[bb];B[kb];W[ja];B[if];W[hf];B[hg];W[ig];B[he];W[jf];B[ie];W[ge];B[gf];W[jd];B[je];W[ke];B[hf];W[kc];B[gj];W[hk];B[kk];W[gk];B[fk];W[gl];B[eg];W[il];B[kl];W[om];B[ln];W[nr];B[oi];W[pn];B[po];W[oh];B[ph];W[re];B[sd];W[rd];B[rc];W[qe];B[sf];W[sb])"
    # game2 = sente.sgf.loads(costam)
    game3 = sente.sgf.loads(gicior)
    sequence = game3.get_default_sequence()
    game3.play_sequence(sequence[0:120])
    goBoard = GoImmutableBoard.from_game(game3)
    print(goBoard.boards)
    fig, ax = plot_go_game(goBoard.boards)
    fig.show()

    # # #print(sente.sgf.dumps(game))
    # sequence = game2.get_default_sequence()
    # print(sequence)
    # # print(sequence[0].get_stone())
    # game2.play_sequence(sequence[0:0])
    #
    # immutBoard = GoImmutableBoard.from_game(game2)
    # print(immutBoard)
    # #print(immutBoard.to_game())
    # immutBoard2 = immutBoard.act(sente.Move(1,1,sente.BLACK))
    #
    # print(immutBoard2.to_game())
    # print(immutBoard.to_game())
    #game = sente.sgf.load("00006007.sgf", disable_warnings=True)
    #print(sente.sgf.dumps(game))
    # print(game2)
    #
    # cos = GoImmutableBoard.from_game(game)
    # cos2 = cos.act((1,1))
    # #print(cos.game)
    #
    # passmov = sente.Move(19,19, sente.WHITE)
    # game2.play(passmov)
    # game2.play(sente.Move(9,9, sente.BLACK))
    # print(game2)
    # print(passmov)
    # a = np.array([[0,1],[1,1]])
    # ii = np.where(a==1)
    # print(ii)
    # # game = sente.Game(9)
    # # game.play(1, 1)
    # # game2 = copy.deepcopy(game)
    # # game2.play(1,2)
    # # print(game)