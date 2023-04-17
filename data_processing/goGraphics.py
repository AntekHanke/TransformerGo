import matplotlib.pyplot as plt
#def plotly_board_from_boards(board):


def plot_go_boards(boards, expmov=None, predmov=None, predmovs=None):
    '''Accepts boards same as boards dumped by sente library (numpy 19x19x4 array)'''
    # create a 8" x 8" board
    fig = plt.figure(figsize=[8, 8])
    fig.patch.set_facecolor((1, 1, .8))
    ax = fig.add_subplot(111)
    # draw the grid
    for x in range(19):
        ax.plot([x, x], [0, 18], 'k')
    for y in range(19):
        ax.plot([0, 18], [y, y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0, 0, 1, 1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 19)

    # draw Go stones at (10,10) and (13,16)


    for y, row in enumerate(boards):
        # print("row: ",row)
        for x, val in enumerate(row):
            # print("val: ",val)
            if (val[0] == 1):
                s2, = ax.plot(x, y, 'o', markersize=30, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                              markeredgewidth=2)
            elif (val[1] == 1):
                s1, = ax.plot(x, y, 'o', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                              markeredgewidth=2)

    if (expmov):
        col = expmov[2]
        x = expmov[1] - 1
        y = expmov[0] - 1
        if (col == 0):
            s2, = ax.plot(x, y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='k',
                          markeredgewidth=2)
        else:
            s1, = ax.plot(x, y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='w',
                          markeredgewidth=2)
    if (predmov):
        x = predmov[1] - 1
        y = predmov[0] - 1
        if (col == 0):
            s2, = ax.plot(x, y, 'o', markersize=15, markeredgecolor=(0, 0, 1), markerfacecolor='k',
                          markeredgewidth=2)
        else:
            s1, = ax.plot(x, y, 'o', markersize=15, markeredgecolor=(0, 0, 1), markerfacecolor='w',
                          markeredgewidth=2)

    if (predmovs):
        for predmov in predmovs:
            x = predmov[1] - 1
            y = predmov[0] - 1
            if (col == 0):
                s2, = ax.plot(x, y, 'o', markersize=15, markeredgecolor=(0, 0, 1), markerfacecolor='k',
                              markeredgewidth=2)
            else:
                s1, = ax.plot(x, y, 'o', markersize=15, markeredgecolor=(0, 0, 1), markerfacecolor='w',
                              markeredgewidth=2)

    return fig, ax

import sente
def plot_go_game(game: sente.Game, lastmove = True):
    '''Accepts boards same as boards dumped by sente library (numpy 19x19x4 array)'''
    # create a 8" x 8" board
    fig = plt.figure(figsize=[8, 8])
    fig.patch.set_facecolor((1, 1, .8))
    ax = fig.add_subplot(111)
    # draw the grid
    for x in range(19):
        ax.plot([x, x], [0, 18], 'k')
    for y in range(19):
        ax.plot([0, 18], [y, y], 'k')

    # scale the axis area to fill the whole figure
    ax.set_position([0, 0, 1, 1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 19)

    # draw Go stones at (10,10) and (13,16)
    boards = game.numpy()

    for y, row in enumerate(boards):
        # print("row: ",row)
        for x, val in enumerate(row):
            # print("val: ",val)
            if (val[0] == 1):
                s2, = ax.plot(x, y, 'o', markersize=30, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                              markeredgewidth=2)
            elif (val[1] == 1):
                s1, = ax.plot(x, y, 'o', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                              markeredgewidth=2)

    if (lastmove):
        try:
            moves = game.get_sequence()
            lastmov = moves[-1]
            y = lastmov.get_x()
            x = lastmov.get_y()
            stone = lastmov.get_stone()
            col = (stone == sente.stone.WHITE)
            if (col == 0):
                s2, = ax.plot(x, y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='k',
                              markeredgewidth=2)
            else:
                s1, = ax.plot(x, y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='w',
                              markeredgewidth=2)
        except:
            print("no last move")
    return fig, ax