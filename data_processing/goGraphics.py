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
def plot_go_game(game: sente.Game, lastmove = True, explore_move_possibs = None, black_winning_prob = None, explore_mask_possibs = None):
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

    if (explore_mask_possibs):
        try:
            ma, mi = max(explore_mask_possibs[1]), min(explore_mask_possibs[1])
            absol = max(abs(ma), abs(mi))
        except:
            absol=1
        for x, row in enumerate(boards):
            # print("row: ",row)
            for y, val in enumerate(row):
                # print("val: ",val)
                if (val[0] == 1):
                    print(f"Finding: ({x}, {y}) in {explore_mask_possibs[0]}")
                    try:
                        probchange = explore_mask_possibs[1][explore_mask_possibs[0].index((x, y))]
                    except:
                        probchange=absol
                    s2, = ax.plot(x, 18-y, 'o', markersize=30, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                                  markeredgewidth=2, alpha=(abs(probchange)+0.001)/(absol+0.001))
                elif (val[1] == 1):
                    print(f"Finding: ({x}, {y}) in {explore_mask_possibs[0]}")
                    try:
                        probchange = explore_mask_possibs[1][explore_mask_possibs[0].index((x, y))]
                    except:
                        probchange=absol
                    probchange = explore_mask_possibs[1][explore_mask_possibs[0].index((x, y))]
                    s1, = ax.plot(x, 18-y, 'o', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                                  markeredgewidth=2, alpha=(abs(probchange)+0.001)/(absol+0.001))
        pass
    else:
        for x, row in enumerate(boards):
            # print("row: ",row)
            for y, val in enumerate(row):
                # print("val: ",val)
                if (val[0] == 1):
                    s2, = ax.plot(x, 18-y, 'o', markersize=30, markeredgecolor=(.5, .5, .5), markerfacecolor='k',
                                  markeredgewidth=2)
                elif (val[1] == 1):
                    s1, = ax.plot(x, 18-y, 'o', markersize=30, markeredgecolor=(0, 0, 0), markerfacecolor='w',
                                  markeredgewidth=2)

    if (lastmove):
        try:
            moves = game.get_sequence()
            lastmov = moves[-1]
            x = lastmov.get_x()
            y = lastmov.get_y()
            stone = lastmov.get_stone()
            col = (stone == sente.stone.WHITE)
            if (col == 0):
                s2, = ax.plot(x, 18-y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='k',
                              markeredgewidth=2)
            else:
                s1, = ax.plot(x, 18-y, 'o', markersize=25, markeredgecolor=(1, 0, 0), markerfacecolor='w',
                              markeredgewidth=2)
        except:
            print("no last move")

    if (explore_move_possibs):
        try:
            for move, prob in zip(explore_move_possibs[0], explore_move_possibs[1]):
                x, y = move
                # x+=1
                # y+=1
                s3 = ax.text(x-1, 19-y, "{:.0%}".format(prob), color='red', fontsize=11+5*prob, ha="center", va="center")
        except:
            for move, prob in zip(explore_move_possibs[0], explore_move_possibs[1]):
                x, y, _ = move
                # x+=1
                # y+=1
                s3 = ax.text(x-1, 19-y, "{:.0%}".format(prob), color='red', fontsize=11+5*prob, ha="center", va="center")

    # if (explore_mask_possibs):
    #     try:
    #         for move, prob in zip(explore_mask_possibs[0], explore_mask_possibs[1]):
    #             x, y = move
    #             x+=1
    #             y+=1
    #             s3 = ax.text(x-1, 19-y, "{:.0%}".format(prob), color='blue', fontsize=11+5*abs(prob), ha="center", va="center")
    #     except:
    #         for move, prob in zip(explore_mask_possibs[0], explore_mask_possibs[1]):
    #             x, y, _ = move
    #             x+=1
    #             y+=1
    #             s3 = ax.text(x-1, 19-y, "{:.0%}".format(prob), color='blue', fontsize=11+5*abs(prob), ha="center", va="center")

    if (black_winning_prob):
        ax.text(0, -1, f"Black winning probability: "+"{:.0%}".format(black_winning_prob), color='black', fontsize=12)

    return fig, ax