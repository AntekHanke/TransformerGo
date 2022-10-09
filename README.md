# Subgoal Search Chess

Subgoal search for chess

## Stockfish installation:

https://installati.one/ubuntu/20.04/stockfish/

Make sure that the command `stockfish` works in the terminal.

## Leela (with ability to save graphs) installation:
Follow the instuction -> https://github.com/jkormu/lc0

Currently the machine is on the Eagale in:  `/home/plgrid/plggracjangoral/leela/build/release/lc0`.

Weigts to Leela can be found in: `/home/plgrid/plggracjangoral/leela_weights`. 

If You want to use Leela, simply use **chess.engine.SimpleEngine.popen_uci(leelas_parms)**, where **leelas_parms** is a list with Leela's specyfications, e.g.

    [
     '/home/plgrid/plggracjangoral/leela/build/release/lc0',
     '--weights=/home/plgrid/plggracjangoral/leela_weights/release744204.pb.gz',
     '--smart-pruning-factor=0.0',
     '--threads=1',
     '--minibatch-size=1',
     '--max-collision-events=1',
     '--max-collision-visits=1',
     '--cpuct=1.0'
     ]

See also **leelas_tree_generator.p** file, which includes an example use of Leela engine.

# Vizualistion of Leela's trees:
If You want make vizualistion of Leela's trees, use: https://github.com/jkormu/Leela-chess-Tree


## Example usages:

### Play with chess policy:

Simple self-play example is in file:
```local\experiments\self_play.py```


## Singularity sandbox:

``` sudo singularity build --sandbox my_sbox my_image.sif```