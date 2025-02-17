# Subgoal Search Chess

Subgoal search for chess

## General imfo and usage examples

### Run experiments locally

```
from local_runner import local_run
local_run(path_to_experiment_specification, use_neptune, local_path_bindings)
```
For example
```local_run("experiments/train/policy/ultra_small_model.py", True, None)```

### Dataset introductory examples

See the file `assets/introductory_examples/read_dataset.py`

## Technical stuff and troubleshooting

### Stockfish installation:

https://installati.one/ubuntu/20.04/stockfish/

Make sure that the command `stockfish` works in the terminal.

### Leela (with ability to save graphs) installation:
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

See also **leelas_tree_generator.py** file, which includes an example use of Leela engine.

### Vizualistion of Leela's trees:
If You want make vizualistion of Leela's trees, use: https://github.com/jkormu/Leela-chess-Tree


## Singularity sandbox:

``` sudo singularity build --sandbox my_sbox my_image.sif```

### Building Lichess Bots:

Install Banksia (https://banksiagui.com/)

If you have troubles connecting Banksiagui to Lichess, follow this instruction (tested on Ubuntu 22.04):

https://linuxpip.org/install-openssl-linux/

