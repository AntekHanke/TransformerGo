## How to run Lichess Bot

## Running from pythonic interface

1. Install updated requirements from `requirements.txt` file in the general directory.
2. Example on how to use pythonic interface can be found in `utils/run_lichess_bot.py`.

## Running as external bot

### Requirements

1. Install requirements from `requirements.txt` file to your 
preexisting virtual environment.

### Setting up the bot
1. There is one main object you will be modifying:
   1. **config.yaml** - this file contains all the configuration needed to run your
   engine as a Lichess bot
2. Put your executable engine in the appropriate directory in *chess_engines/banksia_gui_uci/<your_name>*.
3. Double-check the engine file is executable by running `chmod +x <engine_name>`.
5. In the `config.yaml` file, change the `engine` field to the name of your engine.
6. Change also `dir` so that it points to your directory.
6. If you wish, you can also play around with other settings, according to instructions in
the [Lichess Bot documentation - Engine Configuration](README.md#engine-configuration).

*Sidenote: `uci_options: Move Overhead, Threads, Hash, SyzygyPath` have been commented out to ensure
the engine running according to https://github.com/lichess-bot-devs/lichess-bot/issues/312* .

### Running the bot

1. The first time upgrade to bot account by running `python3 lichess-bot.py -u`.
2. Run `python3 lichess-bot.py` to start the bot. Use `python3 lichess-bot.py -v` to see 
the output and debugging information. 
3. If you want to log the output to a file, use
`python3 lichess-bot.py --logfile log.txt`
4. Stop the bot by pressing `Ctrl+C`.