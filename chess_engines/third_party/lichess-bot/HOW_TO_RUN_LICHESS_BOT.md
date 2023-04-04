## How to run Lichess Bot

### Requirements

1. Install requirements from `requirements.txt` file to your 
preexisting virtual environment.

### Setting up the bot
1. There are two main objects you will be modifying:
   1. **config.yaml** - this file contains all the configuration needed to run your
   engine as a Lichess bot
   2. **engines** directory - this directory contains executables to the engines 
   you will be running
2. Put your executable engine in the `engines` directory following the format of the engines
in *chess_engines/banksia_gui_uci*.
3. Double-check the engine file is executable by running `chmod +x <engine_name>`.
5. In the `config.yaml` file, change the `engine` field to the name of your engine.
6. If you wish, you can also play around with other settings, according to instructions in
the [Lichess Bot documentation - Engine Configuration](README.md#engine-configuration).

### Running the bot

1. Run `python3 lichess-bot.py` to start the bot. Use `python3 lichess-bot.py -v` to see 
the output and debugging information. 
2. If you want to log the output to a file, use
`python3 lichess-bot.py --logfile log.txt`
3. Stop the bot by pressing `Ctrl+C`.