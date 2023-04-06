from chess_engines.third_party.lichess_bot.consts import (
    CONST_BOT_NAME,
    CONST_LIST_OF_BOT_NAMES_TO_CHALLENGE,
    CONST_MAX_TRIAL_NUMBER,
    CONST_STOP_AFTER_WHITELIST_EMPTY,
)
from chess_engines.third_party.lichess_bot.lichess_bot import *

logger = logging.getLogger(__name__)

BOT_NAME = "subgoalchess_dev"
BOTS_LIST = ["maia5:1", "maia1:2"]
MAX_TRIAL_NUMBER = 5
STOP_AFTER_WHITELIST_EMPTY = True
KWARGS = {}


if __name__ == "__main__":
    KWARGS[CONST_BOT_NAME] = BOT_NAME
    KWARGS["matchmaking"] = {}
    KWARGS["matchmaking"][CONST_LIST_OF_BOT_NAMES_TO_CHALLENGE] = BOTS_LIST
    KWARGS["matchmaking"][CONST_MAX_TRIAL_NUMBER] = MAX_TRIAL_NUMBER
    KWARGS["matchmaking"][CONST_STOP_AFTER_WHITELIST_EMPTY] = STOP_AFTER_WHITELIST_EMPTY

    results = run_lichess_bot(**KWARGS)
    breakpoint_to_see_results = 0
