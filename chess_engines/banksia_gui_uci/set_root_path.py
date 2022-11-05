import os


def set_root_path():
    if "SUBGOAL_PROJECT_ROOT" not in os.environ:
        raise Exception(
            "SUBGOAL_PROJECT_ROOT not in os.environ, please set this variable pointing to the root of the project"
        )
    import sys

    # TODO: Fix sys env
    sys.path.append("/home/gracjan/subgoal/subgoal_search_chess-uci_engines")
