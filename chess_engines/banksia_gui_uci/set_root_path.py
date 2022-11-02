import os


def set_root_path():
    if "SUBGOAL_PROJECT_ROOT" not in os.environ:
        raise Exception(
            "SUBGOAL_PROJECT_ROOT not in os.environ, please set this variable pointing to the root of the project"
        )
    import sys

    sys.path.append(os.environ["SUBGOAL_PROJECT_ROOT"])
