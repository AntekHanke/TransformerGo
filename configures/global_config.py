from configures.locally_testing_jobs.detect_local_machine import get_local_machine

if get_local_machine() == "tomasz":
    NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZDRhNWIxNS1hN2RkLTQ1ZjMtOGRmZi02MWI4NGRkZjA5MGMifQ=="
    NEPTUNE_PROJECT = "pmtest/subgoal-chess"

    EAGLE_DATASET = "/home/plgrid/plgtodrzygozdz/subgoal_chess/database.pgn"

    TRAIN_TEST_SPLIT_SEED = 11
    VALUE_FOR_MATE = 100000

    ENTROPY_HOME = "/home/todrzygozdz"
    EAGLE_HOME = "/home/plgrid/plgtodrzygozdz"

    MAX_JOBLIB_N_JOBS = 28

    MAX_NEW_TOKENS_FOR_POLICY = 2
    MAX_MOVES_FOR_CLLP = 6