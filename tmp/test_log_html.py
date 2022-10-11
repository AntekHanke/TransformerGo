from configures.global_config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT
from mrunner_utils.mrunner_client import NeptuneLogger

import neptune.new as neptune

run = neptune.init_run(
    api_token=NEPTUNE_API_TOKEN,
    project=NEPTUNE_PROJECT,
    name="callback_test",
    tags=["callback"],
)

neptune_logger = NeptuneLogger(run)

neptune_logger.log_object("html", "baby_shark.html")
# run["html_code"].log("baby_shark.html")
run["html_web"].upload("baby_shark.html")