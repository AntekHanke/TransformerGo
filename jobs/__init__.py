import gin

from jobs.any_job import AnyJob


def configure_job(goal_generator_class):
    return gin.external_configurable(goal_generator_class, module="jobs")


classes_to_configure = [AnyJob]

for cls in classes_to_configure:
    configure_job(cls)
