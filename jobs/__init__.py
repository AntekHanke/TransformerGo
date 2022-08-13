import gin

from jobs.any_job import AnyJob

import sys, inspect
def print_classes():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        print(name)
        if inspect.isclass(obj):
            print(obj)


def configure_job(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='jobs'
    )

classes_to_configure = [AnyJob]

for cls in classes_to_configure:
    configure_job(cls)