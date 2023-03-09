import socket
from enum import Enum, auto

local_machines = {"TomaszOpc": "tomasz", "g": "gracjan", "dell-latitude-e7450": "malgorzata"}
hostname = socket.gethostname()


def get_local_machine():
    if hostname in local_machines:
        return local_machines[hostname]
    else:
        raise Exception("Unknown hostname: " + hostname + ". Please add it to the list of known machines.")


def is_local_machine():
    return hostname in local_machines.keys()
