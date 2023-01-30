import socket
from enum import Enum, auto


def get_local_machine():
    hostname = socket.gethostname()
    if hostname == "TomaszOpc":
        return "tomasz"
    elif hostname == "g":
        return "gracjan"
    elif hostname == "dell-latitude-e7450":
        return "malgorzata"
    else:
        raise Exception("Unknown hostname: " + hostname + ". Please add it to the list of known machines.")


def is_local_machine():
    hostname = socket.gethostname()
    if hostname in ["TomaszOpc", "g"]:
        return True
    else:
        return False