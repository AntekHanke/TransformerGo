import socket
from enum import Enum, auto


def get_local_machine():
    hostname = socket.gethostname()
    if hostname == "TomaszOpc":
        return "tomasz"
    elif hostname == "g":
        return "gracjan"
    else:
        raise Exception("Unknown hostname: " + hostname + ". Please add it to the list of known machines.")

