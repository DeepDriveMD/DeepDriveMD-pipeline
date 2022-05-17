from enum import Enum, auto, unique


@unique
class DataStructure(Enum):
    array = auto()
    scalar = auto()
    string = auto()
