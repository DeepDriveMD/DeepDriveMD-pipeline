from enum import Enum, auto, unique


@unique
class Policy1(Enum):
    right = auto()
    left = auto()


print(Policy1.right)
