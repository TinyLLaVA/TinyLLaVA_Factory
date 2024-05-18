from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import  Dict, Union, List


SLOT = Union[str, List[str], Dict[str, str]]

@dataclass
class Formatter(ABC):
    slot: SLOT = ""

    @abstractmethod
    def apply(self, **kwargs) -> SLOT: ...



@dataclass
class EmptyFormatter(Formatter):
    def apply(self, **kwargs) -> SLOT:
        return self.slot


@dataclass
class StringFormatter(Formatter):
    def apply(self, **kwargs) -> SLOT:
        msg = ""
        for name, value in kwargs.items():
            if value is None:
                msg = self.slot.split(':')[0] + ":"
                return msg
            if not isinstance(value, str):
                raise RuntimeError("Expected a string, got {}".format(value))
            msg = self.slot.replace("{{" + name + "}}", value, 1)
        return msg
