from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List


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
        msg = self.slot
        for name, value in kwargs.items():
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise TypeError(f"Expected a string, got {type(value).__name__}")
            msg = msg.replace("{{" + name + "}}", value, 1)
        return msg
