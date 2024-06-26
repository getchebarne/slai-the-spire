from abc import ABC
from dataclasses import dataclass


@dataclass
class BaseComponent(ABC):
    pass


@dataclass
class BaseRelationshipComponent(BaseComponent):
    pass


@dataclass
class BaseSingletonComponent(BaseComponent):
    pass
