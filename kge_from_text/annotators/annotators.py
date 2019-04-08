from abc import ABC, abstractmethod


class Annotator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def annotate(self, string_of_text):
        pass
