import pandas as pd
import itertools
from kge_from_text.utils import helpers
import numpy as np


class CleanBridge:
    """
    This class provides an interface for clean bridges. Input is returned
    """

    def get_back(self, entity):
        return entity

    def analogy_disambiguation(self, first, second, third):
        """
        This function implements a method that returns input

        :param first: first element of the analogy
        :param second: second element of the analogy
        :param third: third element of the analogy
        :return:
        """

        return first, second, third

