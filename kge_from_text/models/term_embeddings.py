"""
This module provides the functionalities to embed a text made of terms into a vector space of given dimensionality. The
defined class is a wrapper of the original gensim class.
"""

from gensim.models import word2vec
from pathlib import Path
import gensim
import os
from kge_from_text.utils.exceptions import *


class TermEmbedding():
    """
    Wrapper around the gensim class
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.complete_model_name = None

    def load_model(self, path_of_file):
        """
        Load mode in the self.model parameter, generate exception if a model has already been loaded

        :param path_of_file:
        :return:
        """
        # TODO: implement option: load even if already loaded
        if self.model is None:
            self.model = word2vec.Word2Vec.load(path_of_file)
            _, tail = os.path.split(path_of_file)
            self.complete_model_name = tail
        else:
            raise ModelAlreadyLoadedException

    def fit(self, input_text, output_file_path, _size=100, _window=5, _min_count=5, _workers=15, _sg=1, _iter=15, override_if_exists = False, load_model_if_exits= False):
        """
        Reads the file line by line and then feeds a word2vec model with given parameters. Produced models are saved
        in a local directory

        :param output_file_path:
        :param _size: size of the resulting space
        :param _window: window span considered for each target word
        :param _min_count:
        :param _workers: number of threads
        :param _sg: skip-gram or cbow, default is cbow
        :return:
        """

        self.complete_model_name = self.model_name + ":s" + str(_size) + ":w" + str(_window)
        output_file_path = output_file_path + self.complete_model_name

        # If file already exists and no override option raise an exception
        # If file already exists and load option is True, load the vector file
        check_file = Path(output_file_path)

        if check_file.is_file() and load_model_if_exits is True:
            self.model = word2vec.Word2Vec.load(output_file_path)
            return

        if check_file.is_file() and override_if_exists is False:
            raise ModelAlreadyExistsException

        with open(input_text) as f:
            content = f.readlines()
        sentences = [x.strip() for x in content]
        sentences = [gensim.utils.deaccent(x) for x in sentences]

        model = word2vec.Word2Vec([s.split() for s in sentences],
                                  size=_size,
                                  window=_window, min_count=_min_count,
                                  workers=_workers, sg=_sg, iter=_iter)

        model.save(output_file_path)

        self.model = model
        return
