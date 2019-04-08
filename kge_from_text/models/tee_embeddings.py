"""
This class is used to concatenate the models containing entities with the model containing types.
"""

import gensim
import numpy as np


import pandas as pd
from kge_from_text.utils.exceptions import *
import os
class TeeEmbedding:
    """
    The ConcatenateModels concatenate each entity vector to the respective type vector. Concatentation can be weighted
    using a lambada parameter.
    """
    def __init__(self, _model_name):
        """
        Initialize the class used to concatenate the entity model and the type model

        :param model_name
        """
        self.model_name = _model_name
        self.model = None
        self.complete_model_name = None

    def load_model(self, path_of_file):
        if self.model is None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path_of_file)
            _, tail = os.path.split(path_of_file)
            self.complete_model_name = tail
        else:
            raise ModelAlreadyLoadedException

    def fit(self, output_path, entity_model, type_model, _types_file, type_lambda = 1):
        """
        Method that concatenate the models

        :return: model: the effective model that has been concatenated
        """

        df = pd.read_csv(_types_file, header=None, delimiter=r"\s+",
                         names=["Subject", "Property", "Object", "ToRemove"])

        # Remove unused columns
        df = df.drop('Property', 1)
        df = df.drop('ToRemove', 1)

        df = df.set_index('Subject')

        # Remove duplicated types
        df = df[~df.index.duplicated(keep='first')]
        dictionary = df['Object'].to_dict()
        voc_e = [k for k in entity_model.model.wv.vocab]
        voc_t = [k for k in type_model.model.wv.vocab]

        total_size = len(entity_model.model[voc_e[0]]) + len(type_model.model[voc_t[0]])

        self.complete_model_name = self.model_name + entity_model.complete_model_name + type_model.complete_model_name

        with open(output_path + self.complete_model_name, "w") as text_file:
            text_file.write(str(len(entity_model.model.wv.vocab)) + " " + str(total_size) + " " + "\n")
            for word, obj in entity_model.model.wv.vocab.items():
                try:
                    type = dictionary['<http://dbpedia.org/resource/' + word + '>']
                    type = type.replace(">", "")
                    type = type.replace("<http://dbpedia.org/ontology/", "")
                    type = type.replace("<http://www.w3.org/2002/07/", "")
                    type_array = type_model.model[type]
                except:
                    type_array = type_model.model["owl#Thing"]

                type_array = type_array*type_lambda

                entity_array = entity_model.model[word]
                concatenated_array = np.concatenate((type_array, entity_array))
                concatenated_list = concatenated_array.tolist()

                string_to_save = ' '.join(map(str, concatenated_list))
                text_file.write(word + " " + string_to_save + "\n")
            text_file.flush()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(output_path + self.complete_model_name)



