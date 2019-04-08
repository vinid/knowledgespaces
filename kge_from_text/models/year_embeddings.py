import os
import numpy as np
import gensim

class YearTextEmbeddings:
    """
    Representation of the year using average base embeddings of entities
    """
    def __init__(self, entity_embeddings, years_folder, output_file, tfidf = False):
        self.entity_embeddings = entity_embeddings
        self.years_folder = years_folder
        self.output_file = output_file
        self.tfidf = tfidf
        self.entities_inside_year = dict()
        self.model = None

    def get_entities_of_years(self, year):
        """
        For a given year, retrieves the list of the entities contained inside that year. This method will load
        all the entities for all the years once for all.

        :param year:
        :return:
        """
        if self.entities_inside_year is False:
            for file_name in os.listdir(self.years_folder):
                path = (os.path.join(self.years_folder, file_name))
                with open(path, "r") as filino:
                    read_entities = filino.readlines()
                    entities = read_entities[0].split()
                    self.entities_inside_year[file_name] = entities

        return self.entities_inside_year[year]

    def fit(self):

        with open(self.output_file, "w") as file_with_output_model:

            folders = os.listdir(self.years_folder)
            num_files = len(folders)
            dimensions = len(self.entity_embeddings[self.entity_embeddings.wv.vocab[0]])

            file_with_output_model.write(str(num_files) + " " + str(dimensions) + " " + "\n")
            for name in folders:

                path = (os.path.join(self.years_folder, name))
                with open(path, "r") as filino:
                    coso = filino.readlines()
                    entities = (coso[0].split())
                    collect_embeddings = []

                    for entity in set(entities):
                        try:
                            time_array = self.entity_embeddings[entity]
                            collect_embeddings.append(time_array)
                        except:
                            continue
                    collect_embeddings = np.average(collect_embeddings, axis=0)
                    embedding_list = collect_embeddings.tolist()
                    string_to_save = ' '.join(map(str, embedding_list))
                    self.output_file.write(str(name) + " " + string_to_save + "\n")

        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.output_file)



