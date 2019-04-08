import pandas as pd
import itertools
from kge_from_text.utils import helpers
import numpy as np


class Bridge:
    """
    This class handles bridging from word to entities
    """
    def __init__(self, bridge_location, entity_to_type_location, model_entities, model_types):
        """
        :param bridge_location: location of the bridge file
        :param entity_to_type_location: location of the file that connects entity with types
        :param model_entities: embedding model of the entities
        :param model_types: embedding model of the types
        """
        self.bridge = pd.read_csv(bridge_location, sep="\t", names=["phrase", "entity", "count"])

        self.map_entity_type = pd.read_csv(entity_to_type_location, header=None, delimiter=r"\s+",
                                           names=["Subject", "Property", "Object", "Point"])

        # Remove unused columns
        self.map_entity_type = self.map_entity_type.drop('Property', 1)
        self.map_entity_type = self.map_entity_type.drop('Point', 1)

        self.map_entity_type = self.map_entity_type.set_index('Subject')

        # Remove duplicated types
        self.map_entity_type = self.map_entity_type[~self.map_entity_type.index.duplicated(keep='first')]
        self.map_entity_type = self.map_entity_type['Object'].to_dict()

        self.model_entities = model_entities
        self.mode_types = model_types

        self.dict_of_default_mappings = {"United_Kingdom" : "England"}

    def get_back(self, entity):
        """
        Given an entity, returns the matching word using the get back approach

        :param entity:
        :return:

        >>> self.get_back("Barack_Obama")
        "Barack Obama"

        >>> self.get_back("Paris,_Texas")
        "Paris"

        >>> self.get_back("Paris")
        "Paris"

        """
        if len(self.bridge[self.bridge["phrase"] == entity].values.tolist()) > 0:
            return self.bridge[self.bridge["phrase"] == entity]["phrase"].values[0]
        else:
            return self.bridge[self.bridge["entity"] == entity].sort_values(by=["count"],
                                                                            ascending=False)["phrase"].values[0]

    def find_and_count(self, entity):
        operation = self.bridge[self.bridge["phrase"] == entity]
        find = operation["entity"].values.tolist()
        count = 0
        if find:
            count = helpers.softmax(operation["count"].values.tolist())
        if len(find) == 0:
            inner_op = self.bridge[self.bridge["phrase"].str.contains(entity)]
            find = inner_op["entity"].values.tolist()
            count = helpers.softmax(inner_op["count"].values.tolist())
        ordering = zip(find, count)
        rrr = list(reversed(sorted(ordering, key=lambda x: x[1])))
        find, count = zip(*rrr)
        return find, count

    def analogy_disambiguation(self, first, second, third, weight_1=0.75, weight_2 = 0.25, multiplicative = False):
        """
        This function implements a method to disambiguate three elements using a cross-similarity approach.

        :param first: first element of the analogy
        :param second: second element of the analogy
        :param third: third element of the analogy
        :return:
        """
        first_code, first_count = self.find_and_count(first)
        second_code, second_count = self.find_and_count(second)
        third_code, third_count = self.find_and_count(third)

        first_code = zip(first_code, first_count)
        second_code = zip(second_code, second_count)
        third_code = zip(third_code, third_count)

        combinations = list(itertools.product(first_code, second_code, third_code))

        return_list = []
        for first_t, second_t, third_t in combinations:
            first = first_t[0]
            second = second_t[0]
            third = third_t[0]

            c_first = first_t[1]
            c_second = second_t[1]
            c_third = third_t[1]

            try:

                first_a = first
                second_a = second
                third_a = third

                sim = self.model_entities.similarity(first_a, second_a)
                sim_t = self.mode_types.similarity(self.entity_to_type(first_a), self.entity_to_type(third_a))

                if multiplicative:
                    w_factor = (c_first * c_second * c_third)
                else:
                    w_factor = (c_first + c_second + c_third)

                return_list.append(((first, second, third),
                                    np.average([sim, sim_t],
                                    weights=[weight_1, weight_2]) * w_factor,
                                    sim, sim_t, c_first, c_second, c_third))

            except Exception as e:
                pass

        return list(reversed(sorted(return_list, key=lambda x: x[1])))[0][0]


    def analogy_disambiguation_test(self, first, second, third, weight_1=0.75, weight_2 = 0.25, multiplicative = False, ignore_w = False):
        """
        This function implements a method to disambiguate three elements using a cross-similarity approach.

        :param first: first element of the analogy
        :param second: second element of the analogy
        :param third: third element of the analogy
        :return:
        """
        first_code, first_count = self.find_and_count(first)
        second_code, second_count = self.find_and_count(second)
        third_code, third_count = self.find_and_count(third)

        first_code = zip(first_code, first_count)
        second_code = zip(second_code, second_count)
        third_code = zip(third_code, third_count)

        combinations = list(itertools.product(first_code, second_code, third_code))

        return_list = []
        for first_t, second_t, third_t in combinations:
            first = first_t[0]
            second = second_t[0]
            third = third_t[0]

            c_first = first_t[1]
            c_second = second_t[1]
            c_third = third_t[1]

            try:

                first_a = first
                second_a = second
                third_a = third

                sim = self.model_entities.similarity(first_a, second_a)
                sim_t = self.mode_types.similarity(self.entity_to_type(first_a), self.entity_to_type(third_a))

                if multiplicative:
                    w_factor = (c_first * c_second * c_third)
                else:
                    w_factor = (c_first + c_second + c_third)

                if ignore_w:
                    w_factor = 1

                return_list.append(((first, second, third),
                                    np.average([sim, sim_t],
                                    weights=[weight_1, weight_2]) * w_factor,
                                    sim, sim_t, c_first, c_second, c_third))

            except Exception as e:
                pass

        return list(reversed(sorted(return_list, key=lambda x: x[1])))


    def entity_to_type(self, entity):
        """
        Function used to get the specific type for an entity

        :param entity: entity in input
        :return: the type of the entity, if found. Returns owl#Thing otherwise

        >>> self.entity_to_type("Barack_Obama")
        "Politician"

        """
        if entity in self.dict_of_default_mappings:
            return self.dict_of_default_mappings[entity]

        try:
            type_res = self.map_entity_type['<http://dbpedia.org/resource/' + entity + '>']
            type_res = type_res.replace(">", "")
            type_res = type_res.replace("<http://dbpedia.org/ontology/", "")
            type_res = type_res.replace("<http://www.w3.org/2002/07/", "")
            return type_res
        except:
            return "owl#Thing"

class TableLinkingBridge:
    """
    This class handles bridging from word to entities
    """
    def __init__(self, bridge_location, entity_to_type_location, model_entities, model_types, tee):
        """
        :param bridge_location: location of the bridge file
        :param entity_to_type_location: location of the file that connects entity with types
        :param model_entities: embedding model of the entities
        :param model_types: embedding model of the types
        """
        self.bridge = pd.read_csv(bridge_location, sep="\t", names=["phrase", "entity", "count"])

        self.map_entity_type = pd.read_csv(entity_to_type_location, header=None, delimiter=r"\s+",
                                           names=["Subject", "Property", "Object", "Point"])

        # Remove unused columns
        self.map_entity_type = self.map_entity_type.drop('Property', 1)
        self.map_entity_type = self.map_entity_type.drop('Point', 1)

        self.map_entity_type = self.map_entity_type.set_index('Subject')

        # Remove duplicated types
        self.map_entity_type = self.map_entity_type[~self.map_entity_type.index.duplicated(keep='first')]
        self.map_entity_type = self.map_entity_type['Object'].to_dict()

        self.model_entities = model_entities
        self.mode_types = model_types

        self.dict_of_default_mappings = {"United_Kingdom" : "England"}

        self.tee = tee

    def get_back(self, entity):
        """
        Given an entity, returns the matching word using the get back approach

        :param entity:
        :return:

        >>> self.get_back("Barack_Obama")
        "Barack Obama"

        >>> self.get_back("Paris,_Texas")
        "Paris"

        >>> self.get_back("Paris")
        "Paris"

        """
        if len(self.bridge[self.bridge["phrase"] == entity].values.tolist()) > 0:
            return self.bridge[self.bridge["phrase"] == entity]["phrase"].values[0]
        else:
            return self.bridge[self.bridge["entity"] == entity].sort_values(by=["count"],
                                                                            ascending=False)["phrase"].values[0]

    def find_and_count(self, entity):
        operation = self.bridge[self.bridge["phrase"] == entity]
        find = operation["entity"].values.tolist()
        count = 0
        if find:
            count = helpers.softmax(operation["count"].values.tolist())
        if len(find) == 0:
            inner_op = self.bridge[self.bridge["phrase"].str.contains(entity)]
            find = inner_op["entity"].values.tolist()
            count = helpers.softmax(inner_op["count"].values.tolist())
        ordering = zip(find, count)
        rrr = list(reversed(sorted(ordering, key=lambda x: x[1])))
        find, count = zip(*rrr)
        return find, count

    def analogy_disambiguation(self, first, second, third, fourth, weight_1=0.25, weight_2 = 0.25,
                               weight_3 = 0.25, weight_4 = 0.25, multiplicative = False):
        """
        This function implements a method to disambiguate three elements using a cross-similarity approach.

        :param first: first element of the analogy
        :param second: second element of the analogy
        :param third: third element of the analogy
        :return:
        """
        first_code, first_count = self.find_and_count(first)
        second_code, second_count = self.find_and_count(second)
        third_code, third_count = self.find_and_count(third)
        fourth_code, fourth_count = self.find_and_count(fourth)

        first_code = zip(first_code, first_count)
        second_code = zip(second_code, second_count)
        fourth_code = zip(fourth_code, fourth_count)

        combinations = list(itertools.product(first_code, second_code, third_code, fourth_code))

        return_list = []
        for first_t, second_t, third_t, fourth_t in combinations:
            first = first_t[0]
            second = second_t[0]
            third = third_t[0]
            fourth = fourth_t[0]

            c_first = first_t[1]
            c_second = second_t[1]
            c_third = third_t[1]
            c_fourth = fourth_t[1]

            try:

                first_a = first
                second_a = second
                third_a = third
                fourth_a = fourth

                sim_a_b = self.model_entities.similarity(first_a, second_a)
                sim_t_a_c = self.mode_types.similarity(self.entity_to_type(first_a), self.entity_to_type(third_a))

                sim_c_d = self.model_entities.similarity(third_a, fourth_a)
                sim_t_b_d = self.mode_types.similarity(self.entity_to_type(second_a), self.entity_to_type(fourth_a))


                if multiplicative:
                    w_factor = (c_first * c_second * c_third * c_fourth)
                else:
                    w_factor = (c_first + c_second + c_third + c_fourth)

                first_analogy_factor = self.tee[first_a] - self.tee[second_a]
                second_analogy_factor = self.tee[third_a] - self.tee[fourth_a]

                return_list.append(((first, second, third),
                                    np.average([sim_a_b, sim_t_a_c,
                                                sim_c_d, sim_t_b_d,
                                                first_analogy_factor, second_analogy_factor],
                                    weights=[weight_1,
                                             weight_2,
                                             weight_3,
                                             weight_4,
                                             ]) * w_factor,
                                    first_analogy_factor, second_analogy_factor,
                                    c_first, c_second, c_third, c_fourth))

            except Exception as e:
                print(e)
                pass

        return list(reversed(sorted(return_list, key=lambda x: x[1])))

    def entity_to_type(self, entity):
        """
        Function used to get the specific type for an entity

        :param entity: entity in input
        :return: the type of the entity, if found. Returns owl#Thing otherwise

        >>> self.entity_to_type("Barack_Obama")
        "Politician"

        """
        if entity in self.dict_of_default_mappings:
            return self.dict_of_default_mappings[entity]

        try:
            type_res = self.map_entity_type['<http://dbpedia.org/resource/' + entity + '>']
            type_res = type_res.replace(">", "")
            type_res = type_res.replace("<http://dbpedia.org/ontology/", "")
            type_res = type_res.replace("<http://www.w3.org/2002/07/", "")
            return type_res
        except:
            return "owl#Thing"
