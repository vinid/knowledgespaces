import pandas as pd
from kge_from_text import folder_definitions as fd


class EntityLinkingEvaluator:
    """
    This class implements an evaluator for entity linking.
    """

    def __init__(self, bridge, model, to_test, gold_standard):
        """

        :param bridge: bridge to be used if necessary to connect word with KGs
        :param model: model used for analogy resolution
        :param to_test: pandas dataframe containing analogies
        """
        self.bridge = bridge
        self.model = model
        self.gold_standard = gold_standard
        self.to_test = to_test

    def solve(self):
        """
        Evaluation method

        :return:
        """
        correct = 0
        uncorrect = 0
        total_number = 0
        dbg = open("debug", "w")

        result = pd.concat([self.to_test, self.gold_standard], axis=1)

        for index, row in result.iterrows():
            first = row['First']
            second = row['Second']
            third = row['Third']
            fourth = row['Fourth']

            t_first = row['T_First']
            t_second = row['T_Second']
            t_third = row['T_Third']
            t_fourth = row['T_Fourth']

            dbg.write(first + " " + second + " " + third + " " + fourth + " ")
            first, second, third, fourth = self.bridge.entity_linking_disambiguation(first, second, third)

            dbg.write(predict + " ")

            if fourth == predicted:
                correct = correct + 1
                dbg.write(str(1) + "\n")
            else:
                uncorrect = uncorrect + 1
                dbg.write(str(0) + "\n")
            dbg.flush()
            total_number = total_number + 1

            if(total_number%500 == 0):
                dbg.write(str(correct / total_number))
        dbg.write(str(correct / total_number))


        return correct / total_number, total_number
