import pandas as pd
from kge_from_text import folder_definitions as fd

class AnalogyEvaluator:
    """
    This class implements an evaluator for analogies.
    """

    def __init__(self, bridge, model, analogies):
        """

        :param bridge: bridge to be used if necessary to connect word with KGs
        :param model: model used for analogy resolution
        :param analogies: pandas dataframe containing analogies
        """
        self.bridge = bridge
        self.model = model
        self.analogies = analogies

    def solve(self):
        """
        Evaluation method

        :return:
        """
        correct = 0
        uncorrect = 0
        total_number = 0
        dbg = open("debug", "w")
        for index, row in self.analogies.iterrows():
            second = row['Second']
            first = row['First']
            third = row['Third']
            fourth = row['Fourth']
            dbg.write(first + " "+ second + " "+ third + " " + fourth + " ")
            first, second, third = self.bridge.analogy_disambiguation(first, second, third)

            predict = self.model.model.wv.most_similar(positive=[second, third], negative=[first])[0][0]

            try:
                predicted = self.bridge.get_back(predict)
            except:
                predicted = "wrong_answer_123"

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
