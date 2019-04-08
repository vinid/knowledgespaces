import time


class EvaluatorHandler:
    """
    Handler for evaluation, runs an experiment and save results
    """

    def __init__(self, log_file, name="no_name"):
        self.log_file = open(log_file + "_" + name + "_" + time.strftime("%Y%m%d-%H%M%S"), "a+")

    def run_evaluation(self, evaluator):
        precision, total_number = evaluator.solve()
        self.log_file.write(evaluator.model.complete_model_name + ", " + str(precision) + "\n")
        self.log_file.flush()


