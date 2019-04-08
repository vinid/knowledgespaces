if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.getcwd() + "/../../")

    import pandas as pd
    import itertools
    from kge_from_text import folder_definitions as fd
    import kge_from_text.models.term_embeddings as tt
    import kge_from_text.bridges.clean_bridge as bridge
    from kge_from_text.evaluators.evaluator_handler import EvaluatorHandler
    from kge_from_text.evaluators.analogy_evaluator import AnalogyEvaluator
    import kge_from_text.models.tee_embeddings as tee

    combinations = [(5, 400), (5, 500)]


    entity_vector_name = "2016_data/entity_vectors"
    type_vector_name = "2016_data/type_vectors"
    conactenated_name = "2016_data/concatenated_vectors"
    conactenated_name_time = "2016_data/concatenated_vectors_time"
    temporal_csv = "2016_data/temporal_vectors.csv"

    annotated_entity_file = "2016_data/annotated_text_with_entities"
    annotated_type_file = "2016_data/annotated_text_with_types"
    type_of_entity_file = "2016_data/type_to_entity_data.ttl"

    pure_text_model = "2016_data/text_with_words"

    # Declare An Evaluator
    evalu = EvaluatorHandler(fd.EVALUATION_RESULTS_ROOT, name="word_base")

    for w_e, s_e in combinations:

        # ENTITY
        model_w = tt.TermEmbedding("text")
        model_w.fit(input_text=fd.STARTING_DATA_ROOT + pure_text_model,
                    output_file_path=fd.PRODUCED_MODELS_ROOT + "2016_data/", _size=s_e, _window=w_e, load_model_if_exits = True)


        analogies = pd.read_csv(fd.GOLD_STANDARDS + "mikolov", names=["First", "Second", "Third", "Fourth"],
                                sep=" ")
        br = bridge.CleanBridge()

        analogy_eval = AnalogyEvaluator(br, model_w, analogies)

        evalu.run_evaluation(analogy_eval)

        analogies = pd.read_csv(fd.GOLD_STANDARDS + "currency", names=["First", "Second", "Third", "Fourth"],
                                sep=" ")

        analogy_eval = AnalogyEvaluator(br, model_w, analogies)

        evalu.run_evaluation(analogy_eval)
