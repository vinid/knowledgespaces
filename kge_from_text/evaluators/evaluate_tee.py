if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.getcwd() + "/../../")

    import pandas as pd
    import itertools
    from kge_from_text import folder_definitions as fd
    import kge_from_text.models.term_embeddings as tt
    import kge_from_text.bridges.bridge as bridge
    from kge_from_text.evaluators.evaluator_handler import EvaluatorHandler
    from kge_from_text.evaluators.analogy_evaluator import AnalogyEvaluator
    import kge_from_text.models.tee_embeddings as tee

    window_e = [3, 5, 10]
    size_e = [100, 200]

    window_t = [3, 5, 10]
    size_t = [100, 200]

    entity_vector_name = "2016_data/entity_vectors"
    type_vector_name = "2016_data/type_vectors"
    conactenated_name = "2016_data/concatenated_vectors"
    conactenated_name_time = "2016_data/concatenated_vectors_time"
    temporal_csv = "2016_data/temporal_vectors.csv"

    annotated_entity_file = "2016_data/annotated_text_with_entities"
    annotated_type_file = "2016_data/annotated_text_with_types"
    type_of_entity_file = "2016_data/type_to_entity_data.ttl"

    c_e = list(itertools.product(window_e, size_e))
    c_t = list(itertools.product(window_t, size_t))

    # Declare An Evaluator
    evalu = EvaluatorHandler(fd.EVALUATION_RESULTS_ROOT)

    for w_e, s_e in c_e:

        # ENTITY
        model_e = tt.TermEmbedding("entity")
        model_e.fit(input_text=fd.STARTING_DATA_ROOT + annotated_entity_file,
                    output_file_path=fd.PRODUCED_MODELS_ROOT + "2016_data/", _size=s_e, _window=w_e, load_model_if_exits = True)

        for w_t, s_t in c_t:
            # TYPE
            model_t = tt.TermEmbedding("type")
            model_t.fit(input_text=fd.STARTING_DATA_ROOT + annotated_type_file,
                        output_file_path=fd.PRODUCED_MODELS_ROOT + "2016_data/",
            _size = s_e, _window = w_t, load_model_if_exits=True)

            analogies = pd.read_csv(fd.GOLD_STANDARDS + "mikolov", names=["First", "Second", "Third", "Fourth"],
                                    sep=" ")
            br = bridge.Bridge(fd.STARTING_DATA_ROOT + "2016_data/spotlight_dbpedia_bridge",
                               fd.STARTING_DATA_ROOT + type_of_entity_file, model_e.model, model_t.model)

            model_tee = tee.TeeEmbedding("tee")
            model_tee.fit(fd.PRODUCED_MODELS_ROOT + "2016_data/", entity_model=model_e, type_model=model_t,
                          _types_file=fd.STARTING_DATA_ROOT + type_of_entity_file)

            analogy_eval = AnalogyEvaluator(br, model_tee, analogies)

            evalu.run_evaluation(analogy_eval)

            analogies = pd.read_csv(fd.GOLD_STANDARDS + "currency", names=["First", "Second", "Third", "Fourth"],
                                    sep=" ")

            analogy_eval = AnalogyEvaluator(br, model_tee, analogies)

            evalu.run_evaluation(analogy_eval)
