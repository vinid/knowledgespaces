from scipy.spatial.distance import cosine
import numpy as np
import itertools
import warnings

class TemporalMediator:

    def __init__(self, model_e, temporal_model):
        self.model_e = model_e
        self.temporal_model = temporal_model
        self.max_time_sim, self.min_time_sim = self.compute_min_max_sim()
        warnings.warn("Warning. This module only uses gensim based embeddings. Not the wrapped kind.")

    def number_of_common_elements(self, list_a, list_b):
        return len(set(list_a).intersection(set(list_b))) / len(set(list_a))

    def usual_nearest(self, entity, nearest=100, topn=10):
        return [k[0] for k in self.model_e.most_similar(positive=[entity], topn=nearest)][:topn]

    def atemporal_nearest(self, entity, nearest=100, topn=10, alpha=0.7):
        new_list = []
        pair_match = []
        for k in self.model_e.most_similar(positive=[entity], topn=nearest):
            score = self.atemporal_similarity(entity, k[0], alpha=alpha)
            pair_match.append((k[0], score))

        res = list(reversed(sorted(pair_match, key=lambda tup: tup[1])))
        for k in res[:topn]:
            try:
                new_list.append(k[0])
            except Exception as e:
                pass
        return new_list

    def atemporal_similarity(self, entity_a, entity_b, alpha=0.5, n_years = 1):
        entity_array_a = self.model_e[entity_a]
        entity_array_b = self.model_e[entity_b]

        year_a = np.average(list(map(int, ([k[0] for k in self.temporal_model.most_similar(positive=[entity_array_a], topn=n_years)]))))
        year_b = np.average(list(map(int, ([k[0] for k in self.temporal_model.most_similar(positive=[entity_array_b], topn=n_years)]))))

        year_array_a = self.temporal_model[str(year_a)[:-2]] # :-2 because it's a float on a string "1922.0"
        year_array_b = self.temporal_model[str(year_b)[:-2]]

        a = (1 - cosine(entity_array_a, entity_array_b))
        b = (1 - (cosine(year_array_a, year_array_b)))

        new_b = (b - self.min_time_sim) / (self.max_time_sim - self.min_time_sim)

        return alpha * a - (1 - alpha) * (new_b)  # , a, new_b

    def compute_min_max_sim(self):
        vocab = [k for k in self.temporal_model.wv.vocab]
        comb = itertools.combinations(vocab, 2)
        list_of_similarities = list(map(lambda x: self.temporal_model.similarity(x[0], x[1]), comb))
        return max(list(list_of_similarities)), min(list(list_of_similarities))

