from surprise import AlgoBase
from surprise import PredictionImpossible
from rec_sys.data.loader import DataLoader
import math
import numpy as np
import heapq


class ContentBasedAlgo(AlgoBase):

    def __init__(self, k=10, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, train_set):
        AlgoBase.fit(self, train_set)

        dl = DataLoader()
        item_features = dl.get_items_features()

        print("Computing content-based similarity matrix...")

        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for this_inner_id in range(self.trainset.n_items):
            if this_inner_id % 100 == 0:
                print(this_inner_id, " of ", self.trainset.n_items)
            for next_inner_id in range(this_inner_id + 1, self.trainset.n_items):
                this_raw_id = str(self.trainset.to_raw_iid(this_inner_id))
                next_raw_id = str(self.trainset.to_raw_iid(next_inner_id))
                similarity = self.compute_cos_sim(this_raw_id, next_raw_id, item_features)
                self.similarities[this_inner_id, next_inner_id] = similarity
                self.similarities[next_inner_id, this_inner_id] = self.similarities[this_inner_id, next_inner_id]

        print("...done.")

        return self

    def compute_cos_sim(self, item1_id, item2_id, item_features):
        item1_features = np.array(item_features[item1_id])
        item2_features = np.array(item_features[item2_id])
        return np.dot(item1_features,item2_features)/(np.linalg.norm(item1_features)*np.linalg.norm(item2_features))

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i, rating[0]]
            neighbors.append((genre_similarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        sim_total = weighted_sum = 0
        for (sim_score, rating) in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating

        if sim_total == 0:
            raise PredictionImpossible('No neighbours')

        predicted_rating = weighted_sum / sim_total

        return predicted_rating
