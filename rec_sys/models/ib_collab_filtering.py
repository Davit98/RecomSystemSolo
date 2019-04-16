from rec_sys.data.loader import DataLoader
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter


class ItemBasedCollaborativeFiltering:

    def __init__(self, k=10):
        self.k = k

    def fit(self):

        self.dl = DataLoader()
        data = self.dl.load_rating_matrix()

        self.train_set = data.build_full_trainset()

        sim_options = {'name': 'cosine', 'user_based': False}
        knn_basic = KNNBasic(sim_options=sim_options)
        knn_basic.fit(self.train_set)

        self.sim_matrix = knn_basic.compute_similarities()

    def get_top_n_recs(self, user, n):

        user_inner_id = self.train_set.to_inner_uid(str(user))
        user_ratings = self.train_set.ur[user_inner_id]

        k_neighbors = heapq.nlargest(self.k, user_ratings, key=lambda t: t[1])

        candidates = defaultdict(float)
        for item_id, rating in k_neighbors:
            row = self.sim_matrix[item_id]
            for inner_id, sim_score in enumerate(row):
                candidates[inner_id] += sim_score * (rating / 10.0)

        looked_items = set()
        for item_id, _ in self.train_set.ur[user_inner_id]:
            looked_items.add(item_id)

        count = 1
        for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if item_id not in looked_items:
                item_raw_id = self.train_set.to_raw_iid(item_id)
                print(item_raw_id, "-", self.dl.get_item_name(item_raw_id), "(" +
                      self.dl.get_item_course_name(item_raw_id) + ")", rating_sum)
                if count >= n:
                    break
                count += 1
