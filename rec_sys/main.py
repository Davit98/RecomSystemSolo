from rec_sys.data.loader import DataLoader
from rec_sys.util.evaluator import Evaluator
from rec_sys.models.content_based import ContentBasedAlgo
from rec_sys.models.ub_collab_filtering import UserBasedCollaborativeFiltering
from rec_sys.models.ib_collab_filtering import ItemBasedCollaborativeFiltering

import random
import numpy as np

from surprise import NormalPredictor, KNNBasic, SVD

dl = DataLoader()
data = dl.load_rating_matrix()
rankings = dl.get_popularity_ranks()

dl.print_user_items_info(user_id=12753303, sort=True)

evaluator = Evaluator(data, rankings)

# Content-based KNN
content_knn = ContentBasedAlgo(k=5)
evaluator.add_algo(content_knn,"ContentBased")

# User-based KNN
user_knn = KNNBasic(k=20, sim_options={'name': 'cosine', 'user_based': True})
evaluator.add_algo(user_knn, "User KNN")

# Item-based KNN
item_knn = KNNBasic(k=10, sim_options={'name': 'cosine', 'user_based': False})
evaluator.add_algo(item_knn, "Item KNN")

np.random.seed(0)
random.seed(0)

# SVD
svd = SVD(reg_all=0.5)
evaluator.add_algo(svd, "SVD", tt_split=True)

# Random
random_algo = NormalPredictor()
evaluator.add_algo(random_algo, "Random")

evaluator.evaluate(False)

evaluator.sample_top_n_recs(dl, user=12753303)


#--------------------------------------------------------------------------------

# # User-Based Collaborative Filtering
# ub_model = UserBasedCollaborativeFiltering(k=20)
# ub_model.fit()
# ub_model.get_top_n_recs(user=12753303, n=10)

# # Item-Based Collaborative Filtering
# ib_model = ItemBasedCollaborativeFiltering(k=10)
# ib_model.fit()
# ib_model.get_top_n_recs(user=12753303, n=10)
