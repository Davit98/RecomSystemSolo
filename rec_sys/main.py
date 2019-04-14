from rec_sys.data.loader import DataLoader
from rec_sys.util.evaluator import Evaluator
from rec_sys.models.rating_predictors.content_based import ContentBasedAlgo
from rec_sys.models.ub_collab_filtering import UserBasedCollaborativeFiltering
from rec_sys.models.ib_collab_filtering import ItemBasedCollaborativeFiltering

from surprise import NormalPredictor

# dl = DataLoader()
# print("Loading item ratings...")
# data = dl.load_rating_matrix()
# print("\nComputing item popularity ranks so we can measure novelty later...")
# rankings = dl.get_popularity_ranks()
#
# evaluator = Evaluator(data, rankings)
#
# content_knn = ContentBasedAlgo()
# evaluator.add_algo(content_knn,"ContentBased")
#
# random_algo = NormalPredictor()
# evaluator.add_algo(random_algo, "Random")
#
# evaluator.evaluate(False)
#
# evaluator.sample_top_n_recs(dl, user=12753303)

# ub_model = UserBasedCollaborativeFiltering()
# ub_model.fit()
# ub_model.get_top_n_recs(user=12753303, n=10)

ib_model = ItemBasedCollaborativeFiltering()
ib_model.fit()
ib_model.get_top_n_recs(user=12753303, n=10)







