from rec_sys.data.loader import DataLoader
from rec_sys.util.evaluator import Evaluator
from rec_sys.models.content_based import ContentBasedAlgo
from rec_sys.models.ub_collab_filtering import UserBasedCollaborativeFiltering
from rec_sys.models.ib_collab_filtering import ItemBasedCollaborativeFiltering

from surprise import NormalPredictor, KNNBasic, SVD, SVDpp

dl = DataLoader()
data = dl.load_rating_matrix()
rankings = dl.get_popularity_ranks()

dl.print_user_items_info(user_id=12753303, sort=True)

evaluator = Evaluator(data, rankings)

# k = 5, 20
content_knn = ContentBasedAlgo(k=20)
evaluator.add_algo(content_knn,"ContentBased")

# random_algo = NormalPredictor()
# evaluator.add_algo(random_algo, "Random")

evaluator.evaluate(True)

evaluator.sample_top_n_recs(dl, user=12753303)

