from rec_sys.data.loader import DataLoader
from rec_sys.util.evaluator import Evaluator
from rec_sys.models.content_based import ContentBasedAlgo

from surprise import NormalPredictor

dl = DataLoader()
print("Loading item ratings...")
data = dl.load_rating_matrix()
print("\nComputing item popularity ranks so we can measure novelty later...")
rankings = dl.get_popularity_ranks()


evaluator = Evaluator(data, rankings)

content_knn = ContentBasedAlgo()
evaluator.add_algo(content_knn,"ContentBased")

random_algo = NormalPredictor()
evaluator.add_algo(random_algo, "Random")

evaluator.evaluate(False)

evaluator.sample_top_n_recs(dl, user=12753303)








