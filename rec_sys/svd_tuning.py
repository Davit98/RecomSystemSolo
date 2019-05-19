from rec_sys.data.loader import DataLoader
from rec_sys.util.evaluator import Evaluator

from surprise import SVD
from surprise.model_selection import GridSearchCV

import random
import numpy as np

dl = DataLoader()
data = dl.load_rating_matrix()
rankings = dl.get_popularity_ranks()

dl.print_user_items_info(user_id=12753303, sort=True)

param_grid = {'n_factors': [50, 100, 150, 200],
              'n_epochs': [20, 50, 80],
              'lr_all': [0.002, 0.005, 0.01],
              'reg_all': [0.02, 0.06, 0.1, 0.5, 1]}

np.random.seed(0)
random.seed(0)

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=1)
gs.fit(data)

print("RMSE score: ", gs.best_score['rmse'])
print("MAE score: ", gs.best_score['mae'])

print(gs.best_params['rmse'])
print('--------------------------------------------')
print(gs.best_params['mae'])

evaluator = Evaluator(data, rankings)

params = gs.best_params['rmse']

np.random.seed(0)
random.seed(0)

svd = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'], reg_all=params['reg_all'])
evaluator.add_algo(svd, "Tuned SVD", tt_split=True)

evaluator.evaluate(True)

evaluator.sample_top_n_recs(dl, user=12753303)
