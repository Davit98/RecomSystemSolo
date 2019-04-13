from rec_sys.data.loader import DataLoader
import pandas as pd

dl = DataLoader()
data = dl.load_rating_matrix()
rankings = dl.get_popularity_ranks()

print(dl.get_user_ratings(18700))

print(dl.get_item_name('L2269'))
