import csv

from surprise import Dataset
from surprise import Reader

from collections import defaultdict


class DataLoader:
    ratings_path = '../../datasets/rating_matrix.csv'
    features_path = '../../datasets/items_feature_matrix.csv'
    items_names_path = '../../datasets/items_name.csv'

    def __init__(self):
        self.item_id_to_name = {}
        self.item_id_to_course_name = {}

    def load_rating_matrix(self):

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

        ratings_dataset = Dataset.load_from_file(self.ratings_path, reader=reader)

        with open(self.items_names_path, newline='', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                item_id = str(row[0])
                self.item_id_to_name[item_id] = str(row[2])
                self.item_id_to_course_name[item_id] = str(row[1])

        return ratings_dataset

    def get_items_features(self):
        features = defaultdict(list)
        with open(self.features_path, newline='', encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                item_id = str(row[0])
                features[item_id] = [float(row[i]) for i in range(1, len(row))]

        return features

    def get_user_ratings(self, user):
        user_ratings = []
        hit_user = False
        with open(self.ratings_path, newline='') as csv_file:
            rating_reader = csv.reader(csv_file)
            next(rating_reader)
            for row in rating_reader:
                user_id = int(row[0])
                if user == user_id:
                    item_id = str(row[1])
                    rating = float(row[2])
                    user_ratings.append((item_id, rating))
                    hit_user = True
                if hit_user and (user != user_id):
                    break

        return user_ratings

    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csv_file:
            rating_reader = csv.reader(csv_file)
            next(rating_reader)
            for row in rating_reader:
                item_id = str(row[1])
                ratings[item_id] += 1
        rank = 1
        for item_id, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[item_id] = rank
            rank += 1
        return rankings

    def get_item_name(self, item_id):
        if item_id in self.item_id_to_name:
            return self.item_id_to_name[item_id]
        else:
            return ""

    def get_item_course_name(self, item_id):
        if item_id in self.item_id_to_course_name:
            return self.item_id_to_course_name[item_id]
        else:
            return ""

    def print_user_items_info(self, user_id, sort=False):
        item_ratings = self.get_user_ratings(user_id)
        if sort:
            item_ratings.sort(key = lambda x: x[1], reverse=True)
        for item_id, rating in item_ratings:
            print(item_id, "-", self.get_item_name(item_id), "(" + self.get_item_course_name(item_id) + ")", rating)