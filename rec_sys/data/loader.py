import csv

from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class DataLoader:

    ratings_path = '../../datasets/rating_matrix.csv'
    features_path = '../../datasets/items_feature_matrix.csv'
    items_names_path = '../../datasets/items_name.csv'

    def load_rating_matrix(self):

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

        ratings_dataset = Dataset.load_from_file(self.ratings_path, reader=reader)

        self.itemID_to_name = {}
        self.itemID_to_course_name = {}

        with open(self.items_names_path, newline='', encoding='ISO-8859-1') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                itemID = str(row[0])
                self.itemID_to_name[itemID] = str(row[2])
                self.itemID_to_course_name[itemID] = str(row[1])

        return ratings_dataset

    def get_items_features(self):
        features = defaultdict(list)
        with open(self.features_path, newline='', encoding='ISO-8859-1') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                itemID = str(row[0])
                features[itemID] = [float(row[i]) for i in range(1, len(row))]

        return features

    def get_user_ratings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratings_path, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    itemID = str(row[1])
                    rating = float(row[2])
                    userRatings.append((itemID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def get_popularity_ranks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                itemID = str(row[1])
                ratings[itemID] += 1
        rank = 1
        for itemID, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[itemID] = rank
            rank += 1
        return rankings

    def get_item_name(self, itemID):
        if itemID in self.itemID_to_name:
            return self.itemID_to_name[itemID]
        else:
            return ""

    def get_item_course_name(self, itemID):
        if itemID in self.itemID_to_course_name:
            return self.itemID_to_course_name[itemID]
        else:
            return ""