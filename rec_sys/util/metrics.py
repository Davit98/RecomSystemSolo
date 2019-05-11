import itertools

from surprise import accuracy
from collections import defaultdict


class Metrics:

    @staticmethod
    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    @staticmethod
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)

    @staticmethod
    def get_top_n(predictions, n=10, minimum_rating=0.0):
        top_n = defaultdict(list)

        for user_id, item_id, actual_rating, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                top_n[int(user_id)].append((str(item_id), estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n

    @staticmethod
    def hit_rate(top_n_predicted, left_out_predictions):
        hits = 0

        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_item_id = left_out[1]
            hit = False
            for item_id, predicted_rating in top_n_predicted[int(user_id)]:
                if str(left_out_item_id) == str(item_id):
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / len(left_out_predictions)

    @staticmethod
    def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cutoff=1.38):
        hits = 0

        for user_id, left_out_item_id, actual_rating, estimated_rating, _ in left_out_predictions:
            if actual_rating >= rating_cutoff:
                hit = False
                for item_id, predicted_rating in top_n_predicted[int(user_id)]:
                    if str(left_out_item_id) == item_id:
                        hit = True
                        break
                if hit:
                    hits += 1

        return hits / len(left_out_predictions)

    @staticmethod
    def rating_hit_rate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        for user_id, left_out_item_id, actual_rating, estimated_rating, _ in left_out_predictions:
            hit = False
            for item_id, predicted_rating in top_n_predicted[int(user_id)]:
                if str(left_out_item_id) == item_id:
                    hit = True
                    break
            if hit:
                hits[actual_rating] += 1

            total[actual_rating] += 1

        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    @staticmethod
    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
        summation = 0

        for user_id, left_out_item_id, actual_rating, estimated_rating, _ in left_out_predictions:
            hit_rank = 0
            rank = 0
            for item_id, predicted_rating in top_n_predicted[int(user_id)]:
                rank = rank + 1
                if str(left_out_item_id) == item_id:
                    hit_rank = rank
                    break
            if hit_rank > 0:
                summation += 1.0 / hit_rank

        return summation / len(left_out_predictions)

    # What percentage of users have at least one "good" recommendation
    @staticmethod
    def user_coverage(top_n_predicted, n_users, rating_threshold):
        hits = 0
        for user_id in top_n_predicted.keys():
            hit = False
            for item_id, predicted_rating in top_n_predicted[user_id]:
                if predicted_rating >= rating_threshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / n_users

    @staticmethod
    def diversity(top_n_predicted, sims_algo):
        n = 0
        total = 0
        sim_matrix = sims_algo.compute_similarities()
        for user_id in top_n_predicted.keys():
            pairs = itertools.combinations(top_n_predicted[user_id], 2)
            for pair in pairs:
                item1 = pair[0][0]
                item2 = pair[1][0]
                item1_inner_id = sims_algo.trainset.to_inner_iid(str(item1))
                item2_inner_id = sims_algo.trainset.to_inner_iid(str(item2))
                similarity = sim_matrix[item1_inner_id][item2_inner_id]
                total += similarity
                n += 1

        s = total / n
        return 1 - s

    @staticmethod
    def novelty(top_n_predicted, rankings):
        # min novelty = 5.5
        # max novelty = 1312.5
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for elem in top_n_predicted[user_id]:
                item_id = elem[0]
                rank = rankings[item_id]
                total += rank
                n += 1
        return total / n
