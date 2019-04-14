import itertools

from surprise import accuracy
from collections import defaultdict

class Metrics:

    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def get_top_n(predictions, n=10, minimum_rating=0.0):
        top_n = defaultdict(list)

        for userID, itemID, actualRating, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                top_n[int(userID)].append((str(itemID), estimated_rating))

        for userID, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(userID)] = ratings[:n]

        return top_n

    def HitRate(top_n_predicted, left_out_predictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in left_out_predictions:
            userID = leftOut[0]
            leftOutItemID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for itemID, predictedRating in top_n_predicted[int(userID)]:
                if (str(leftOutItemID) == str(itemID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for itemID, predictedRating in topNPredicted[int(userID)]:
                    if (str(leftOutItemID) == itemID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for itemID, predictedRating in topNPredicted[int(userID)]:
                if (str(leftOutItemID) == itemID):
                    hit = True
                    break
            if hit:
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutItemID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for itemID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (str(leftOutItemID) == itemID):
                    hitRank = rank
                    break
            if hitRank > 0:
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for itemID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                item1 = pair[0][0]
                item2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(item1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(item2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                itemID = rating[0]
                rank = rankings[itemID]
                total += rank
                n += 1
        return total / n