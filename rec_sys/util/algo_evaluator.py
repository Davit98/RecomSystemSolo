from rec_sys.util.metrics import Metrics

class AlgoEvaluator:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def evaluate(self, data, do_top_n, n=10, verbose=True):
        metrics = {}
        if verbose:
            print("Evaluating accuracy...")
        self.algorithm.fit(data.get_train_set())
        predictions = self.algorithm.test(data.get_test_set())
        metrics["RMSE"] = Metrics.rmse(predictions)
        metrics["MAE"] = Metrics.mae(predictions)

        if do_top_n:
            # Evaluate top-10 with Leave One Out testing
            if verbose:
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(data.get_loocv_train_set())
            left_out_predictions = self.algorithm.test(data.get_loocv_test_set())
            # Build predictions for all ratings not in the training set
            all_predictions = self.algorithm.test(data.get_loocv_anti_test_set())
            # Compute top 10 recs for each user
            top_n_predicted = Metrics.get_top_n(all_predictions, n)
            if verbose:
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended an item the user actually rated
            metrics["HR"] = Metrics.HitRate(top_n_predicted, left_out_predictions)
            # See how often we recommended an item the user actually liked
            metrics["cHR"] = Metrics.CumulativeHitRate(top_n_predicted, left_out_predictions)
            # Compute ARHR
            metrics["ARHR"] = Metrics.AverageReciprocalHitRank(top_n_predicted, left_out_predictions)

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(data.get_full_train_set())
            all_predictions = self.algorithm.test(data.get_full_anti_test_set())
            top_n_predicted = Metrics.get_top_n(all_predictions, n)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = Metrics.UserCoverage(top_n_predicted,
                                                       data.get_full_train_set().n_users,
                                                       ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = Metrics.Diversity(top_n_predicted, data.get_similarities())

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = Metrics.Novelty(top_n_predicted,
                                                 data.get_popularity_rankings())

        if verbose:
            print("Analysis complete.")

        return metrics

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm