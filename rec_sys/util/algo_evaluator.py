from rec_sys.util.metrics import Metrics


class AlgoEvaluator:

    def __init__(self, algorithm, name, tt_split):
        self.algorithm = algorithm
        self.name = name
        self.tt_split = tt_split

    def evaluate(self, data, do_top_n, n=10, verbose=True):
        metrics = {}
        if verbose:
            print("Evaluating accuracy...")

        if self.tt_split:
            train_set = data.get_train_set()
            self.algorithm.fit(train_set)
            train_set_list = [(train_set.to_raw_uid(uid),train_set.to_raw_iid(iid),rating) for uid, iid, rating in
                              train_set.all_ratings()]
            print('RMSE on Train' + str(Metrics.rmse(self.algorithm.test(train_set_list))))
            print('MAE on Train' + str(Metrics.mae(self.algorithm.test(train_set_list))))
            predictions = self.algorithm.test(data.get_test_set())

        else:
            train_set = data.get_full_train_set()
            self.algorithm.fit(train_set)
            train_set_list = [(train_set.to_raw_uid(uid), train_set.to_raw_iid(iid), rating) for uid, iid, rating in
                              train_set.all_ratings()]
            predictions = self.algorithm.test(train_set_list)

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
            metrics["HR"] = Metrics.hit_rate(top_n_predicted, left_out_predictions)
            # See how often we recommended an item the user actually liked
            metrics["cHR"] = Metrics.cumulative_hit_rate(top_n_predicted, left_out_predictions)
            # Compute ARHR
            metrics["ARHR"] = Metrics.average_reciprocal_hit_rank(top_n_predicted, left_out_predictions)

            # Evaluate properties of recommendations on full training set
            if verbose:
                print("Computing recommendations with full data set...")
            self.algorithm.fit(data.get_full_train_set())
            all_predictions = self.algorithm.test(data.get_full_anti_test_set())
            top_n_predicted = Metrics.get_top_n(all_predictions, n)
            if verbose:
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 1.73 (Q3 of ratings):
            metrics["Coverage"] = Metrics.user_coverage(top_n_predicted,
                                                        data.get_full_train_set().n_users,
                                                        rating_threshold=1.73)
            # Measure diversity of recommendations:
            metrics["Diversity"] = Metrics.diversity(top_n_predicted,
                                                     data.get_sims_algo())

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = Metrics.novelty(top_n_predicted,
                                                 data.get_popularity_rankings())

        if verbose:
            print("Analysis complete.")

        return metrics

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm
