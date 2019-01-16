def single_model():
    
    classifier = AdaBoostClassifier(
            n_estimators = 20,
            learning_rate = 0.75,
            base_estimator = ExtraTreesClassifier(
                n_estimators = 400,
                max_features = 30,
                max_depth = 12,
                min_samples_leaf = 100,
                min_samples_split = 100,
                verbose = 1,
                n_jobs = -1))

    return classifier