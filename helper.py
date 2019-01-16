def train_and_predict(X, W, Y, X_test):
    X, X_test = preprocess(X, X_test)

    classifier = single_model()
    classifier.fit(X, Y, sample_weight = W)

    Y_pred = classifier.predict_proba(X)[:,1]
    Y_test_pred = classifier.predict_proba(X_test)[:,1]

    signal_threshold = 83
    cut = np.percentile(Y_test_pred, signal_threshold)
    thresholded_Y_pred = Y_pred > cut
    thresholded_Y_test_pred = Y_test_pred > cut
    
    return [Y_test_pred, thresholded_Y_test_pred]


def write_submission_file(ids_test, Y_test_pred, thresholded_Y_test_pred):
    ids_probs = np.transpose(np.vstack((ids_test, Y_test_pred)))
    ids_probs = np.array(sorted(ids_probs, key = lambda x: -x[1]))
    ids_probs_ranks = np.hstack((
        ids_probs,
        np.arange(1, ids_probs.shape[0]+1).reshape((ids_probs.shape[0], 1))))

    test_ids_map = {}
    for test_id, prob, rank in ids_probs_ranks:
        test_id = int(test_id)
        rank = int(rank)
        test_ids_map[test_id] = rank

    f = open('submission_og.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['EventId', 'RankOrder', 'Class'])
    for i, pred in enumerate(thresholded_Y_test_pred):
        event_id = int(ids_test[i])
        rank = test_ids_map[ids_test[i]]
        klass = pred and 's' or 'b'
        writer.writerow([event_id, rank, klass])
    f.close()