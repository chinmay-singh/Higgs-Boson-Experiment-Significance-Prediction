
def preprocess(X, X_test):
    imputer = Imputer(missing_values = -999.0, strategy = 'most_frequent')
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    X_inv_log_cols = np.log(1 / (1 + X[:, inv_log_cols]))
    X = np.hstack((X, X_inv_log_cols))
    X_test_inv_log_cols = np.log(1 / (1 + X_test[:, inv_log_cols]))
    X_test = np.hstack((X_test, X_test_inv_log_cols))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X, X_test