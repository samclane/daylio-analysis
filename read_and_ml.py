import datetime
import time

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# PARAMETER DEFAULTS
NUM_ESTIMATORS = 50
TEST_RATIO = .25
KERNEL_DEFAULT = 'rbf'
C_VALUE = 7
NUM_NEIGHBORS = 5
NUM_NODES = 50
NUM_HIDDEN_LAYERS = 1
MOOD_STR_TO_INT = {'awful': -1, 'bad': -.5, 'meh': 0, 'good': .5, 'rad': 1}
MOOD_INT_TO_STR = {n: s for s, n in MOOD_STR_TO_INT.items()}


def read_file(filepath: str) -> pd.DataFrame:
    return pd.DataFrame.from_csv(filepath, index_col=None).sort_index(ascending=False).reset_index(drop=True)


def row_datetime_to_timestamp(row: pd.Series) -> int:
    s = '/'.join([str(row["year"]), row["date"], row["weekday"], row["time"]])
    return int(time.mktime(datetime.datetime.strptime(s, "%Y/%B %d/%A/%I:%M %p").timetuple()))


def binarize_activities(df: pd.DataFrame) -> pd.DataFrame:
    actcols = ','.join(list(set([x.strip() for y in df["activities"].values.tolist() for x in y])))
    acts = df["activities"].tolist()
    actdf = pd.read_csv(pd.compat.StringIO(actcols), engine='python')

    actrows = np.array([np.in1d(actdf.columns.values, a) for a in acts])
    actdf = pd.DataFrame(actrows, columns=actdf.columns)

    return actdf


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = df.apply(lambda row: row_datetime_to_timestamp(row), axis=1)
    df["weekend"] = df.apply(lambda row: row["weekday"] in ["Saturday", "Sunday"], axis=1)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # create new column that's purely timestamp
    df = generate_features(df)
    df = df.dropna(subset=["activities"]).reset_index(drop=True)
    df["activities"] = df["activities"].apply(lambda x: [t.strip() for t in x.split('|')])
    df["mood"] = df["mood"].map(MOOD_STR_TO_INT)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features = binarize_activities(df)
    features["weekend"] = df["weekend"]
    features["yesterday"] = df["mood"].shift(-1).fillna(df["mood"].mode()[0])
    features["tomorrow"] = df["mood"].shift(1).fillna(df["mood"].mode()[0])

    # Note Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    features["sentiment"] = df["note"].fillna("").apply(lambda x: sid.polarity_scores(x)["compound"])

    return features


def feature_select(X: pd.DataFrame, y: pd.Series, n_est=NUM_ESTIMATORS) -> pd.DataFrame:
    clf = ExtraTreesRegressor(n_estimators=n_est, random_state=42)
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, prefit=True)
    reduced_features = X[[X.columns[x] for x in model.get_support(True)]]

    return reduced_features


def regress_and_report(X: pd.DataFrame, y: pd.Series, clf, test_ratio=TEST_RATIO) -> object:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def regress_svm(X: pd.DataFrame, y: pd.Series, kernel_val=KERNEL_DEFAULT, C_val=C_VALUE, test_ratio=TEST_RATIO) -> SVR:
    # n_features = len(set(y.values))
    clf = SVR(kernel=kernel_val, C=C_val)
    regress_and_report(X, y, clf, test_ratio)
    return clf


def regress_knn(X: pd.DataFrame, y: pd.Series, K_val=None, test_ratio=TEST_RATIO) -> neighbors.KNeighborsRegressor:
    if K_val is None:
        K_val = NUM_NEIGHBORS
    clf = neighbors.KNeighborsRegressor(K_val,
                                        weights='uniform')  # k = len(set(y)) is a shorthand heuristic I invented
    regress_and_report(X, y, clf, test_ratio)
    return clf


def regress_mlp(X: pd.DataFrame, y: pd.Series, n_nodes=NUM_NODES, n_layers=NUM_HIDDEN_LAYERS, test_ratio=TEST_RATIO) -> MLPRegressor:
    clf = MLPRegressor((n_nodes, n_layers), random_state=42)
    regress_and_report(X, y, clf, test_ratio)
    return clf


def regress_ard(X: pd.DataFrame, y: pd.Series, test_ratio=TEST_RATIO) -> ARDRegression:
    clf = ARDRegression(n_iter=400)
    regress_and_report(X, y, clf, test_ratio)
    return clf


def main() -> int:
    data = read_file("daylio_export.csv")
    data = preprocess(data)
    features = extract_features(data)
    X, y = features, data["mood"]
    X_reduced = feature_select(X, y)

    print("SVM: ")
    regress_svm(X, y)
    print("-----")
    regress_svm(X_reduced, y)
    print("-----")
    print("-----")
    print("KNN: ")
    regress_knn(X, y)
    print("-----")
    regress_knn(X_reduced, y)
    print("-----")
    print("-----")
    print("MLP: ")
    regress_mlp(X, y)
    print("-----")
    regress_mlp(X_reduced, y)

    return 0


if __name__ == "__main__":
    main()
