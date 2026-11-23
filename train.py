"""
train.py — логистикалық регрессияны оқыту (тек оқыту бөлігіндегі студенттерде).

Іске қосу: python train.py   →  model.joblib
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml import FEATURES, topic_features


def build_dataset(answers, questions, topics, true_levels, ids):
    parts = []
    for sid, part in answers[answers["student_id"].isin(ids)].groupby("student_id"):
        f = topic_features(part, questions, topics).reset_index()
        f["student_id"] = sid
        parts.append(f)
    return pd.concat(parts).merge(true_levels, on=["student_id", "topic_id"])


def train_model():
    topics = pd.read_csv("data/topics.csv")
    questions = pd.read_csv("data/questions.csv")
    answers = pd.read_csv("data/students.csv")
    true_levels = pd.read_csv("data/true_levels.csv")
    test_ids = set(pd.read_csv("data/test_students.csv")["student_id"])
    train_ids = [s for s in answers["student_id"].unique() if s not in test_ids]
    data = build_dataset(answers, questions, topics, true_levels, train_ids)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(data[FEATURES], data["true_level"])
    return model, len(train_ids)


if __name__ == "__main__":
    model, n = train_model()
    joblib.dump(model, "model.joblib")
    print(f"Модель {n} студентте оқытылды → model.joblib")
