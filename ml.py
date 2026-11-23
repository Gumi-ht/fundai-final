"""
ml.py — классикалық машиналық оқыту: логистикалық регрессия (салыстыру моделі).

Әр (студент, тақырып) жұбы үшін белгілер (features) есептейміз де, деңгейді (0/1/2) болжаймыз.
Бұл — мұғаліммен оқытылатын (supervised) классификация. Ол Байес желісінен айырмашылығы:
тақырыптар арасындағы байланысты білмейді және оқыту үшін көп деректер керек.
"""
import pandas as pd

FEATURES = ["score", "conf_correct", "guess_rate", "conf_wrong", "n_answers"]


def topic_features(answers, questions, topics):
    """answers: q_id, correct, confident бағандары бар кесте. Жауап жоқ тақырыпқа — бейтарап мәндер."""
    a = answers.merge(questions[["q_id", "topic_id"]], on="q_id", how="left", suffixes=("", "_q"))
    if "topic_id_q" in a:
        a["topic_id"] = a["topic_id"].fillna(a["topic_id_q"])
    rows = []
    for t in topics["topic_id"]:
        part = a[a["topic_id"] == t]
        n = len(part)
        if n == 0:
            rows.append({"topic_id": t, "score": 0.5, "conf_correct": 0.25, "guess_rate": 0.5,
                         "conf_wrong": 0.1, "n_answers": 0})
            continue
        rows.append({"topic_id": t,
                     "score": part["correct"].mean(),
                     "conf_correct": ((part["correct"] == 1) & (part["confident"] == 1)).mean(),
                     "guess_rate": (part["confident"] == 0).mean(),
                     "conf_wrong": ((part["correct"] == 0) & (part["confident"] == 1)).mean(),
                     "n_answers": n})
    return pd.DataFrame(rows).set_index("topic_id")


def predict_levels(feats, model):
    return dict(zip(feats.index, model.predict(feats[FEATURES]).astype(int)))
