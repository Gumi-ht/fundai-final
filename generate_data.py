"""
generate_data.py — жасанды (синтетикалық) студенттерді жасау.

Нақты оқушылардың деректері жоқ, сондықтан студенттерді «білім моделі» арқылы модельдейміз:
  1. Әр студенттің әр тақырыпты білуі тақырыптар графына (пререквизиттерге) тәуелді:
     базалық тақырыпты білмесе, күрделі тақырыпты білу ықтималдығы төмен.
  2. Деңгей: 0 — білмейді, 1 — жартылай біледі, 2 — біледі.
  3. Әр сұраққа жауап деңгейге байланысты кездейсоқ: білетін студент кейде қателеседі (slip),
     білмейтін студент кейде табады (guess, 4 нұсқадан 1/4). «Сенімдімін / болжадым» белгісі де модельденеді.

Маңызды: бұл модель knowledge.py-дағы Байес желісінен әдейі басқаша (параметрлері де, деңгей саны да),
сондықтан бағалау «өз жауабын өзі тексеру» болмайды.

Шығатын файлдар:
  data/students.csv      — student_id, q_id, correct, confident, time_sec (барлық test сұрақтарына жауап)
  data/true_levels.csv   — student_id, topic_id, true_level (шын деңгей, модель оны көрмейді)
  data/test_students.csv — бағалауға қалдырылған студенттер (модель оларда оқытылмайды)

Іске қосу: python generate_data.py
"""
import numpy as np
import pandas as pd

N_STUDENTS = 300
N_TEST = 60
SEED = 42

# Генератордың параметрлері (knowledge.py параметрлерінен әдейі өзгеше)
P_ROOT = 0.55            # базалық тақырыпты білу ықтималдығы
P_PARENTS_OK = 0.80      # барлық пререквизиттерді білсе
P_PARENT_MISSING = 0.20  # кем дегенде бір пререквизитті білмесе
# деңгей → (дұрыс жауап ықтималдығы, дұрыс болғанда «сенімдімін», қате болғанда «сенімдімін»)
LEVEL_PARAMS = {0: (0.25, 0.25, 0.50),
                1: (0.60, 0.55, 0.35),
                2: (0.92, 0.85, 0.30)}
LEVEL_TIME = {0: 45, 1: 35, 2: 22}  # орташа жауап уақыты, секунд


def parse_prereqs(topics):
    return {r.topic_id: [p for p in str(r.prereqs).split(";") if p and p != "nan"]
            for r in topics.itertuples()}


def topo_order(prereqs):
    order, seen = [], set()

    def visit(t):
        if t in seen:
            return
        for p in prereqs[t]:
            visit(p)
        seen.add(t)
        order.append(t)

    for t in prereqs:
        visit(t)
    return order


def sample_student(rng, prereqs):
    """Бір студенттің шын деңгейлері: {topic_id: 0/1/2}."""
    knows, levels = {}, {}
    for t in topo_order(prereqs):
        if not prereqs[t]:
            p = P_ROOT
        else:
            p = P_PARENTS_OK if all(knows[q] for q in prereqs[t]) else P_PARENT_MISSING
        knows[t] = rng.random() < p
        if knows[t]:
            levels[t] = 2 if rng.random() < 0.65 else 1
        else:
            levels[t] = 0 if rng.random() < 0.80 else 1
    return levels


def simulate_answer(rng, level):
    """Деңгейі level студенттің бір сұраққа жауабы: (correct, confident, time_sec)."""
    p_correct, conf_if_right, conf_if_wrong = LEVEL_PARAMS[level]
    correct = int(rng.random() < p_correct)
    confident = int(rng.random() < (conf_if_right if correct else conf_if_wrong))
    time_sec = round(max(5.0, rng.normal(LEVEL_TIME[level], 8)), 1)
    return correct, confident, time_sec


def main():
    rng = np.random.default_rng(SEED)
    topics = pd.read_csv("data/topics.csv")
    questions = pd.read_csv("data/questions.csv")
    test_q = questions[questions["use"] == "test"]
    prereqs = parse_prereqs(topics)

    answers, truth = [], []
    for k in range(1, N_STUDENTS + 1):
        sid = f"S{k:03d}"
        levels = sample_student(rng, prereqs)
        for t, lv in levels.items():
            truth.append({"student_id": sid, "topic_id": t, "true_level": lv})
        for q in test_q.itertuples():
            c, conf, ts = simulate_answer(rng, levels[q.topic_id])
            answers.append({"student_id": sid, "q_id": q.q_id, "correct": c, "confident": conf, "time_sec": ts})

    ids = [f"S{k:03d}" for k in range(1, N_STUDENTS + 1)]
    test_ids = sorted(rng.choice(ids, size=N_TEST, replace=False))

    pd.DataFrame(answers).to_csv("data/students.csv", index=False)
    pd.DataFrame(truth).to_csv("data/true_levels.csv", index=False)
    pd.DataFrame({"student_id": test_ids}).to_csv("data/test_students.csv", index=False)
    lv = pd.DataFrame(truth)["true_level"].value_counts(normalize=True).sort_index()
    print(f"{N_STUDENTS} студент, {len(answers)} жауап. Деңгейлер үлесі:", lv.round(2).to_dict())


if __name__ == "__main__":
    main()
