"""
evaluate.py — қорғауға арналған тәжірибелер. Барлығы тест бөлігіндегі 60 студентте
(олар логистикалық регрессияны да, Q-learning-ті де оқытуға қатыспаған).

  1. Деңгейді анықтау дәлдігі: логистикалық регрессия vs Байес желісі; толық тест vs бейімделгіш тест.
  2. Сұрақ саны → дәлдік қисығы: кездейсоқ / expectimax / Q-learning агенттері.
  3. Оқу бағыты: BFS / UCS / A* — ашылған түйіндер және бағыт құны.
  4. Жасыту әдісі (SA) vs толық перебор: SA оңтайлы шешімді қаншалықты жиі табады.
  5. Гиббс сэмплингі (MCMC) vs дәл есептеу: үлгі саны өскенде қателік азаяды.

Іске қосу: python evaluate.py   →  results/metrics.md және results/*.png
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive import MAX_Q, load_q_table, run_adaptive
from knowledge import KnowledgeNet, parse_prereqs, to_level
from ml import predict_levels, topic_features
from planner import anneal, brute_force
from search import make_problem, path_cost, search, topological_path, depth_of

topics = pd.read_csv("data/topics.csv")
questions = pd.read_csv("data/questions.csv")
answers = pd.read_csv("data/students.csv")
truth = pd.read_csv("data/true_levels.csv")
test_ids = list(pd.read_csv("data/test_students.csv")["student_id"])
prereqs = parse_prereqs(topics)
net = KnowledgeNet(topics["topic_id"], prereqs)
model = joblib.load("model.joblib")
q_table = load_q_table()
test_q = questions[questions["use"] == "test"].reset_index(drop=True)
q_topic = dict(zip(questions["q_id"], questions["topic_id"]))
true_lv = {(r.student_id, r.topic_id): r.true_level for r in truth.itertuples()}
lookup = {(r.student_id, r.q_id): (r.correct, r.confident) for r in answers.itertuples()}
T = list(topics["topic_id"])
out = []


def acc(pred, sid):
    return np.mean([pred[t] == true_lv[(sid, t)] for t in T])


def weak_recall(pred, sid):
    weak = [t for t in T if true_lv[(sid, t)] == 0]
    return np.mean([pred[t] == 0 for t in weak]) if weak else np.nan


def bn_levels(lp):
    return {t: to_level(p) for t, p in net.marginals_dict(lp).items()}


# ---------- 1. Дәлдік ----------
rows = []
res = {"LogReg, толық тест (45 сұрақ)": [], "Байес желісі, толық тест (45 сұрақ)": []}
for sid in test_ids:
    part = answers[answers["student_id"] == sid]
    lr = predict_levels(topic_features(part, questions, topics), model)
    lp = net.from_answers([(q_topic[r.q_id], r.correct, r.confident) for r in part.itertuples()])
    res["LogReg, толық тест (45 сұрақ)"].append((acc(lr, sid), weak_recall(lr, sid), 45))
    res["Байес желісі, толық тест (45 сұрақ)"].append((acc(bn_levels(lp), sid), weak_recall(bn_levels(lp), sid), 45))
for pol, name in [("random", "кездейсоқ агент"), ("expectimax", "expectimax агенті"), ("rl", "Q-learning агенті")]:
    key = f"Байес желісі, бейімделгіш тест ({name})"
    res[key] = []
    for k, sid in enumerate(test_ids):
        lp, ans, _, _ = run_adaptive(net, test_q, lambda q, s=sid: lookup[(s, q["q_id"])], pol, q_table, seed=k)
        pred = bn_levels(lp)
        res[key].append((acc(pred, sid), weak_recall(pred, sid), len(ans)))
for k, v in res.items():
    a = np.array(v, dtype=float)
    rows.append({"Модель": k, "Деңгей дәлдігі": f"{a[:, 0].mean():.1%}",
                 "Әлсіз тақырыпты табу (recall)": f"{np.nanmean(a[:, 1]):.1%}",
                 "Орташа сұрақ саны": f"{a[:, 2].mean():.1f}"})
t1 = pd.DataFrame(rows)
out += ["## 1. Деңгейді анықтау дәлдігі (60 тест студенті × 9 тақырып)", "", t1.to_markdown(index=False), ""]

# ---------- 2. Сұрақ саны → дәлдік ----------
curves = {}
for pol, name in [("random", "Кездейсоқ"), ("expectimax", "Expectimax"), ("rl", "Q-learning")]:
    accs = np.zeros(MAX_Q)
    for k, sid in enumerate(test_ids):
        _, _, hist, _ = run_adaptive(net, test_q, lambda q, s=sid: lookup[(s, q["q_id"])], pol, q_table,
                                     seed=k, early_stop=False)
        for step in range(1, MAX_Q + 1):
            m = hist[min(step, len(hist) - 1)]
            accs[step - 1] += np.mean([to_level(m[i]) == true_lv[(sid, t)] for i, t in enumerate(T)])
    curves[name] = accs / len(test_ids)
plt.figure(figsize=(7, 4.2))
for name, c in curves.items():
    plt.plot(range(1, MAX_Q + 1), c, marker="o", ms=3, label=name)
full_lr = np.mean([v[0] for v in res["LogReg, толық тест (45 сұрақ)"]])
plt.axhline(full_lr, ls="--", c="gray", label="LogReg, 45 сұрақ")
plt.xlabel("Қойылған сұрақ саны")
plt.ylabel("Деңгей дәлдігі")
plt.title("Бейімделгіш тест: қай агент тезірек анықтайды?")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=130)
plt.close()
out += ["## 2. Сұрақ саны → дәлдік", "", "![](accuracy_curve.png)", "",
        pd.DataFrame({k: [f"{v[4]:.1%}", f"{v[9]:.1%}", f"{v[14]:.1%}", f"{v[19]:.1%}"] for k, v in curves.items()},
                     index=["5 сұрақ", "10 сұрақ", "15 сұрақ", "20 сұрақ"]).to_markdown(), ""]

# ---------- RL оқу қисығы ----------
curve = pd.read_csv("results/rl_curve.csv")
plt.figure(figsize=(7, 3.8))
plt.plot(curve["episode"], curve["avg_return"])
plt.xlabel("Эпизод (жасанды студентпен тест)")
plt.ylabel("Орташа сыйақы, бит")
plt.title("Q-learning агентінің үйренуі")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("results/rl_learning_curve.png", dpi=130)
plt.close()

# ---------- 3. Іздеу ----------
depth = depth_of(prereqs)
srows = []
for sid in test_ids:
    part = answers[answers["student_id"] == sid]
    lp = net.from_answers([(q_topic[r.q_id], r.correct, r.confident) for r in part.itertuples()])
    p = net.marginals_dict(lp)
    prob = make_problem(p, {t: to_level(v) for t, v in p.items()}, topics, prereqs)
    if len(prob["targets"]) < 2:
        continue
    r = {"targets": len(prob["targets"])}
    for name, m in [("BFS", "bfs"), ("UCS", "ucs"), ("A*", "astar")]:
        _, cost, exp = search(prob, m)
        r[f"{name} түйін"], r[f"{name} құн"] = exp, cost
    r["Тереңдік құн"] = path_cost(prob, topological_path(prob, depth))
    srows.append(r)
s = pd.DataFrame(srows)
better = (s["Тереңдік құн"] - s["A* құн"] > 1e-6).mean()
saving = ((s["Тереңдік құн"] - s["A* құн"]) / s["Тереңдік құн"]).mean()
t3 = pd.DataFrame([
    {"Әдіс": "BFS", "Орташа ашылған түйін": s["BFS түйін"].mean(), "Орташа құн": s["BFS құн"].mean()},
    {"Әдіс": "UCS", "Орташа ашылған түйін": s["UCS түйін"].mean(), "Орташа құн": s["UCS құн"].mean()},
    {"Әдіс": "A*", "Орташа ашылған түйін": s["A* түйін"].mean(), "Орташа құн": s["A* құн"].mean()},
    {"Әдіс": "Тек тереңдік бойынша (ICT нұсқасы)", "Орташа ашылған түйін": np.nan, "Орташа құн": s["Тереңдік құн"].mean()},
]).round(1)
plt.figure(figsize=(6, 3.6))
plt.bar(["BFS", "UCS", "A*"], [s["BFS түйін"].mean(), s["UCS түйін"].mean(), s["A* түйін"].mean()],
        color=["#94a3b8", "#60a5fa", "#22d3ee"])
plt.ylabel("Орташа ашылған түйін")
plt.title("Оқу бағытын іздеу")
plt.tight_layout()
plt.savefig("results/search_nodes.png", dpi=130)
plt.close()
out += ["## 3. Оқу бағыты: BFS / UCS / A*", "", t3.to_markdown(index=False).replace("nan", " —"), "",
        f"A* бағыты тереңдік бойынша сұрыптаудан {better:.0%} жағдайда арзанырақ, "
        f"орташа үнем {saving:.1%}. UCS пен A* құны бірдей (екеуі де оңтайлы), бірақ A* түйінді аз ашады.", "",
        "![](search_nodes.png)", ""]

# ---------- 4. SA vs толық перебор ----------
hits, gaps = 0, []
budgets = [8, 15, 25]
n_cases = 0
for k, sid in enumerate(test_ids):
    part = answers[answers["student_id"] == sid]
    lp = net.from_answers([(q_topic[r.q_id], r.correct, r.confident) for r in part.itertuples()])
    p = net.marginals_dict(lp)
    prob = make_problem(p, {t: to_level(v) for t, v in p.items()}, topics, prereqs)
    if not prob["targets"]:
        continue
    for b in budgets:
        _, v_sa, _ = anneal(prob, b, seed=k)
        _, v_bf = brute_force(prob, b)
        n_cases += 1
        hits += abs(v_sa - v_bf) < 1e-6
        gaps.append(0 if v_bf == 0 else (v_bf - v_sa) / v_bf)
out += ["## 4. Жасыту әдісі (simulated annealing) vs толық перебор", "",
        f"{n_cases} жағдай (студент × уақыт шегі {budgets} сағ): SA оңтайлы шешімді {hits / n_cases:.0%} жағдайда тапты, "
        f"орташа айырма {np.mean(gaps):.2%}.", ""]

# ---------- 5. Гиббс vs дәл ----------
sizes = [100, 300, 1000, 3000]
errs = {n: [] for n in sizes}
for k, sid in enumerate(test_ids[:5]):
    part = answers[answers["student_id"] == sid].head(15)
    ev = [(q_topic[r.q_id], r.correct, r.confident) for r in part.itertuples()]
    exact = net.marginals_dict(net.from_answers(ev))
    for n in sizes:
        g = net.gibbs(ev, n_samples=n, burn_in=100, seed=k)
        errs[n].append(np.mean([abs(g[t] - exact[t]) for t in T]))
plt.figure(figsize=(6, 3.6))
plt.plot(sizes, [np.mean(errs[n]) for n in sizes], marker="o")
plt.xscale("log")
plt.xlabel("Гиббс үлгілерінің саны")
plt.ylabel("Орташа абсолют қателік")
plt.title("MCMC дәл posterior-ға жақындайды")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("results/gibbs_error.png", dpi=130)
plt.close()
out += ["## 5. Гиббс сэмплингі (MCMC) vs дәл есептеу (enumeration)", "",
        pd.DataFrame({"Үлгі саны": sizes, "Орташа қателік": [f"{np.mean(errs[n]):.3f}" for n in sizes]}).to_markdown(index=False),
        "", "![](gibbs_error.png)", ""]

with open("results/metrics.md", "w", encoding="utf-8") as f:
    f.write("# Нәтижелер\n\n" + "\n".join(out))
print("\n".join(out))
