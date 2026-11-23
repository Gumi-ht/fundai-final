"""
train_rl.py — Q-learning агентін үйрету (нығайтпалы оқыту, RL).

MDP:
  • Күй (s)      — әр тақырыптың P(біледі) мәні 3 топқа бөлінген (төмен / орташа / жоғары) → 3^9 күй;
  • Әрекет (a)   — келесі сұрақтың тақырыбы (9 әрекет);
  • Сыйақы (r)   — жауаптан кейін белгісіздіктің (энтропияның) азаюы, бит;
  • Ауысу        — белгісіз: студенттің жауабына байланысты (симуляция арқылы үйренеміз).
Q-learning: Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',·) − Q(s,a)], ε-greedy зерттеу.

Нақты студенттерде сынақ жүргізу ұзақ, сондықтан агент generate_data.py-дағы
жасанды студенттермен «ойнап» үйренеді (тест бөлігіндегі студенттер қолданылмайды).

Іске қосу: python train_rl.py   →  q_table.npz, results/rl_curve.csv
"""
import numpy as np
import pandas as pd

from adaptive import rl_state, save_q_table
from generate_data import parse_prereqs as gen_prereqs
from generate_data import sample_student, simulate_answer
from knowledge import KnowledgeNet, parse_prereqs

EPISODES = 20000
HORIZON = 15
ALPHA, GAMMA = 0.2, 0.9
EPS_START, EPS_END = 1.0, 0.05
Q_PER_TOPIC = 5


def train(seed=1):
    rng = np.random.default_rng(seed)
    topics = pd.read_csv("data/topics.csv")
    net = KnowledgeNet(topics["topic_id"], parse_prereqs(topics))
    prereqs_gen = gen_prereqs(topics)
    n = len(net.topics)
    Q = {}
    curve, window = [], []
    for ep in range(EPISODES):
        eps = EPS_END + (EPS_START - EPS_END) * max(0.0, 1 - ep / (0.7 * EPISODES))
        levels = sample_student(rng, prereqs_gen)
        lp = net.start()
        left = np.full(n, Q_PER_TOPIC)
        h = net.total_entropy(lp)
        s = rl_state(net.marginals(lp))
        total = 0.0
        for _ in range(HORIZON):
            q_s = Q.setdefault(s, np.zeros(n))
            valid = np.flatnonzero(left > 0)
            if rng.random() < eps:
                a = int(rng.choice(valid))
            else:
                a = int(valid[np.argmax(q_s[valid])])
            t = net.topics[a]
            c, f, _ = simulate_answer(rng, levels[t])
            lp = net.update(lp, t, c, f)
            left[a] -= 1
            h2 = net.total_entropy(lp)
            r = h - h2
            s2 = rl_state(net.marginals(lp))
            q_s2 = Q.setdefault(s2, np.zeros(n))
            q_s[a] += ALPHA * (r + GAMMA * q_s2.max() - q_s[a])
            s, h = s2, h2
            total += r
        window.append(total)
        if (ep + 1) % 500 == 0:
            curve.append({"episode": ep + 1, "avg_return": float(np.mean(window)), "epsilon": round(eps, 3)})
            window = []
    return Q, pd.DataFrame(curve)


if __name__ == "__main__":
    Q, curve = train()
    save_q_table(Q)
    curve.to_csv("results/rl_curve.csv", index=False)
    print(f"Q-кестесі: {len(Q)} күй. Соңғы орташа сыйақы: {curve['avg_return'].iloc[-1]:.3f} бит")
