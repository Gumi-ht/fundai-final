"""
adaptive.py — бейімделгіш тест: келесі сұрақты агент таңдайды.

Агенттің үш стратегиясы (саясаты):
  • random     — кездейсоқ тақырып (салыстыру үшін базалық деңгей);
  • expectimax — бір қадам алға қарау: әр тақырып үшін студенттің мүмкін 4 жауабы (кездейсоқ түйін)
                 бойынша күтілетін ақпарат пайдасын есептеп, ең көп пайда беретінін таңдайды;
  • rl         — Q-learning арқылы үйретілген саясат (train_rl.py), белгісіз күйде — expectimax.

Ақпарат пайдасы = белгісіздіктің (энтропияның) күтілетін азаюы, бит.
Тест тоқтайды: MAX_Q сұрақ қойылса немесе барлық тақырып бойынша жеткілікті сенімді болсақ.
"""
import numpy as np

from knowledge import binary_entropy

MAX_Q = 20        # ең көп сұрақ саны (толық тест — 45)
MIN_Q = 9         # әр тақырыптан кем дегенде бір сұрақ қоюға мүмкіндік
STOP_ENTROPY = 0.6  # әр тақырыптың энтропиясы осыдан төмен болса — тоқтаймыз
POLICY_NAMES = {"rl": "Q-learning агенті", "expectimax": "Ақпарат пайдасы (expectimax)", "random": "Кездейсоқ"}


def save_q_table(Q, path="q_table.npz"):
    """Q-кестесін numpy форматында сақтаймыз (pickle-сыз — кез келген нұсқада ашылады)."""
    states = np.array(list(Q.keys()), dtype=np.int8)
    values = np.array(list(Q.values()), dtype=np.float32)
    np.savez_compressed(path, states=states, values=values)


def load_q_table(path="q_table.npz"):
    data = np.load(path)
    return {tuple(int(x) for x in s): v.astype(float) for s, v in zip(data["states"], data["values"])}


def rl_state(marginals):
    """Q-кестесінің күйі: әр тақырыптың P(біледі) мәні 3 топқа бөлінеді (төмен / орташа / жоғары)."""
    return tuple(int(b) for b in np.digitize(marginals, [0.25, 0.75]))


def info_gain(net, log_post, topic):
    """Expectimax: E[энтропияның азаюы] = H(қазір) − Σ P(жауап)·H(жауаптан кейін)."""
    h_now = net.total_entropy(log_post)
    exp_h = 0.0
    for (c, f), p in net.outcome_probs(log_post, topic).items():
        exp_h += p * net.total_entropy(net.update(log_post, topic, c, f))
    return h_now - exp_h


def choose_topic(net, log_post, available, policy, q_table=None, rng=None):
    """Келесі сұрақтың тақырыбын таңдайды. Қайтарады: (тақырып, {тақырып: баға}, түсіндірме)."""
    rng = rng or np.random.default_rng()
    if policy == "random":
        t = available[rng.integers(len(available))]
        return t, {}, "кездейсоқ таңдалды"
    gains = {t: info_gain(net, log_post, t) for t in available}
    if policy == "rl" and q_table is not None:
        state = rl_state(net.marginals(log_post))
        q = q_table.get(state)
        if q is not None and np.any(q != 0):
            scores = {t: float(q[net.idx[t]]) for t in available}
            t = max(scores, key=scores.get)
            return t, scores, f"Q-кестесі: Q(күй, {t}) = {scores[t]:.3f}"
        t = max(gains, key=gains.get)
        return t, gains, f"күй Q-кестесінде жоқ → ақпарат пайдасы {gains[t]:.3f} бит"
    t = max(gains, key=gains.get)
    return t, gains, f"ең көп ақпарат пайдасы: {gains[t]:.3f} бит"


def should_stop(net, log_post, n_asked, available):
    if n_asked >= MAX_Q or not available:
        return True
    if n_asked < MIN_Q:
        return False
    return bool(np.all(binary_entropy(net.marginals(log_post)) < STOP_ENTROPY))


def available_topics(test_q, asked_qids):
    left = test_q[~test_q["q_id"].isin(asked_qids)]
    return sorted(left["topic_id"].unique())


def pick_question(test_q, topic, asked_qids, rng):
    left = test_q[(test_q["topic_id"] == topic) & (~test_q["q_id"].isin(asked_qids))]
    return left.iloc[rng.integers(len(left))]


def run_adaptive(net, test_q, answer_fn, policy, q_table=None, seed=0, max_q=MAX_Q, early_stop=True):
    """Тестті толық жүргізеді. answer_fn(q_row) → (correct, confident).
    Қайтарады: log_post, жауаптар тізімі, әр қадамдағы P(біледі) тарихы, шешімдер журналы."""
    rng = np.random.default_rng(seed)
    lp = net.start()
    asked, answers, history, log = [], [], [net.marginals(lp)], []
    while True:
        avail = available_topics(test_q, asked)
        if len(asked) >= max_q or not avail or (early_stop and should_stop(net, lp, len(asked), avail)):
            break
        t, scores, why = choose_topic(net, lp, avail, policy, q_table, rng)
        q = pick_question(test_q, t, asked, rng)
        c, f = answer_fn(q)
        lp = net.update(lp, t, c, f)
        asked.append(q["q_id"])
        answers.append({"q_id": q["q_id"], "topic_id": t, "correct": int(c), "confident": int(f)})
        history.append(net.marginals(lp))
        log.append({"step": len(asked), "topic": t, "why": why, "scores": scores})
    return lp, answers, history, log
