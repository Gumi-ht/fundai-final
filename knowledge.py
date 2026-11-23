"""
knowledge.py — студенттің білім моделі (жобаның «миы»).

Лекциялардағы тақырыптар:
  • Байес теоремасы        — әр жауаптан кейін P(біледі) жаңартылады;
  • Байес желісі           — тақырыптар графы: базалық тақырыпты білмесе, күрделісін білу ықтималдығы төмен;
  • Дәл қорытынды (enumeration) — 9 тақырып → 2^9 = 512 күйдің бәрін санап шығамыз;
  • MCMC / Гиббс сэмплингі — сол posterior-ды жуықтап есептеу (дәл әдіспен салыстыру үшін);
  • HMM (BKT)              — оқу кезінде: жасырын күй «біледі/білмейді», ауысу = үйрену, бақылау = жауап.

Бақылау моделі (emission): P(жауап, сенімділік | біледі/білмейді).
«Сенімдімін» деп қате жауап беру — білмеудің күшті белгісі; «болжадым» деп дұрыс жауап беру — әлсіз белгі.
"""
import itertools

import numpy as np

# ---- Байес желісінің шартты ықтималдықтары (CPT) ----
P_ROOT = 0.5             # базалық тақырыпты білу (априорлық)
P_PARENTS_OK = 0.75      # барлық пререквизиттерді білсе
P_PARENT_MISSING = 0.15  # кем дегенде бір пререквизитті білмесе

# ---- Бақылау моделі: P(дұрыс), P(сенімдімін | дұрыс), P(сенімдімін | қате) ----
EMIT = {1: (0.90, 0.80, 0.30),   # біледі
        0: (0.28, 0.30, 0.50)}   # білмейді (4 нұсқа + кейде шамамен табу)

# ---- HMM: оқу кезінде бір жаттығудан кейін үйрену ықтималдығы ----
P_LEARN = 0.15

# P(біледі) → деңгей (0 қызыл, 1 сары, 2 жасыл).
# Шекаралар оқыту бөлігіндегі студенттерде іріктелген (тест студенттері қолданылмаған).
LOW, HIGH = 0.25, 0.65
LEVEL_NAMES = {0: "Білмейді", 1: "Жартылай", 2: "Біледі"}


def emission(know, correct, confident):
    pc, conf_r, conf_w = EMIT[know]
    if correct:
        return pc * (conf_r if confident else 1 - conf_r)
    return (1 - pc) * (conf_w if confident else 1 - conf_w)


def to_level(p):
    return 0 if p < LOW else (1 if p < HIGH else 2)


def binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class KnowledgeNet:
    """Тақырыптар бойынша Байес желісі. Posterior лог-масштабта сақталады (жылдам әрі тұрақты)."""

    def __init__(self, topic_ids, prereqs):
        self.topics = list(topic_ids)
        self.idx = {t: i for i, t in enumerate(self.topics)}
        self.parents = {t: list(prereqs.get(t, [])) for t in self.topics}
        n = len(self.topics)
        # барлық 2^n күй: әр жол — қай тақырыпты біледі (1) / білмейді (0)
        self.states = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
        self.log_prior = np.zeros(len(self.states))
        for t in self.topics:
            x = self.states[:, self.idx[t]]
            if self.parents[t]:
                ok = np.all(self.states[:, [self.idx[p] for p in self.parents[t]]] == 1, axis=1)
                p1 = np.where(ok, P_PARENTS_OK, P_PARENT_MISSING)
            else:
                p1 = np.full(len(self.states), P_ROOT)
            self.log_prior += np.log(np.where(x == 1, p1, 1 - p1))
        # әр (тақырып, дұрыс, сенімді) үшін әр күйдің лог-ықтималдығы — алдын ала есептейміз
        self.log_lik = {}
        for t in self.topics:
            x = self.states[:, self.idx[t]]
            for c in (0, 1):
                for f in (0, 1):
                    self.log_lik[(t, c, f)] = np.log(np.where(x == 1, emission(1, c, f), emission(0, c, f)))

    # ----- дәл қорытынды (enumeration) -----
    def start(self):
        return self.log_prior.copy()

    def update(self, log_post, topic, correct, confident):
        """Байес теоремасы: posterior ∝ prior × likelihood (лог-масштабта — қосу)."""
        return log_post + self.log_lik[(topic, int(correct), int(confident))]

    def from_answers(self, answers):
        """answers: [(topic, correct, confident), ...]"""
        lp = self.start()
        for t, c, f in answers:
            lp = self.update(lp, t, c, f)
        return lp

    @staticmethod
    def normalize(log_post):
        w = np.exp(log_post - log_post.max())
        return w / w.sum()

    def marginals(self, log_post):
        """P(тақырыпты біледі | барлық жауаптар) — әр тақырып үшін."""
        return self.normalize(log_post) @ self.states

    def marginals_dict(self, log_post):
        return dict(zip(self.topics, self.marginals(log_post)))

    def total_entropy(self, log_post):
        return float(binary_entropy(self.marginals(log_post)).sum())

    def outcome_probs(self, log_post, topic):
        """Келесі жауаптың 4 нұсқасының ықтималдығы (expectimax-тағы кездейсоқ түйін)."""
        w = self.normalize(log_post)
        return {(c, f): float(w @ np.exp(self.log_lik[(topic, c, f)])) for c in (0, 1) for f in (0, 1)}

    # ----- MCMC: Гиббс сэмплингі -----
    def _log_joint(self, x, answers):
        s = 0.0
        for t in self.topics:
            i = self.idx[t]
            if self.parents[t]:
                p1 = P_PARENTS_OK if all(x[self.idx[p]] for p in self.parents[t]) else P_PARENT_MISSING
            else:
                p1 = P_ROOT
            s += np.log(p1 if x[i] else 1 - p1)
        for t, c, f in answers:
            s += np.log(emission(int(x[self.idx[t]]), c, f))
        return s

    def gibbs(self, answers, n_samples=2000, burn_in=200, seed=0):
        """Әр қадамда бір айнымалыны қалғандарына шартталған ықтималдықпен қайта таңдаймыз.
        Ұзақ жүрген соң үлгілер нақты posterior-дан алынғандай болады."""
        rng = np.random.default_rng(seed)
        n = len(self.topics)
        x = rng.integers(0, 2, n)
        counts = np.zeros(n)
        for step in range(burn_in + n_samples):
            for i in range(n):
                x[i] = 1
                l1 = self._log_joint(x, answers)
                x[i] = 0
                l0 = self._log_joint(x, answers)
                p1 = 1 / (1 + np.exp(l0 - l1))
                x[i] = int(rng.random() < p1)
            if step >= burn_in:
                counts += x
        return dict(zip(self.topics, counts / n_samples))


# ----- HMM / Bayesian Knowledge Tracing: оқу кезеңі -----
def bkt_step(p_know, correct, confident, p_learn=P_LEARN):
    """1) Ауысу: жаттығу кезінде студент тақырыпты үйренуі мүмкін.
       2) Бақылау: жауапқа қарап Байес бойынша жаңартамыз."""
    p = p_know + (1 - p_know) * p_learn
    num = p * emission(1, correct, confident)
    den = num + (1 - p) * emission(0, correct, confident)
    return num / den


def parse_prereqs(topics):
    return {r.topic_id: [p for p in str(r.prereqs).split(";") if p and p != "nan"]
            for r in topics.itertuples()}
