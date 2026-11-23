"""
planner.py — ҰБТ-ға дейінгі жоспар: жергілікті іздеу (simulated annealing — «жасыту әдісі»).

Есеп: ҰБТ-ға дейін H сағат бар. Қай әлсіз тақырыптарды оқыған жөн, күтілетін балл максимум болуы үшін?
  • Күй (толық конфигурация) — таңдалған тақырыптар жиыны;
  • Көрші күй — бір тақырыпты қосу немесе алып тастау (пререквизиттерімен бірге);
  • Мақсат функциясы — Σ күтілетін балл, уақыт шегінен асса — айыппұл.
Жасыту: басында нашар қадамдарды да жиі қабылдаймыз (жергілікті максимумнан шығу үшін),
температура төмендеген сайын тек жақсартатын қадамдар қалады.

Тексеру үшін brute_force() барлық 2^k нұсқаны қарап шығады — SA табған шешіммен салыстырамыз.
"""
import itertools
import math

import numpy as np

PENALTY = 3.0


def closure_add(t, prob):
    """t тақырыбын қосқанда оның әлсіз пререквизиттерін де қосамыз."""
    out, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack += [p for p in prob["prereqs"][x] if p in prob["targets"]]
    return out


def closure_remove(t, prob):
    """t тақырыбын алып тастағанда соған тәуелді тақырыптарды да алып тастаймыз."""
    out, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack += [y for y in prob["targets"] if x in prob["prereqs"][y]]
    return out


def value(sel, prob, budget):
    gain = sum(prob["weight"][t] for t in sel)
    used = sum(prob["hours"][t] for t in sel)
    return gain - PENALTY * max(0.0, used - budget)


def anneal(prob, budget, iters=3000, t0=3.0, cooling=0.998, seed=0):
    rng = np.random.default_rng(seed)
    targets = prob["targets"]
    cur, cur_v = set(), 0.0
    best, best_v = set(), 0.0
    temp = t0
    history = []
    for k in range(iters):
        t = targets[rng.integers(len(targets))]
        nxt = cur - closure_remove(t, prob) if t in cur else cur | closure_add(t, prob)
        v = value(nxt, prob, budget)
        if v >= cur_v or rng.random() < math.exp((v - cur_v) / max(temp, 1e-9)):
            cur, cur_v = nxt, v
            if cur_v > best_v and sum(prob["hours"][x] for x in cur) <= budget:
                best, best_v = set(cur), cur_v
        temp *= cooling
        if k % 10 == 0:
            history.append({"қадам": k, "ағымдағы": round(cur_v, 2), "ең жақсы": round(best_v, 2),
                            "температура": round(temp, 3)})
    return best, best_v, history


def brute_force(prob, budget):
    targets, best, best_v = prob["targets"], set(), 0.0
    for mask in itertools.product([0, 1], repeat=len(targets)):
        sel = {t for t, m in zip(targets, mask) if m}
        if any(p in targets and p not in sel for t in sel for p in prob["prereqs"][t]):
            continue
        if sum(prob["hours"][t] for t in sel) > budget:
            continue
        v = sum(prob["weight"][t] for t in sel)
        if v > best_v:
            best, best_v = sel, v
    return best, best_v


def schedule(order, prob, hours_per_day):
    """Таңдалған тақырыптарды күндерге бөлеміз (A* тәртібімен)."""
    days, day, left = [], 1, hours_per_day
    for t in order:
        need = prob["hours"][t]
        while need > 1e-9:
            take = min(need, left)
            days.append({"Күн": day, "Тақырып": t, "Сағат": round(take, 1)})
            need -= take
            left -= take
            if left <= 1e-9:
                day, left = day + 1, hours_per_day
    return days
