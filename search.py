"""
search.py — жеке оқу бағытын іздеу (A*).

Есеп (іздеу есебі ретінде):
  • Күй          — қазір меңгерілген тақырыптар жиыны;
  • Әрекет       — пререквизиттері меңгерілген келесі тақырыпты оқу;
  • Бастапқы күй — студент білетін (жасыл) тақырыптар;
  • Мақсат       — барлық әлсіз тақырыптар меңгерілген;
  • Құн          — «балл күту уақыты»: Σ (тақырыптың күтілетін балы × сол тақырып меңгерілетін уақыт).
    Яғни көп балл беретін, тез оқылатын тақырыптарды ертерек оқыған тиімді — бірақ пререквизиттер тәртібін бұзбай.
    Қадам құны = оқу уақыты × әлі меңгерілмеген тақырыптардың жалпы салмағы.

Эвристика h(n): пререквизит шектеуін алып тастаймыз да, қалған тақырыптарды
«салмақ/уақыт» қатынасы бойынша сұрыптаймыз (Смит ережесі). Шектеуі жоқ есептің оңтайлы шешімі
нақты шешімнен қымбат бола алмайды → h жол бағасын асырмайды (admissible), сондықтан A* оңтайлы жолды табады.

Салыстыру үшін: BFS (тек қадам санын ескереді), UCS (h = 0), A*.
"""
import heapq
from collections import deque

PARTIAL_FACTOR = 0.6  # жартылай білетін тақырыпты оқуға уақыттың 60%-ы жеткілікті


def make_problem(p_know, levels, topics, prereqs):
    """Әлсіз тақырыптар (мақсат), олардың салмағы (күтілетін балл) және уақыты."""
    targets = [t for t in topics["topic_id"] if levels[t] < 2]
    info = topics.set_index("topic_id")
    weight = {t: float(info.loc[t, "ent_weight"]) * (1 - float(p_know[t])) for t in targets}
    hours = {t: float(info.loc[t, "study_hours"]) * (PARTIAL_FACTOR if levels[t] == 1 else 1.0) for t in targets}
    known = frozenset(t for t in topics["topic_id"] if levels[t] == 2)
    return {"targets": targets, "weight": weight, "hours": hours, "known": known, "prereqs": prereqs}


def successors(prob, done):
    have = prob["known"] | done
    for t in prob["targets"]:
        if t not in done and all(p in have or p not in prob["targets"] for p in prob["prereqs"][t]):
            yield t


def step_cost(prob, done, t):
    remaining = sum(prob["weight"][x] for x in prob["targets"] if x not in done)
    return prob["hours"][t] * remaining


def heuristic(prob, done):
    """Смит ережесі (пререквизиттерсіз): w/h бойынша кему ретімен — жол бағасының төменгі шекарасы."""
    rest = [t for t in prob["targets"] if t not in done]
    rest.sort(key=lambda t: prob["weight"][t] / prob["hours"][t], reverse=True)
    time, cost = 0.0, 0.0
    for t in rest:
        time += prob["hours"][t]
        cost += prob["weight"][t] * time
    return cost


def search(prob, method="astar"):
    """method: 'astar', 'ucs', 'bfs'. Қайтарады: (жол, құн, ашылған түйіндер саны)."""
    start = frozenset()
    goal = frozenset(prob["targets"])
    expanded = 0
    if method == "bfs":
        frontier, seen = deque([(start, [])]), {start}
        while frontier:
            done, path = frontier.popleft()
            expanded += 1
            if done == goal:
                return path, path_cost(prob, path), expanded
            for t in successors(prob, done):
                nxt = done | {t}
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, path + [t]))
        return [], 0.0, expanded
    h = (lambda d: heuristic(prob, d)) if method == "astar" else (lambda d: 0.0)
    counter = 0
    frontier = [(h(start), 0.0, counter, start, [])]
    best_g = {start: 0.0}
    closed = set()
    while frontier:
        f, g, _, done, path = heapq.heappop(frontier)
        if done in closed:
            continue
        closed.add(done)
        expanded += 1
        if done == goal:
            return path, g, expanded
        for t in successors(prob, done):
            nxt = done | {t}
            g2 = g + step_cost(prob, done, t)
            if g2 < best_g.get(nxt, float("inf")) - 1e-12:
                best_g[nxt] = g2
                counter += 1
                heapq.heappush(frontier, (g2 + h(nxt), g2, counter, nxt, path + [t]))
    return [], 0.0, expanded


def path_cost(prob, path):
    done, cost = frozenset(), 0.0
    for t in path:
        cost += step_cost(prob, done, t)
        done = done | {t}
    return cost


def topological_path(prob, depth):
    """Ескі тәсіл (ICT нұсқасы): тек тереңдік бойынша — алдымен базалық тақырыптар."""
    return sorted(prob["targets"], key=lambda t: (depth[t], t))


def depth_of(prereqs):
    depth = {}

    def d(t):
        if t not in depth:
            depth[t] = 0 if not prereqs[t] else 1 + max(d(p) for p in prereqs[t])
        return depth[t]

    for t in prereqs:
        d(t)
    return depth


def build_route(p_know, levels, topics, prereqs):
    """A* бағыты + әр қадамның түсіндірмесі (сайтта көрсету үшін)."""
    prob = make_problem(p_know, levels, topics, prereqs)
    if not prob["targets"]:
        return [], prob, {}
    path, cost, expanded = search(prob, "astar")
    info = topics.set_index("topic_id")
    route, done = [], set()
    for t in path:
        unlocks = [x for x in prob["targets"] if t in prereqs[x] and x not in done]
        if unlocks:
            reason = "Негіз тақырып: " + ", ".join(unlocks) + " тақырыбына жол ашады"
        else:
            reason = (f"Күтілетін балл {prob['weight'][t]:.1f} · {prob['hours'][t]:.1f} сағ "
                      f"→ сағатына {prob['weight'][t] / prob['hours'][t]:.2f} балл")
        route.append({"topic_id": t, "name": info.loc[t, "name"], "level": levels[t],
                      "p": float(p_know[t]), "hours": prob["hours"][t], "weight": prob["weight"][t],
                      "reason": reason, "explanation": info.loc[t, "explanation"],
                      "key_points": info.loc[t, "key_points"], "material": info.loc[t, "material"]})
        done.add(t)
    return route, prob, {"cost": cost, "expanded": expanded}


def compare_methods(prob, prereqs):
    """BFS / UCS / A* / топологиялық тәртіп — құны мен ашылған түйіндер саны."""
    rows = []
    for name, m in [("BFS", "bfs"), ("UCS", "ucs"), ("A*", "astar")]:
        path, cost, expanded = search(prob, m)
        rows.append({"Әдіс": name, "Жол": " → ".join(path), "Құны": round(cost, 1), "Ашылған түйін": expanded})
    topo = topological_path(prob, depth_of(prereqs))
    rows.append({"Әдіс": "Тек тереңдік бойынша (ICT нұсқасы)", "Жол": " → ".join(topo),
                 "Құны": round(path_cost(prob, topo), 1), "Ашылған түйін": "—"})
    return rows
