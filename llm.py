"""
llm.py — қатені түсіндіру: үлкен тіл моделі (LLM) + қауіпсіздік шаралары.

Қауіпсіздік (AI safety) шаралары:
  1. Дұрыс жауапты LLM өзі шешпейді — оны жауап кілтінен (questions.csv) береміз.
     LLM тек «неге» деген сұраққа түсіндірме жазады → галлюцинация қаупі азаяды.
  2. Сайтта дұрыс жауап әрқашан кілттен көрсетіледі, LLM мәтіні тек қосымша түсіндірме.
  3. LLM жауабы бос болса, тым ұзын болса немесе API қолжетімсіз болса — дайын түсіндірме (questions.csv/explanation).
  4. Промпт тек оқу тақырыбымен шектеледі, студенттің жеке деректері жіберілмейді.

Кілттер Streamlit Secrets ішінде сақталады (.streamlit/secrets.toml немесе Streamlit Cloud → Settings → Secrets):
  GEMINI_API_KEY = "..."        # немесе
  ANTHROPIC_API_KEY = "..."
  LLM_MODEL = "..."             # міндетті емес
Кілт жоқ болса, сайт дайын түсіндірмелермен жұмыс істейді.
"""
import requests

TIMEOUT = 20
MAX_CHARS = 1500

PROMPT = """Сен информатика пәнінен ҰБТ-ға дайындайтын мұғалімсің. Тек қазақ тілінде жауап бер.
Оқушы тест сұрағына қате жауап берді. Неге қате екенін және дұрыс жауапқа қалай жетуге болатынын
3–5 қысқа сөйлеммен, қадамдап түсіндір. Дұрыс жауап төменде берілген — оны өзгертпе және басқа жауап ұсынба.

Сұрақ: {question}
{code}Оқушының жауабы: {chosen}
Дұрыс жауап: {correct}"""


def _secret(name, default=None):
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def _gemini(prompt, key, model):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, params={"key": key}, timeout=TIMEOUT,
                      json={"contents": [{"parts": [{"text": prompt}]}]})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _anthropic(prompt, key, model):
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=TIMEOUT,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": model, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def explain(q, chosen_letter):
    """q — questions.csv жолы. Қайтарады: (мәтін, дереккөз)."""
    opts = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"]}
    fallback = str(q.get("explanation", "")).strip()
    code = f"Код:\n{q['code']}\n" if str(q.get("code", "")).strip() not in ("", "nan") else ""
    prompt = PROMPT.format(question=q["question"], code=code,
                           chosen=f"{chosen_letter}) {opts[chosen_letter]}",
                           correct=f"{q['correct']}) {opts[q['correct']]}")
    try:
        if _secret("GEMINI_API_KEY"):
            text = _gemini(prompt, _secret("GEMINI_API_KEY"), _secret("LLM_MODEL", "gemini-2.5-flash"))
            source = "LLM (Gemini)"
        elif _secret("ANTHROPIC_API_KEY"):
            text = _anthropic(prompt, _secret("ANTHROPIC_API_KEY"), _secret("LLM_MODEL", "claude-haiku-4-5"))
            source = "LLM (Claude)"
        else:
            return fallback, "дайын түсіндірме"
        text = text.strip()
        if not text or len(text) > MAX_CHARS:
            return fallback, "дайын түсіндірме (LLM жауабы сүзгіден өтпеді)"
        return text, source
    except Exception:
        return fallback, "дайын түсіндірме (LLM қолжетімсіз)"
