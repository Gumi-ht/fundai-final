"""
app.py — «ҰБТ Навигатор»: информатикадан ҰБТ-ға дайындайтын бейімделгіш ЖИ-тренер (Streamlit).

Беттер (жоғарғы мәзір):
  1. Тест             — бейімделгіш тест: келесі сұрақты агент таңдайды (expectimax / Q-learning / кездейсоқ);
  2. Менің картам     — тақырыптар графы, түсі — P(біледі) бойынша деңгей (Байес желісі);
  3. Оқу бағытым      — A* тапқан оңтайлы оқу реті + жаттығулар (HMM/BKT) + ЖИ түсіндірмесі (LLM);
  4. ҰБТ жоспары      — уақыт шегіне қарай тақырыптарды таңдау (simulated annealing);
  5. Ішінде не болды  — барлық алгоритмдердің жұмысы: ықтималдықтар, агент шешімдері, A*, MCMC, графиктер;
  6. Мұғалім          — топ нәтижесі (тест бөлігіндегі 60 жасанды студент);
  7. Демо             — қорғауға арналған 3 дайын мысал.

Іске қосу: streamlit run app.py
"""
import html as h
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from adaptive import (MAX_Q, POLICY_NAMES, available_topics, choose_topic, load_q_table, pick_question, run_adaptive,
                      should_stop)
from knowledge import LEVEL_NAMES, KnowledgeNet, bkt_step, parse_prereqs, to_level
from llm import explain
from ml import predict_levels, topic_features
from planner import anneal, brute_force, schedule
from search import build_route, compare_methods, depth_of, search

st.set_page_config(page_title="ҰБТ Навигатор", layout="wide", initial_sidebar_state="collapsed")

# Деңгей түстері: (негізгі түс, мөлдір фон)
LV = {0: ("#f87171", "rgba(248,113,113,.13)"),
      1: ("#fbbf24", "rgba(251,191,36,.13)"),
      2: ("#34d399", "rgba(52,211,153,.13)")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700;800&display=swap');
:root{color-scheme:dark;--ink:#eef1f8;--muted:#8b93a7;--line:rgba(255,255,255,.09);--glass:rgba(255,255,255,.045);
      --bg:#070b14;--c1:#22d3ee;--c2:#8b5cf6;--c3:#f472b6;}
html,body,.stApp,[class*="css"],button,input,label,p,div,span{font-family:'Golos Text',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none;}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:2.2rem;max-width:1140px;position:relative;z-index:1;}
h1,h2,h3,h4{letter-spacing:-.02em;color:var(--ink);}

/* ---------- жанды фон: жылжитын жарық дақтар + тор ---------- */
.stApp::before{content:"";position:fixed;inset:-25%;z-index:0;pointer-events:none;
  background:radial-gradient(38% 32% at 20% 25%,rgba(34,211,238,.20),transparent 70%),
             radial-gradient(34% 30% at 80% 20%,rgba(139,92,246,.22),transparent 70%),
             radial-gradient(30% 28% at 60% 85%,rgba(244,114,182,.14),transparent 70%);
  animation:aurora 22s ease-in-out infinite alternate;}
.stApp::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
  background-size:46px 46px;mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%);}

/* ---------- анимациялар ---------- */
@keyframes aurora{0%{transform:translate(0,0) rotate(0)}50%{transform:translate(4%,-3%) rotate(8deg)}100%{transform:translate(-4%,3%) rotate(-6deg)}}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes enter{from{opacity:0;transform:translateY(22px) scale(.985);filter:blur(6px)}to{opacity:1;transform:none;filter:none}}
@keyframes word{from{opacity:0;transform:translateY(.6em) rotate(4deg);filter:blur(8px)}to{opacity:1;transform:none;filter:none}}
@keyframes cardIn{from{opacity:0;transform:perspective(900px) translateX(60px) rotateY(-14deg);filter:blur(8px)}to{opacity:1;transform:none;filter:none}}
@keyframes slideIn{from{opacity:0;transform:translateX(-26px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:translate(-50%,-50%) scale(.6)}to{opacity:1;transform:translate(-50%,-50%) scale(1)}}
@keyframes popIn{0%{opacity:0;transform:scale(.4) rotate(-90deg)}70%{transform:scale(1.12) rotate(8deg)}100%{opacity:1;transform:none}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes flow{to{stroke-dashoffset:-1}}
@keyframes grow{from{width:0}}
@keyframes growY{from{transform:scaleY(0)}}
@keyframes ring{from{stroke-dashoffset:100}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes pulse{0%{opacity:.9;transform:scale(1)}100%{opacity:0;transform:scale(1.35)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(34,211,238,.55)}50%{box-shadow:0 0 18px 3px rgba(34,211,238,.55)}}
@keyframes shine{0%{transform:translateX(-120%) skewX(-20deg)}60%,100%{transform:translateX(220%) skewX(-20deg)}}
@keyframes shimmer{to{background-position:200% center}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes up{0%{transform:translateY(0);opacity:0}15%{opacity:.8}100%{transform:translateY(-180px);opacity:0}}
@keyframes wave{from{opacity:0;transform:scale(.3)}to{opacity:1;transform:none}}

/* әр бет ауысқанда элементтер кезекпен шығады */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
.block-container [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]{animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3){animation-delay:.05s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(4){animation-delay:.12s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(5){animation-delay:.19s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(6){animation-delay:.26s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(7){animation-delay:.33s}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(n+8){animation-delay:.4s}

/* ---------- жоғарғы бренд пен мәзір ---------- */
.topbar{display:flex;align-items:center;gap:14px;margin-bottom:4px;}
.logo{width:42px;height:42px;border-radius:13px;position:relative;overflow:hidden;flex:none;
  background:conic-gradient(from 0deg,var(--c1),var(--c2),var(--c3),var(--c1));animation:spin 8s linear infinite;}
.logo::after{content:"";position:absolute;inset:3px;border-radius:10px;background:#0b1120;}
.logo-wrap{position:relative;width:42px;height:42px;flex:none;}
.logo-wrap svg{position:absolute;inset:0;margin:auto;width:22px;height:22px;z-index:2;}
.logo-wrap svg circle{fill:var(--c1);} .logo-wrap svg line{stroke:#cbd5e1;stroke-width:1.6;}
.brand{font-size:21px;font-weight:800;letter-spacing:-.03em;
  background:linear-gradient(90deg,#fff,#a5f3fc,#c4b5fd,#fff);background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:shimmer 6s linear infinite;}
.brand-sub{font-size:12.5px;color:var(--muted);margin-top:1px;}

.st-key-nav div[role="radiogroup"],.st-key-demo div[role="radiogroup"]{
  display:flex;flex-wrap:nowrap;gap:4px;padding:5px;border-radius:999px;width:fit-content;max-width:100%;overflow-x:auto;
  background:rgba(255,255,255,.05);border:1px solid var(--line);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);scrollbar-width:none;}
.st-key-nav div[role="radiogroup"]::-webkit-scrollbar{display:none;}
.st-key-nav label[data-baseweb="radio"],.st-key-demo label[data-baseweb="radio"]{
  margin:0 !important;padding:9px 18px;border-radius:999px;white-space:nowrap;cursor:pointer;
  transition:background .35s ease,box-shadow .35s ease,transform .2s ease;}
.st-key-nav label[data-baseweb="radio"] > div:first-child,.st-key-demo label[data-baseweb="radio"] > div:first-child{display:none;}
.st-key-nav label[data-baseweb="radio"] *,.st-key-demo label[data-baseweb="radio"] *{color:var(--muted) !important;font-weight:600;transition:color .3s;}
.st-key-nav label[data-baseweb="radio"]:hover,.st-key-demo label[data-baseweb="radio"]:hover{background:rgba(255,255,255,.06);}
.st-key-nav label[data-baseweb="radio"]:has(input:checked),.st-key-demo label[data-baseweb="radio"]:has(input:checked){
  background:linear-gradient(135deg,var(--c1),var(--c2));box-shadow:0 6px 24px -6px rgba(139,92,246,.7);}
.st-key-nav label[data-baseweb="radio"]:has(input:checked) *,.st-key-demo label[data-baseweb="radio"]:has(input:checked) *{color:#fff !important;}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:34px 38px;margin:6px 0 8px;isolation:isolate;
  background:linear-gradient(135deg,rgba(15,23,42,.85),rgba(30,27,75,.75));border:1px solid rgba(255,255,255,.1);}
.hero::before{content:"";position:absolute;inset:-60%;z-index:-2;
  background:conic-gradient(from 0deg,transparent 0 70%,rgba(34,211,238,.35),rgba(139,92,246,.35),transparent);animation:spin 9s linear infinite;}
.hero::after{content:"";position:absolute;inset:1px;border-radius:25px;z-index:-1;background:linear-gradient(135deg,#0b1222,#151236);}
.hero .eyebrow{display:inline-flex;align-items:center;gap:8px;color:#a5f3fc;font-size:12px;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:12px;padding:6px 12px;border-radius:999px;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.25);
  animation:rise .5s ease backwards;}
.hero .eyebrow i{width:7px;height:7px;border-radius:50%;background:var(--c1);animation:glow 2s infinite;}
.hero h1{margin:0 0 10px;font-size:2.35rem;line-height:1.12;font-weight:800;color:#fff;}
.hero h1 span{display:inline-block;animation:word .7s cubic-bezier(.2,.8,.2,1) backwards;margin-right:.24em;}
.hero h1 span.g{background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3));-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:#b6bfd3;margin:0;max-width:680px;line-height:1.6;font-size:15.5px;animation:rise .7s ease .35s backwards;}
.orb{position:absolute;border-radius:50%;z-index:-1;opacity:.55;animation:float 7s ease-in-out infinite;}
.spark{position:absolute;bottom:-10px;width:4px;height:4px;border-radius:50%;background:#a5f3fc;z-index:-1;animation:up 6s linear infinite;}

/* ---------- шыны карточкалар ---------- */
.card{position:relative;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:22px 24px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);animation:rise .5s ease backwards;}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 10px;}
.how .card{animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;transition:transform .3s,border-color .3s;}
.how .card:hover{transform:translateY(-4px);border-color:rgba(34,211,238,.45);}
.how .n{font-size:13px;font-weight:800;letter-spacing:.1em;color:var(--c1);}
.how h4{margin:6px 0 4px;font-size:17px;} .how p{margin:0;color:var(--muted);font-size:14px;line-height:1.5;}

/* сұрақ */
.qhead{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:4px 0 10px;}
.qnum{font-size:46px;font-weight:800;line-height:1;letter-spacing:-.04em;
  background:linear-gradient(135deg,#fff,var(--c1));-webkit-background-clip:text;background-clip:text;color:transparent;animation:popIn .6s ease backwards;}
.qnum small{font-size:18px;color:var(--muted);-webkit-text-fill-color:var(--muted);font-weight:600;}
.segs{display:grid;grid-template-columns:repeat(32,1fr);gap:4px;flex:1;max-width:640px;}
.seg{height:8px;border-radius:4px;background:rgba(255,255,255,.08);}
.seg.s-ok{background:linear-gradient(90deg,var(--c1),var(--c2));}
.seg.s-new{animation:wave .5s cubic-bezier(.2,.8,.2,1.4) backwards;}
.seg.s-cur{background:rgba(34,211,238,.35);animation:glow 1.6s ease-in-out infinite;}
.qcard{position:relative;overflow:hidden;border-radius:24px;padding:28px 30px;border:1px solid rgba(255,255,255,.1);
  background:linear-gradient(135deg,rgba(34,211,238,.08),rgba(139,92,246,.08));animation:cardIn .7s cubic-bezier(.2,.8,.2,1) backwards;}
.qcard .wm{position:absolute;right:-6px;bottom:-34px;font-size:150px;font-weight:800;color:rgba(255,255,255,.04);line-height:1;pointer-events:none;}
.chip{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#c4b5fd;
  background:rgba(139,92,246,.14);border:1px solid rgba(139,92,246,.3);padding:5px 11px;border-radius:999px;}
.q-text{font-size:23px;font-weight:700;color:#fff;line-height:1.38;margin-top:14px;position:relative;}

/* жауап нұсқалары — карточкалар */
[class*="st-key-ans_"] div[role="radiogroup"],[class*="st-key-r_"] div[role="radiogroup"],[class*="st-key-demo_"] div[role="radiogroup"]{gap:8px;width:100%;}
[class*="st-key-ans_"] label[data-baseweb="radio"],[class*="st-key-r_"] label[data-baseweb="radio"],[class*="st-key-demo_"] label[data-baseweb="radio"]{
  width:100%;box-sizing:border-box;margin:0 !important;padding:14px 18px;border-radius:16px;background:var(--glass);border:1px solid var(--line);cursor:pointer;
  transition:transform .25s ease,border-color .25s,background .25s,box-shadow .25s;animation:slideIn .45s cubic-bezier(.2,.8,.2,1) backwards;}
[class*="st-key-ans_"] label[data-baseweb="radio"]:nth-child(1){animation-delay:.15s}
[class*="st-key-ans_"] label[data-baseweb="radio"]:nth-child(2){animation-delay:.23s}
[class*="st-key-ans_"] label[data-baseweb="radio"]:nth-child(3){animation-delay:.31s}
[class*="st-key-ans_"] label[data-baseweb="radio"]:nth-child(4){animation-delay:.39s}
[class*="st-key-ans_"] label[data-baseweb="radio"]:hover,[class*="st-key-r_"] label[data-baseweb="radio"]:hover,[class*="st-key-demo_"] label[data-baseweb="radio"]:hover{
  transform:translateX(6px);border-color:rgba(34,211,238,.5);background:rgba(34,211,238,.06);}
[class*="st-key-ans_"] label[data-baseweb="radio"]:has(input:checked),[class*="st-key-r_"] label[data-baseweb="radio"]:has(input:checked),[class*="st-key-demo_"] label[data-baseweb="radio"]:has(input:checked){
  border-color:var(--c1);background:linear-gradient(135deg,rgba(34,211,238,.16),rgba(139,92,246,.14));box-shadow:0 8px 30px -10px rgba(34,211,238,.6);}

/* сенімділік — екі түйме */
[class*="st-key-conf_"] div[role="radiogroup"]{gap:10px;}
[class*="st-key-conf_"] label[data-baseweb="radio"]{margin:0 !important;padding:10px 20px;border-radius:999px;border:1px solid var(--line);
  background:var(--glass);cursor:pointer;transition:all .3s ease;animation:rise .4s ease .45s backwards;}
[class*="st-key-conf_"] label[data-baseweb="radio"] > div:first-child{display:none;}
[class*="st-key-conf_"] label[data-baseweb="radio"]:nth-child(1):has(input:checked){background:rgba(52,211,153,.16);border-color:#34d399;box-shadow:0 0 22px -6px #34d399;}
[class*="st-key-conf_"] label[data-baseweb="radio"]:nth-child(2):has(input:checked){background:rgba(251,191,36,.16);border-color:#fbbf24;box-shadow:0 0 22px -6px #fbbf24;}

/* статистика және нәтиже сақинасы */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 14px;}
.stat{display:flex;align-items:center;gap:16px;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:18px 20px;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);animation:enter .6s cubic-bezier(.2,.8,.2,1) backwards;transition:transform .3s;}
.stat:hover{transform:translateY(-4px);}
.stat svg{width:58px;height:58px;flex:none;transform:rotate(-90deg);}
.stat .bg,.gauge .bg{fill:none;stroke:rgba(255,255,255,.08);}
.stat .fg,.gauge .fg{fill:none;stroke-linecap:round;stroke-dasharray:100;animation:ring 1.4s cubic-bezier(.2,.8,.2,1) backwards;}
.stat .v{font-size:30px;font-weight:800;line-height:1;}
.stat .l{font-size:13px;color:var(--muted);margin-top:4px;}
.result{display:grid;grid-template-columns:auto 1fr;gap:28px;align-items:center;}
.gauge{position:relative;width:170px;height:170px;}
.gauge svg{width:100%;height:100%;transform:rotate(-90deg);}
.gauge .pc{position:absolute;inset:0;display:grid;place-items:center;text-align:center;}
.gauge .pc b{font-size:40px;font-weight:800;display:block;line-height:1;animation:popIn .8s ease .5s backwards;}
.gauge .pc span{font-size:12px;color:var(--muted);}

/* ---------- майндмап ---------- */
.map-wrap{overflow-x:auto;background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:22px;padding:6px;animation:rise .5s ease backwards;}
.map{position:relative;height:440px;min-width:800px;}
.map svg{position:absolute;inset:0;width:100%;height:100%;}
.map path{fill:none;stroke:rgba(255,255,255,.16);stroke-width:2;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1s ease forwards;}
.map path.flow{stroke:url(#gr);stroke-width:2.4;stroke-dasharray:.04 .1;stroke-dashoffset:0;opacity:0;animation:flow 2.4s linear infinite, fadeIn .6s ease forwards;}
@keyframes fadeIn{to{opacity:1}}
.node{position:absolute;width:17.5%;opacity:0;animation:pop .6s cubic-bezier(.2,.8,.25,1.2) forwards;}
.node-in{position:relative;padding:12px 13px;border-radius:15px;border:1.5px solid;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  animation:float 6s ease-in-out infinite;transition:box-shadow .25s ease, scale .25s ease;cursor:default;}
.node-in:hover{scale:1.06;box-shadow:0 14px 34px -8px currentColor;}
.node.weak .node-in::after{content:"";position:absolute;inset:-2px;border-radius:16px;border:2px solid #f87171;animation:pulse 1.8s ease-out infinite;}
.node .id{font-size:11px;font-weight:800;letter-spacing:.08em;}
.node .nm{font-size:13px;font-weight:600;line-height:1.28;color:#fff;margin-top:3px;}
.node .bar{height:4px;border-radius:4px;background:rgba(255,255,255,.1);margin-top:9px;overflow:hidden;}
.node .bar i{display:block;height:100%;border-radius:4px;animation:grow 1.1s ease backwards;}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 6px 4px;color:var(--muted);font-size:13px;align-items:center;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;box-shadow:0 0 8px currentColor;}

/* ---------- кесте ---------- */
.tbl{width:100%;border-collapse:separate;border-spacing:0;background:var(--glass);border:1px solid var(--line);border-radius:20px;overflow:hidden;margin-top:6px;}
.tbl th{font-size:12px;font-weight:700;color:var(--muted);text-align:left;padding:13px 16px;background:rgba(255,255,255,.03);border-bottom:1px solid var(--line);letter-spacing:.05em;text-transform:uppercase;}
.tbl td{padding:13px 16px;border-bottom:1px solid rgba(255,255,255,.05);font-size:14px;color:var(--ink);}
.tbl tr{transition:background .2s;} .tbl tr:hover td{background:rgba(255,255,255,.035);}
.tbl tr:last-child td{border-bottom:none;}
.pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;font-size:12.5px;font-weight:700;white-space:nowrap;}
.pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor;}
.mini{height:6px;width:120px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px;}
.mini i{display:block;height:100%;border-radius:6px;animation:grow 1s ease backwards;}

/* ---------- оқу бағыты ---------- */
.route{position:relative;margin:10px 0 0 20px;padding-left:34px;}
.route::before{content:"";position:absolute;left:0;top:6px;bottom:6px;width:2px;border-radius:2px;transform-origin:top;
  background:linear-gradient(var(--c1),var(--c2),var(--c3));animation:growY 1.4s cubic-bezier(.2,.8,.2,1) backwards;}
.step{position:relative;margin-bottom:14px;animation:slideIn .6s cubic-bezier(.2,.8,.2,1) backwards;}
.step .num{position:absolute;left:-54px;top:20px;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;font-weight:800;color:#0b1120;font-size:15px;
  box-shadow:0 0 0 5px var(--bg),0 0 22px -2px currentColor;animation:popIn .6s ease backwards;animation-delay:inherit;}
.step .card{transition:transform .3s,border-color .3s;} .step .card:hover{transform:translateX(6px);border-color:rgba(255,255,255,.2);}
.step h4{margin:0 0 4px;font-size:17px;}
.step .why{font-size:13px;font-weight:700;margin-bottom:8px;}
.step p{margin:0 0 10px;color:#b6bfd3;line-height:1.55;font-size:14.5px;}
.step a{color:var(--c1);font-weight:700;text-decoration:none;}
.done{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.4);color:#6ee7b7;border-radius:18px;padding:18px 22px;font-weight:700;animation:enter .6s ease backwards;}

/* ---------- мұғалім ---------- */
.insight{border-left:4px solid #f87171;}
.bars .row{display:grid;grid-template-columns:minmax(160px,300px) 1fr 44px;align-items:center;gap:14px;margin:10px 0;font-size:14px;}
.track{height:12px;background:rgba(255,255,255,.07);border-radius:6px;overflow:hidden;}
.fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#fb7185,#f87171,#fbbf24);box-shadow:0 0 14px rgba(248,113,113,.5);animation:grow 1.2s cubic-bezier(.2,.8,.2,1) backwards;}
.heat{display:grid;grid-template-columns:64px repeat(8,1fr);gap:5px;font-size:11.5px;color:var(--muted);}
.heat .c{height:18px;border-radius:6px;animation:wave .5s cubic-bezier(.2,.8,.2,1.4) backwards;transition:transform .15s;}
.heat .c:hover{transform:scale(1.35);z-index:2;}
.heat .hd{text-align:center;font-weight:700;}

/* ---------- Streamlit виджеттері ---------- */
.stButton>button,.stDownloadButton>button{position:relative;overflow:hidden;border-radius:14px;padding:.68rem 1.5rem;font-weight:700;
  border:1px solid var(--line);background:var(--glass);color:var(--ink);transition:transform .2s, box-shadow .3s, border-color .3s;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);border-color:rgba(34,211,238,.5);box-shadow:0 10px 30px -10px rgba(34,211,238,.6);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--c1),var(--c2));border:none;color:#fff;box-shadow:0 10px 30px -12px rgba(139,92,246,.9);}
.stButton>button[kind="primary"]::after{content:"";position:absolute;top:0;bottom:0;left:0;width:40%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);animation:shine 3.2s ease-in-out infinite;}
.stButton>button:disabled{opacity:.4;box-shadow:none;}
.stButton>button:disabled::after{display:none;}
[data-baseweb="tab-list"]{gap:4px;background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:16px;padding:5px;width:fit-content;max-width:100%;overflow-x:auto;}
button[data-baseweb="tab"]{border-radius:12px !important;padding:8px 18px !important;transition:background .3s;}
button[data-baseweb="tab"][aria-selected="true"]{background:rgba(255,255,255,.08);}
button[data-baseweb="tab"] p{font-weight:700;}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--c1),var(--c2)) !important;height:3px !important;border-radius:3px;}
[data-baseweb="tab-border"]{display:none;}
[data-baseweb="tab-panel"]{animation:enter .55s cubic-bezier(.2,.8,.2,1) backwards;}
[data-testid="stExpander"] details{background:var(--glass);border-radius:16px;border:1px solid var(--line);}
[data-testid="stAlert"]{border-radius:16px;}

/* ---------- жаңа: агент, код, ішкі жұмыс ---------- */
.agent{display:flex;gap:10px;align-items:center;margin:10px 0 2px;padding:10px 14px;border-radius:14px;font-size:13.5px;color:#c7d2fe;
  background:rgba(139,92,246,.08);border:1px dashed rgba(139,92,246,.35);animation:rise .5s ease .3s backwards;}
.agent b{color:#fff;}
.peas{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:6px 0 10px;}
.peas .card{padding:16px 18px;} .peas .k{font-size:28px;font-weight:800;background:linear-gradient(135deg,var(--c1),var(--c2));
  -webkit-background-clip:text;background-clip:text;color:transparent;} .peas p{margin:4px 0 0;color:#b6bfd3;font-size:13.5px;line-height:1.45;}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 12px;}
.kpi .card b{display:block;font-size:28px;font-weight:800;color:#fff;} .kpi .card span{color:var(--muted);font-size:13px;}
[data-testid="stCode"]{border-radius:16px;}
@media (max-width:760px){.peas,.kpi{grid-template-columns:1fr 1fr}}

@media (max-width:760px){
  .stats,.how{grid-template-columns:1fr}.result{grid-template-columns:1fr;justify-items:center;text-align:center}
  .hero{padding:24px 22px}.hero h1{font-size:1.65rem}.q-text{font-size:19px}.qnum{font-size:36px}
  .qhead{flex-direction:column;align-items:stretch}.segs{max-width:none}
  .st-key-nav div[role="radiogroup"]{flex-wrap:wrap;border-radius:20px}.st-key-nav label[data-baseweb="radio"]{padding:8px 14px}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;}}
</style>
"""
st.html(CSS)


# ---------- Деректер мен модельдер ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/topics.csv"), pd.read_csv("data/questions.csv").fillna({"code": ""})


@st.cache_resource
def load_models():
    topics_, _ = load_data()
    net_ = KnowledgeNet(topics_["topic_id"], parse_prereqs(topics_))
    try:
        lr = joblib.load("model.joblib")
    except Exception:  # файл жоқ болса — қайта оқытамыз
        from train import train_model
        lr = train_model()[0]
    try:
        qt = load_q_table()
    except Exception:
        qt = None
    return net_, lr, qt


topics, questions = load_data()
net, lr_model, q_table = load_models()
prereqs = parse_prereqs(topics)
depth = depth_of(prereqs)
TOPICS = list(topics["topic_id"])
topic_name = dict(zip(topics["topic_id"], topics["name"]))
test_q = questions[questions["use"] == "test"].reset_index(drop=True)
q_by_id = questions.set_index("q_id")


def pill(level):
    fg, bg = LV[level]
    return f'<span class="pill" style="color:{fg};background:{bg}">{LEVEL_NAMES[level]}</span>'


def hero(eyebrow, title, text, accent_last=True):
    """Үлкен тақырып: сөздер кезекпен шығады, соңғы сөз — градиентпен."""
    words = title.split()
    spans = "".join(
        f'<span class="{"g" if accent_last and k == len(words) - 1 else ""}" style="animation-delay:{0.08 + k * 0.07:.2f}s">'
        f'{h.escape(w)}</span>' for k, w in enumerate(words))
    sparks = "".join(f'<i class="spark" style="left:{8 + k * 11}%;animation-delay:{-k * 0.8:.1f}s;'
                     f'animation-duration:{5 + k % 3}s"></i>' for k in range(8))
    st.html(f"""
    <div class="hero">
      <div class="orb" style="width:240px;height:240px;right:-60px;top:-90px;background:radial-gradient(circle,rgba(34,211,238,.55),transparent 65%)"></div>
      <div class="orb" style="width:220px;height:220px;right:200px;bottom:-130px;background:radial-gradient(circle,rgba(139,92,246,.6),transparent 65%);animation-delay:-3s"></div>
      {sparks}
      <div class="eyebrow"><i></i>{h.escape(eyebrow)}</div><h1>{spans}</h1><p>{h.escape(text)}</p>
    </div>""")


def ring_svg(value, color, cls="fg"):
    return (f'<svg viewBox="0 0 120 120"><circle class="bg" cx="60" cy="60" r="50" stroke-width="12"/>'
            f'<circle class="{cls}" cx="60" cy="60" r="50" stroke-width="12" pathLength="100" '
            f'style="stroke:{color};stroke-dashoffset:{100 - value:.1f};filter:drop-shadow(0 0 6px {color})"/></svg>')


def answers_to_evidence(answers):
    return [(a["topic_id"], a["correct"], a["confident"]) for a in answers]


# ---------- Нәтижені сақтау ----------
def set_result(answers, history, log, label):
    lp = net.from_answers(answers_to_evidence(answers))
    p = net.marginals_dict(lp)
    st.session_state.result = {"p": p, "answers": answers, "history": history, "log": log, "label": label}
    st.session_state.live = dict(p)          # оқу кезінде HMM/BKT арқылы жаңарады
    st.session_state.live_log = []
    st.session_state.checked = {}
    st.session_state.explained = {}


def current_p():
    return st.session_state.get("live") or st.session_state.result["p"]


def levels_of(p):
    return {t: to_level(v) for t, v in p.items()}


# ---------- Майндмап ----------
def layout():
    max_d = max(depth.values()) or 1
    pos = {}
    for col in range(max_d + 1):
        ids = [t for t in TOPICS if depth[t] == col]
        ids.sort(key=lambda t: sum(pos[p][1] for p in prereqs[t]) / len(prereqs[t]) if prereqs[t] else 0)
        n = len(ids)
        ys = [220] if n == 1 else [90 + i * (260 / (n - 1)) for i in range(n)]
        for t, y in zip(ids, ys):
            pos[t] = (90 + col * (820 / max_d), y)
    return pos


def show_map(p):
    levels = levels_of(p)
    pos = layout()
    paths = []
    for t, ps in prereqs.items():
        for q in ps:
            (x1, y1), (x2, y2) = pos[q], pos[t]
            xm = (x1 + x2) / 2
            d = f"M{x1:.0f} {y1:.0f} C{xm:.0f} {y1:.0f},{xm:.0f} {y2:.0f},{x2:.0f} {y2:.0f}"
            delay = 0.25 + depth[q] * 0.2
            paths.append(f'<path pathLength="1" vector-effect="non-scaling-stroke" style="animation-delay:{delay:.2f}s" d="{d}"/>'
                         f'<path class="flow" pathLength="1" vector-effect="non-scaling-stroke" '
                         f'style="animation-delay:0s,{delay + 1:.2f}s" d="{d}"/>')
    nodes = []
    for i, t in enumerate(TOPICS):
        x, y = pos[t]
        lv = levels[t]
        fg, bg = LV[lv]
        pct = int(round(p[t] * 100))
        nodes.append(
            f'<div class="node{" weak" if lv == 0 else ""}" '
            f'style="left:{x / 10:.1f}%;top:{y / 440 * 100:.1f}%;animation-delay:{depth[t] * 0.2 + i * 0.04:.2f}s">'
            f'<div class="node-in" style="background:linear-gradient({bg},{bg}),#0e1426;border-color:{fg};color:{fg};animation-delay:{-i * 0.7:.1f}s">'
            f'<div class="id">{t} · {LEVEL_NAMES[lv]} · {pct}%</div>'
            f'<div class="nm">{h.escape(topic_name[t])}</div>'
            f'<div class="bar"><i style="width:{pct}%;background:{fg};animation-delay:{0.6 + depth[t] * 0.2:.2f}s"></i></div>'
            f'</div></div>')
    legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]};color:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                     for k in (2, 1, 0))
    st.html(f"""
    <div class="map-wrap"><div class="map">
      <svg viewBox="0 0 1000 440" preserveAspectRatio="none">
        <defs><linearGradient id="gr" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>
        {''.join(paths)}</svg>
      {''.join(nodes)}
    </div></div>
    <div class="legend">{legend}<span style="margin-left:auto">Жолақ — P(біледі), Байес желісі бойынша · сызық — пререквизит</span></div>""")


def show_table(p, answers):
    levels = levels_of(p)
    df = pd.DataFrame(answers) if answers else pd.DataFrame(columns=["topic_id", "correct", "confident"])
    rows = []
    for i, t in enumerate(TOPICS):
        part = df[df["topic_id"] == t]
        n = len(part)
        right = int(part["correct"].sum()) if n else 0
        guess = int((part["confident"] == 0).sum()) if n else 0
        pct = int(round(p[t] * 100))
        fg = LV[levels[t]][0]
        asked = f"{right} / {n}" if n else '<span style="color:#8b93a7">сұралмады — граф бойынша</span>'
        rows.append(f'<tr style="animation:slideIn .45s ease backwards;animation-delay:{i * 0.06:.2f}s">'
                    f'<td><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</td>'
                    f'<td><span class="mini"><i style="width:{pct}%;background:{fg};animation-delay:{0.2 + i * 0.06:.2f}s"></i></span>{pct}%</td>'
                    f'<td>{asked}</td><td>{guess}</td><td>{pill(levels[t])}</td></tr>')
    st.html(f'<table class="tbl"><tr><th>Тақырып</th><th>P(біледі)</th><th>Дұрыс / сұрақ</th><th>Болжадым</th>'
            f'<th>Деңгей</th></tr>{"".join(rows)}</table>')


def show_stats(p):
    levels = levels_of(p)
    counts = {k: sum(1 for v in levels.values() if v == k) for k in (2, 1, 0)}
    n = len(levels) or 1
    tiles = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.1:.2f}s">{ring_svg(counts[k] / n * 100, LV[k][0])}'
        f'<div><div class="v" style="color:{LV[k][0]}">{counts[k]}<span style="font-size:15px;color:#8b93a7"> / {n}</span></div>'
        f'<div class="l">{LEVEL_NAMES[k]}</div></div></div>' for i, k in enumerate((2, 1, 0)))
    st.html(f'<div class="stats">{tiles}</div>')


# ---------- Оқу бағыты (A*) + жаттығу (HMM/BKT) + LLM ----------
def show_route(key_prefix="r"):
    p_test = st.session_state.result["p"]
    route, prob, stats = build_route(p_test, levels_of(p_test), topics, prereqs)
    if not route:
        st.html('<div class="done">Барлық тақырыптар жасыл — оқу бағыты қажет емес. Жарайсыз!</div>')
        return
    total_h = sum(s["hours"] for s in route)
    steps = []
    for i, s in enumerate(route, 1):
        fg, bg = LV[s["level"]]
        steps.append(f"""
        <div class="step" style="animation-delay:{0.15 + i * 0.14:.2f}s">
          <div class="num" style="background:{fg};color:{fg}"><span style="color:#0b1120">{i}</span></div>
          <div class="card" style="animation:none">
            <h4>{s['topic_id']} · {h.escape(s['name'])}</h4>
            <div class="why" style="color:{fg}">{h.escape(s['reason'])}</div>
            <p>{h.escape(s['explanation'])}</p>
            <div style="font-size:13.5px;color:#8b93a7">{h.escape(s['key_points'])} · ≈ {s['hours']:.1f} сағ ·
              <a href="{h.escape(s['material'])}" target="_blank">қосымша оқу →</a></div>
          </div>
        </div>""")
    st.html(f'<p style="color:#b6bfd3;margin:4px 0 6px"><b style="color:#fff">{len(route)} тақырып</b>, барлығы ≈ {total_h:.0f} сағ. '
            f'Ретті A* алгоритмі тапты: көп балл беретін тақырыптар ертерек, бірақ пререквизиттер бұзылмайды '
            f'(ашылған түйін: {stats["expanded"]}).</p><div class="route">{"".join(steps)}</div>')
    with st.expander("Неге дәл осы рет? BFS / UCS / A* салыстыруы"):
        st.dataframe(pd.DataFrame(compare_methods(prob, prereqs)), hide_index=True)
        st.caption("Құн = Σ (тақырыптың күтілетін балы × ол меңгерілетін уақыт). UCS пен A* — оңтайлы, "
                   "бірақ A* эвристиканың арқасында түйінді аз ашады.")

    st.markdown("#### Жаттығу — білім картасы жауап сайын жаңарады")
    st.caption("Әр жауаптан кейін P(біледі) HMM (Bayesian Knowledge Tracing) бойынша жаңартылады: "
               "алдымен «үйрену» ауысуы, содан кейін жауапқа қарап Байес бойынша түзету.")
    for s in route:
        t = s["topic_id"]
        with st.expander(f"{t} · {s['name']} — қазір P(біледі) = {st.session_state.live[t]:.0%}"):
            practice = questions[(questions["topic_id"] == t) & (questions["use"] == "practice")]
            for q in practice.itertuples():
                key = f"{key_prefix}_{q.q_id}"
                opts = {"a": q.a, "b": q.b, "c": q.c, "d": q.d}
                st.markdown(f"**{q.question}**")
                if q.code:
                    st.code(q.code, language="sql" if t == "T9" else ("python" if t in ("T7", "T8") else None))
                choice = st.radio("Жауап", list(opts), format_func=lambda k, o=opts: f"{k})  {o[k]}",
                                  index=None, key=key, label_visibility="collapsed")
                conf = st.radio("Сенімділік", ["Сенімдімін", "Болжадым"], index=None, horizontal=True,
                                key=f"conf_{key}", label_visibility="collapsed")
                if choice and conf and key not in st.session_state.checked:
                    if st.button("Тексеру", key=f"chk_{key}"):
                        correct = int(choice == q.correct)
                        before = st.session_state.live[t]
                        after = bkt_step(before, correct, int(conf == "Сенімдімін"))
                        st.session_state.live[t] = after
                        st.session_state.checked[key] = (choice, correct, before, after)
                        st.session_state.live_log.append({"Тақырып": t, "Сұрақ": q.q_id, "Дұрыс": "иә" if correct else "жоқ",
                                                          "Сенімділік": conf, "P(біледі) бұрын": round(before, 3),
                                                          "P(біледі) кейін": round(after, 3)})
                        st.rerun()
                if key in st.session_state.checked:
                    ch, correct, before, after = st.session_state.checked[key]
                    if correct:
                        st.success(f"Дұрыс! P(біледі): {before:.0%} → {after:.0%}")
                    else:
                        st.error(f"Қате. Дұрыс жауап (кілт бойынша): {q.correct}) {opts[q.correct]}. "
                                 f"P(біледі): {before:.0%} → {after:.0%}")
                        if key in st.session_state.explained:
                            text, src = st.session_state.explained[key]
                            st.info(text)
                            st.caption(f"Дереккөз: {src}. Дұрыс жауапты ЖИ емес, жауап кілті анықтайды.")
                        elif st.button("Неге қате? — ЖИ түсіндіреді", key=f"why_{key}"):
                            with st.spinner("Түсіндірме дайындалуда..."):
                                st.session_state.explained[key] = explain(q_by_id.loc[q.q_id].to_dict() | {"q_id": q.q_id}, ch)
                            st.rerun()
                st.divider()


def result_tabs(key_prefix):
    p = current_p()
    t1, t2, t3 = st.tabs(["Білім картасы", "Кесте", "Оқу бағыты"])
    with t1:
        show_map(p)
    with t2:
        show_table(p, st.session_state.result["answers"])
    with t3:
        show_route(key_prefix=key_prefix)


def need_result():
    if "result" not in st.session_state:
        st.info("Алдымен тестті тапсырыңыз немесе «Демо» бөлімінен мысал таңдаңыз.")
        return False
    return True


# ---------- Жоғарғы бренд пен мәзір ----------
st.html("""
<div class="topbar">
  <div class="logo-wrap"><div class="logo"></div>
    <svg viewBox="0 0 24 24"><line x1="6" y1="7" x2="12" y2="12"/><line x1="18" y1="6" x2="12" y2="12"/><line x1="12" y1="12" x2="12" y2="19"/>
      <circle cx="6" cy="7" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="12" r="2.8" style="fill:#8b5cf6"/><circle cx="12" cy="19" r="2.4"/></svg></div>
  <div><div class="brand">ҰБТ Навигатор</div><div class="brand-sub">AI Fundamentals · қорытынды жоба · информатика</div></div>
</div>""")
page = st.radio("Бөлім", ["Тест", "Менің картам", "Оқу бағытым", "ҰБТ жоспары", "Ішінде не болды", "Мұғалім", "Демо"],
                horizontal=True, label_visibility="collapsed", key="nav")

if "started" not in st.session_state:
    st.session_state.update(started=False, finished=False, answers=[], history=[], log=[], cur=None)


# ---------- 1. Тест ----------
if page == "Тест":
    if not st.session_state.started:
        hero("Бейімделгіш тест", "Информатиканың қай тақырыбын білесіз?",
             "ҰБТ информатика спецификациясының 9 тақырыбы. Барлық 45 сұрақтың орнына ЖИ-агент "
             "ең көп ақпарат беретін сұрақтарды өзі таңдайды — әдетте 15–20 сұрақ жеткілікті.")
        st.html("""<div class="how">
          <div class="card" style="animation-delay:.15s"><div class="n">01</div><h4>Бейімделгіш тест</h4><p>Агент келесі сұрақты сіздің жауаптарыңызға қарай таңдайды</p></div>
          <div class="card" style="animation-delay:.27s"><div class="n">02</div><h4>Байес картасы</h4><p>Әр тақырып үшін P(біледі) — тіпті сұралмаған тақырыптар үшін де</p></div>
          <div class="card" style="animation-delay:.39s"><div class="n">03</div><h4>A* бағыты</h4><p>Оңтайлы оқу реті, ҰБТ жоспары және ЖИ түсіндірмелері</p></div>
        </div>""")
        code = st.text_input("Студент коды — атыңызды жазбаңыз (мысалы, R01)", value="R01")
        options = ["expectimax", "rl", "random"] if q_table is not None else ["expectimax", "random"]
        policy = st.radio("Сұрақтарды қай агент таңдайды?", options, format_func=lambda k: POLICY_NAMES[k], horizontal=True)
        if st.button("Тестті бастау", type="primary"):
            st.session_state.update(started=True, finished=False, answers=[], log=[], cur=None, code=code, policy=policy,
                                    history=[net.marginals(net.start())], q_start=time.time())
            st.rerun()

    elif not st.session_state.finished:
        answers = st.session_state.answers
        asked = [a["q_id"] for a in answers]
        lp = net.from_answers(answers_to_evidence(answers))
        avail = available_topics(test_q, asked)
        if should_stop(net, lp, len(asked), avail):
            set_result(answers, st.session_state.history, st.session_state.log, f"Студент {st.session_state.code}")
            st.session_state.finished = True
            st.rerun()
        if st.session_state.cur is None:
            rng = np.random.default_rng(len(asked) + 1000 * abs(hash(st.session_state.code)) % 10_000)
            t, scores, why = choose_topic(net, lp, avail, st.session_state.policy, q_table, rng)
            q = pick_question(test_q, t, asked, rng)
            st.session_state.cur = {"q_id": q["q_id"], "topic": t, "why": why, "scores": scores}
        cur = st.session_state.cur
        q = q_by_id.loc[cur["q_id"]]
        i = len(asked)
        segs = "".join(
            f'<div class="seg{" s-ok" if k < i else ""}{" s-new" if k == i - 1 else ""}{" s-cur" if k == i else ""}"></div>'
            for k in range(MAX_Q))
        st.html(f'<div class="qhead"><div class="qnum">{i + 1:02d}<small> / ≤{MAX_Q}</small></div>'
                f'<div class="segs" style="grid-template-columns:repeat({MAX_Q},1fr)">{segs}</div></div>')
        st.html(f'<div class="qcard"><div class="wm">{cur["topic"]}</div>'
                f'<span class="chip">{cur["topic"]} · {h.escape(topic_name[cur["topic"]])}</span>'
                f'<div class="q-text">{h.escape(q["question"])}</div></div>')
        if q["code"]:
            st.code(q["code"], language="sql" if cur["topic"] == "T9" else ("python" if cur["topic"] in ("T7", "T8") else None))
        opts = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"]}
        choice = st.radio("Жауапты таңдаңыз", list(opts), format_func=lambda k: f"{k})  {opts[k]}",
                          index=None, key=f"ans_{i}")
        conf = st.radio("Жауабыңызға сенімдісіз бе?", ["Сенімдімін", "Болжадым"], index=None,
                        horizontal=True, key=f"conf_{i}")
        st.html(f'<div class="agent">🤖 <span><b>{h.escape(POLICY_NAMES[st.session_state.policy])}</b> осы тақырыпты таңдады — '
                f'{h.escape(cur["why"])}</span></div>')
        if st.button("Келесі сұрақ  →", type="primary", disabled=(choice is None or conf is None)):
            a = {"q_id": cur["q_id"], "topic_id": cur["topic"], "correct": int(choice == q["correct"]),
                 "confident": int(conf == "Сенімдімін"), "time_sec": round(time.time() - st.session_state.q_start, 1)}
            st.session_state.answers.append(a)
            lp2 = net.update(lp, a["topic_id"], a["correct"], a["confident"])
            st.session_state.history.append(net.marginals(lp2))
            st.session_state.log.append({"step": i + 1, "topic": cur["topic"], "why": cur["why"], "scores": cur["scores"]})
            st.session_state.cur = None
            st.session_state.q_start = time.time()
            st.rerun()

    else:
        answers = st.session_state.answers
        p = current_p()
        right = sum(a["correct"] for a in answers)
        n = len(answers)
        st.html(f"""<div class="hero"><div class="result">
          <div class="gauge">{ring_svg(right / max(n, 1) * 100, "#22d3ee")}
            <div class="pc"><div><b>{n}</b><span>сұрақ ({right} дұрыс)</span></div></div></div>
          <div><div class="eyebrow"><i></i>Тест аяқталды</div>
            <h1><span>Сіздің</span><span class="g" style="animation-delay:.1s">білім картаңыз</span></h1>
            <p>Толық тест 45 сұрақтан тұрады — агент {n} сұрақпен ({(1 - n / 45):.0%} аз) барлық 9 тақырыптың деңгейін бағалады.
            Сұралмаған тақырыптар Байес желісі арқылы бағаланды.</p></div>
        </div></div>""")
        show_stats(p)
        result_tabs(key_prefix="r")
        out = pd.DataFrame(answers)
        out.insert(0, "student_id", st.session_state.code)
        out["p_know_after_test"] = out["topic_id"].map(st.session_state.result["p"]).round(3)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Жауаптарды жүктеп алу (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name=f"answers_{st.session_state.code}.csv", mime="text/csv")
        with c2:
            if st.button("Тестті қайта тапсыру"):
                st.session_state.update(started=False, finished=False, answers=[], log=[], cur=None)
                st.rerun()


# ---------- 2. Менің картам ----------
elif page == "Менің картам":
    hero("Нәтиже", "Менің білім картам",
         "Түсі — P(біледі) бойынша деңгей. Жаттығу орындаған сайын карта жаңарады (HMM).")
    if need_result():
        p = current_p()
        show_stats(p)
        t1, t2 = st.tabs(["Білім картасы", "Кесте"])
        with t1:
            show_map(p)
        with t2:
            show_table(p, st.session_state.result["answers"])


# ---------- 3. Оқу бағытым ----------
elif page == "Оқу бағытым":
    hero("Жеке жоспар", "Менің оқу бағытым",
         "A* алгоритмі әлсіз тақырыптардың оңтайлы ретін табады: көп балл беретіндері ертерек, негізі бұзылмай.")
    if need_result():
        show_route(key_prefix="route")


# ---------- 4. ҰБТ жоспары ----------
elif page == "ҰБТ жоспары":
    hero("Уақыт шектеулі", "ҰБТ-ға дейінгі жоспар",
         "Барлығын оқуға уақыт жетпесе — қай тақырыптарды таңдау керек? Жасыту әдісі (simulated annealing) "
         "күтілетін баллды максимумға жеткізетін жиынды іздейді.")
    if need_result():
        c1, c2 = st.columns(2)
        days = c1.slider("ҰБТ-ға дейін неше күн қалды?", 1, 60, 10)
        per_day = c2.slider("Күніне неше сағат оқи аласыз?", 0.5, 6.0, 1.5, step=0.5)
        budget = days * per_day
        p = current_p()
        _, prob, _ = build_route(p, levels_of(p), topics, prereqs)
        if not prob.get("targets"):
            st.html('<div class="done">Әлсіз тақырып жоқ — жоспар қажет емес.</div>')
        else:
            best, best_v, hist = anneal(prob, budget, seed=days * 10 + int(per_day * 2))
            _, opt_v = brute_force(prob, budget)
            total_v = sum(prob["weight"].values())
            used = sum(prob["hours"][t] for t in best)
            st.html(f"""<div class="kpi">
              <div class="card"><b>{len(best)} / {len(prob['targets'])}</b><span>тақырып таңдалды</span></div>
              <div class="card"><b>+{best_v:.1f}</b><span>күтілетін балл (мүмкін {total_v:.1f}-ден)</span></div>
              <div class="card"><b>{used:.1f} / {budget:.1f}</b><span>сағат қолданылды</span></div></div>""")
            if abs(best_v - opt_v) < 1e-6:
                st.success(f"Тексеріс: толық перебор да {opt_v:.1f} балл тапты — жасыту әдісі оңтайлы шешімге жетті.")
            else:
                st.warning(f"Толық перебор {opt_v:.1f} балл тапты, SA — {best_v:.1f}. Жуық әдіс әрдайым оңтайлы бола бермейді.")
            if best:
                sub = dict(prob, targets=[t for t in prob["targets"] if t in best])
                order, _, _ = search(sub, "astar")
                plan = pd.DataFrame(schedule(order, prob, per_day))
                plan["Тақырып"] = plan["Тақырып"].map(lambda t: f"{t} · {topic_name[t]}")
                st.markdown("#### Күндер бойынша жоспар")
                st.dataframe(plan, hide_index=True)
                skipped = [t for t in prob["targets"] if t not in best]
                if skipped:
                    st.caption("Уақыт жетпейтіндіктен қалдырылды: " + ", ".join(f"{t} · {topic_name[t]}" for t in skipped))
            st.markdown("#### Жасыту әдісі қалай жұмыс істеді")
            st.line_chart(pd.DataFrame(hist).set_index("қадам")[["ағымдағы", "ең жақсы"]])
            st.caption("Басында температура жоғары — алгоритм нашар қадамдарды да қабылдайды (ағымдағы мән секіреді). "
                       "Температура төмендеген сайын тек жақсартулар қалады.")


# ---------- 5. Ішінде не болды ----------
elif page == "Ішінде не болды":
    hero("ЖИ ішінен", "Ішінде не болды?",
         "Жобадағы әр алгоритмнің жұмысы: агент, Байес желісі, HMM, A*, MCMC, логистикалық регрессия және тәжірибе нәтижелері.")
    st.html("""<div class="peas">
      <div class="card"><div class="k">P</div><p><b>Performance:</b> деңгейді дәл анықтау, сұрақ саны аз, ҰБТ балының өсуі</p></div>
      <div class="card"><div class="k">E</div><p><b>Environment:</b> студент (ішкі білімі жасырын) және сұрақтар банкі</p></div>
      <div class="card"><div class="k">A</div><p><b>Actuators:</b> келесі сұрақты таңдау, оқу бағыты мен жоспар беру</p></div>
      <div class="card"><div class="k">S</div><p><b>Sensors:</b> жауаптың дұрыстығы және «сенімдімін / болжадым» белгісі</p></div>
    </div>""")
    if need_result():
        res = st.session_state.result
        st.caption(f"Қаралып отырған нәтиже: {res['label']}")
        tabs = st.tabs(["Агент шешімдері", "Байес vs LogReg", "A* іздеу", "MCMC", "HMM (жаттығу)", "Тәжірибелер"])
        with tabs[0]:
            hist = pd.DataFrame(res["history"], columns=TOPICS)
            hist.index.name = "сұрақ"
            st.markdown("**P(біледі) әр жауаптан кейін қалай өзгерді** (0 — тест басталғанға дейінгі априорлық мән)")
            st.line_chart(hist)
            log = res["log"]
            if log:
                rows = []
                for e, a in zip(log, res["answers"]):
                    rows.append({"Қадам": e["step"], "Тақырып": e["topic"], "Себебі": e["why"],
                                 "Жауап": ("дұрыс" if a["correct"] else "қате") + (", сенімді" if a["confident"] else ", болжады")})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                step = st.slider("Қай қадамдағы агент бағаларын көрсету?", 1, len(log), 1)
                sc = log[step - 1]["scores"]
                if sc:
                    st.markdown(f"**{step}-қадам: әр тақырыптың бағасы** (ақпарат пайдасы, бит немесе Q-мәні) — ең үлкені таңдалды")
                    st.bar_chart(pd.Series(sc, name="баға"))
                else:
                    st.caption("Кездейсоқ агент бағаламайды.")
        with tabs[1]:
            ans_df = pd.DataFrame(res["answers"])
            lr = predict_levels(topic_features(ans_df[["q_id", "topic_id", "correct", "confident"]], questions, topics), lr_model)
            comp = pd.DataFrame({"Тақырып": [f"{t} · {topic_name[t]}" for t in TOPICS],
                                 "Сұрақ саны": [int((ans_df["topic_id"] == t).sum()) for t in TOPICS],
                                 "P(біледі), Байес": [round(res["p"][t], 3) for t in TOPICS],
                                 "Деңгей, Байес желісі": [LEVEL_NAMES[to_level(res["p"][t])] for t in TOPICS],
                                 "Деңгей, LogReg": [LEVEL_NAMES[lr[t]] for t in TOPICS]})
            st.dataframe(comp, hide_index=True)
            st.caption("Логистикалық регрессия әр тақырыпты бөлек қарайды, сұралмаған тақырып туралы ештеңе білмейді. "
                       "Байес желісі пререквизиттер арқылы сұралмаған тақырыптарды да бағалайды.")
        with tabs[2]:
            _, prob, stats = build_route(res["p"], levels_of(res["p"]), topics, prereqs)
            if prob.get("targets"):
                st.dataframe(pd.DataFrame(compare_methods(prob, prereqs)), hide_index=True)
                st.markdown("- **Күй:** меңгерілген тақырыптар жиыны · **Әрекет:** пререквизиті дайын тақырыпты оқу\n"
                            "- **g(n):** Σ күтілетін балл × меңгеру уақыты · **h(n):** пререквизитсіз Смит ережесі (асырмайды)\n"
                            "- **f(n) = g(n) + h(n)** — A* ең кіші f-ті ашады")
            else:
                st.info("Әлсіз тақырып жоқ — іздеудің қажеті жоқ.")
        with tabs[3]:
            st.markdown("Дәл есептеу 2⁹ = 512 күйдің бәрін санайды. Гиббс сэмплингі (MCMC) сол ықтималдықтарды "
                        "кездейсоқ үлгілер арқылы жуықтайды — үлкен желілерде дәл есептеу мүмкін болмағанда қолданылады.")
            n_s = st.select_slider("Үлгі саны", [100, 300, 1000, 3000], value=1000)
            if st.button("Гиббс сэмплингін іске қосу"):
                with st.spinner("Сэмплинг..."):
                    g = net.gibbs(answers_to_evidence(res["answers"]), n_samples=n_s, burn_in=100)
                st.dataframe(pd.DataFrame({"Тақырып": TOPICS,
                                           "Дәл (enumeration)": [round(res["p"][t], 3) for t in TOPICS],
                                           "Гиббс (MCMC)": [round(g[t], 3) for t in TOPICS],
                                           "Айырма": [round(abs(g[t] - res["p"][t]), 3) for t in TOPICS]}),
                             hide_index=True)
        with tabs[4]:
            if st.session_state.get("live_log"):
                st.dataframe(pd.DataFrame(st.session_state.live_log), hide_index=True)
            else:
                st.info("«Оқу бағытым» бетінде жаттығу сұрақтарына жауап беріңіз — мұнда HMM жаңартулары шығады.")
            st.markdown("HMM: жасырын күй — «біледі / білмейді». **Ауысу:** жаттығу кезінде үйрену ықтималдығы P(learn). "
                        "**Бақылау:** P(жауап | күй) — білетін де қателесуі (slip), білмейтін де табуы (guess) мүмкін.")
        with tabs[5]:
            if os.path.exists("results/metrics.md"):
                for img, cap in [("results/accuracy_curve.png", "Сұрақ саны → дәлдік"),
                                 ("results/rl_learning_curve.png", "Q-learning үйрену қисығы"),
                                 ("results/search_nodes.png", "BFS / UCS / A*"),
                                 ("results/gibbs_error.png", "Гиббс vs дәл есептеу")]:
                    if os.path.exists(img):
                        st.image(img, caption=cap)
                with open("results/metrics.md", encoding="utf-8") as f:
                    st.markdown(f.read().replace("![](", "<!-- ").replace(".png)", " -->"))
            else:
                st.info("Нәтижелер жоқ. Алдымен `python evaluate.py` іске қосыңыз.")


# ---------- 6. Мұғалім ----------
elif page == "Мұғалім":
    hero("Мұғалім панелі", "Топтың нәтижесі",
         "Мысал ретінде — тест бөлігіндегі 60 жасанды студент (data/test_students.csv), Байес желісі бойынша.")

    @st.cache_data
    def group_levels():
        answers = pd.read_csv("data/students.csv").merge(questions[["q_id", "topic_id"]], on="q_id")
        ids = pd.read_csv("data/test_students.csv")["student_id"]
        out = {}
        for sid in ids:
            part = answers[answers["student_id"] == sid]
            p = net.marginals_dict(net.from_answers(list(zip(part["topic_id"], part["correct"], part["confident"]))))
            out[sid] = levels_of(p)
        return pd.DataFrame(out).T

    grid = group_levels()
    weak = (grid == 0).sum()
    worst = weak.idxmax()
    total = len(grid)
    st.html(f'<div class="card insight"><b style="color:#fff">Ең қиын тақырып: {worst} · '
            f'{h.escape(topic_name[worst])}</b><br><span style="color:#8b93a7">{weak.max()} студент ({weak.max() * 100 // total}%) '
            f'бұл тақырыпты білмейді — оны қайта түсіндіру ұсынылады.</span></div>')
    t1, t2 = st.tabs(["Әлсіз тақырыптар", "Жылу картасы"])
    with t1:
        rows = "".join(
            f'<div class="row"><div><b>{t}</b> &nbsp;{h.escape(topic_name[t])}</div>'
            f'<div class="track"><div class="fill" style="width:{weak[t] / total * 100:.0f}%;animation-delay:{0.1 + i * 0.08:.2f}s"></div></div>'
            f'<div style="text-align:right;font-weight:700">{weak[t]}</div></div>'
            for i, t in enumerate(weak.sort_values(ascending=False).index))
        st.html(f'<div class="card"><h3 style="margin:0 0 6px;font-size:18px">'
                f'Қанша студент тақырыпты білмейді</h3><div class="bars">{rows}</div></div>')
    with t2:
        cells = ['<div></div>'] + [f'<div class="hd">{t}</div>' for t in grid.columns]
        for r, (sid, row) in enumerate(grid.head(20).iterrows()):
            cells.append(f'<div>{sid}</div>')
            for c, t in enumerate(grid.columns):
                cells.append(f'<div class="c" title="{sid} · {t}: {LEVEL_NAMES[row[t]]}" '
                             f'style="background:{LV[row[t]][0]}b3;border:1px solid {LV[row[t]][0]};'
                             f'animation-delay:{(r + c) * 0.03:.2f}s"></div>')
        legend = "".join(f'<span><span class="dot" style="background:{LV[k][0]};color:{LV[k][0]}"></span>{LEVEL_NAMES[k]}</span>'
                         for k in (2, 1, 0))
        st.html(f'<div class="card"><h3 style="margin:0 0 12px;font-size:18px">'
                f'Студент × тақырып <span style="font-weight:400;color:#8b93a7;font-size:14px">(алғашқы 20 студент)</span></h3>'
                f'<div class="heat" style="grid-template-columns:64px repeat({len(grid.columns)},1fr)">{"".join(cells)}</div>'
                f'<div class="legend">{legend}</div></div>')


# ---------- 7. Демо ----------
elif page == "Демо":
    hero("Қорғауға арналған", "Үш студент — үш түрлі жағдай",
         "Тест бөлігіндегі жасанды студенттер. Агент (expectimax) олардың жазылған жауаптарымен бейімделгіш тест жүргізеді.")

    @st.cache_data
    def demo_students():
        truth = pd.read_csv("data/true_levels.csv")
        answers = pd.read_csv("data/students.csv")
        ids = list(pd.read_csv("data/test_students.csv")["student_id"])
        tl = truth[truth["student_id"].isin(ids)].pivot(index="student_id", columns="topic_id", values="true_level")
        guess = answers[answers["student_id"].isin(ids)].groupby("student_id")["confident"].mean()
        strong = tl.mean(axis=1).idxmax()
        gap = tl[(tl["T5"] == 0) & (tl["T1"] == 2)]
        base_gap = gap.index[0] if len(gap) else tl.mean(axis=1).idxmin()
        guesser = guess.drop([strong, base_gap], errors="ignore").idxmin()
        return {"Күшті студент": strong, "Санау жүйелерінде олқылық": base_gap, "Көп болжаған студент": guesser}

    @st.cache_data
    def demo_run(sid):
        answers = pd.read_csv("data/students.csv")
        part = answers[answers["student_id"] == sid]
        lookup = {r.q_id: (r.correct, r.confident) for r in part.itertuples()}
        _, ans, hist, log = run_adaptive(net, test_q, lambda q: lookup[q["q_id"]], "expectimax", q_table, seed=7)
        return ans, [list(map(float, m)) for m in hist], log

    demos = demo_students()
    choice = st.radio("Мысал", list(demos), horizontal=True, label_visibility="collapsed", key="demo")
    sid = demos[choice]
    if st.session_state.get("result", {}).get("label") != f"Демо · {sid}":
        ans, hist, log = demo_run(sid)
        set_result(ans, hist, log, f"Демо · {sid}")
        st.session_state.demo_sid = sid
    st.html(f'<span class="chip" style="animation:rise .4s ease backwards">Студент коды: {sid} · '
            f'{len(st.session_state.result["answers"])} сұрақ</span>')
    show_stats(current_p())
    result_tabs(key_prefix=f"demo_{sid}")
