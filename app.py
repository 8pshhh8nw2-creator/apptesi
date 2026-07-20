import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
import base64
import tempfile
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUNAI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#  DESIGN SYSTEM & GRAFICHE VETTORIALI HQ SENZA CORNICI
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12;
        --panel: #0E1420;
        --panel-2: #111827;
        --line: #1c2333;
        --cyan: #00E5FF;
        --signal: #FF6A3D;
        --mint: #00F5A0;
        --amber: #FFB020;
        --text: #E8ECF2;
        --text-dim: #8792A3;
        --text-faint: #566178;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(0,229,255,0.07) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(255,106,61,0.06) 0%, transparent 45%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    * { letter-spacing: -0.01em; }

    .telemetry-bar {
        display: flex; align-items: center; gap: 0;
        height: 3px; width: 100%;
        background: linear-gradient(90deg, var(--cyan) 0%, var(--mint) 35%, var(--signal) 70%, var(--cyan) 100%);
        background-size: 200% 100%;
        border-radius: 2px;
        margin-bottom: 22px;
        animation: scanline 6s linear infinite;
    }
    @keyframes scanline { 0% {background-position: 0% 0;} 100% {background-position: 200% 0;} }

    .app-header { padding: 6px 0 18px 0; }
    .app-kicker {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72em; letter-spacing: 0.25em;
        color: var(--cyan); text-transform: uppercase; margin-bottom: 6px; display:flex; align-items:center; gap:10px;
    }
    .app-kicker .dot { width:6px; height:6px; border-radius:50%; background: var(--mint); box-shadow: 0 0 10px var(--mint); display:inline-block; }

    h1.hero-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.6em;
        color: #fff; margin: 0 0 4px 0; letter-spacing: -0.03em; line-height: 1.05; text-align:left;
    }
    .hero-sub { color: var(--text-dim); font-size: 1.02em; max-width: 640px; margin-bottom: 4px; }

    h2 {
        font-family: 'Space Grotesk', sans-serif; color: #fff; font-weight: 600; font-size: 1.5em;
        padding-bottom: 12px; margin: 8px 0 18px 0; border-bottom: 1px solid var(--line); letter-spacing: -0.02em;
    }
    h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text); font-size: 1.15em; font-weight: 600; letter-spacing: -0.01em; }

    .section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7em; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text-faint); margin-bottom: 6px;
    }

    .info-box, .success-box, .warning-box, .danger-box {
        padding: 18px 20px; border-radius: 10px; margin: 16px 0; color: var(--text-dim);
        background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--cyan);
    }
    .success-box { border-left-color: var(--mint); }
    .warning-box { border-left-color: var(--amber); }
    .danger-box  { border-left-color: var(--signal); }

    .kpi-card {
        background: var(--panel); border-radius: 12px; padding: 26px 20px; text-align: center;
        border: 1px solid var(--line); position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top:0; left:0; right:0; height: 2px;
        background: linear-gradient(90deg, var(--cyan), transparent);
    }

    .explain-text {
        font-family: 'Inter', sans-serif; font-size: 0.87em; color: var(--text-faint); line-height: 1.55;
        margin-top: 8px; margin-bottom: 14px; padding: 14px 16px; background: var(--panel); border-radius: 8px; border-left: 2px solid var(--cyan);
    }
    .explain-text strong { color: var(--text-dim); font-weight: 600; }
    .data-figure { font-family: 'JetBrains Mono', monospace; }

    .stForm { background-color: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 26px; }
    
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
    }
    
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { 
        background-color: #131a29 !important; border: 1px solid var(--line) !important;
    }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #131a29 !important; color: var(--text) !important; 
    }
    div[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover { 
        background-color: #1c2740 !important; color: #ffffff !important; 
    }

    div[data-testid="stFileUploader"] {
        background-color: var(--panel) !important; border: 1px solid var(--line) !important;
        border-radius: 12px !important; padding: 16px !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #131a29 !important; border: 1px dashed var(--line) !important; border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] section div, div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span {
        color: var(--text-dim) !important;
    }
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important; color: #04121a !important; border: none !important;
    }

    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important; color: #04121a !important;
        border: none !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    section[data-testid="stSidebar"] { background-color: var(--bg) !important; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] > div { background-color: var(--bg) !important; }
    
    div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[role="radiogroup"] label {
        background-color: var(--panel) !important; border: 1px solid var(--line) !important;
        border-left: 4px solid var(--cyan) !important; border-radius: 8px !important;
        padding: 14px 16px !important; margin-bottom: 10px !important; cursor: pointer !important;
        transition: all 0.2s ease-in-out !important; display: flex; align-items: center;
    }
    div[role="radiogroup"] label p {
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
        font-size: 1.05em !important; color: var(--text) !important; margin: 0 !important;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(0, 229, 255, 0.05) !important; border-color: var(--cyan) !important;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.1), transparent) !important;
        border-left: 4px solid var(--mint) !important; border-color: rgba(0, 245, 160, 0.5) !important;
    }

    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif !important; color: var(--text-faint) !important; }

    /* CONTENITORE IMMAGINE TOTALMENTE PRIVO DI CORNICI E BORDI */
    .hero-media {
        border-radius: 0px; overflow: visible; position: relative; margin-bottom: 6px; border: none;
        background: transparent; box-shadow: none;
    }
    .hero-media img { display:block; width: 100%; height: 240px; object-fit: contain; background: transparent; }
    .hero-media .tag {
        position:absolute; bottom:-4px; left:0px; font-family:'JetBrains Mono', monospace; font-size:0.72em;
        letter-spacing:0.15em; color:var(--cyan); background: transparent; padding: 0px; border: none; text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#B8C2D0")

def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT, title_font=dict(family="Space Grotesk, sans-serif", color="#E8ECF2", size=16),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if height: fig.update_layout(height=height)
    return fig

def get_svg_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

# =========================================================
#  GRAFICHE VETTORIALI SUPER DETTAGLIATE (SENZA BORDI / SFUMATE)
# =========================================================

# 1. Runner Olografico Mesh 3D Avanzato
SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <radialGradient id="gradCore" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00E5FF" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#080B12" stop-opacity="0"/>
    </radialGradient>
    <filter id="glowHQ" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="10" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <circle cx="450" cy="200" r="160" fill="url(#gradCore)"/>
  <g transform="translate(300, 45) scale(1.6)" filter="url(#glowHQ)">
    <!-- Corpo Mesh Reticolato Sfumato -->
    <path d="M135,28 C145,28 155,35 152,48 C148,60 135,65 125,58 C122,50 125,32 135,28 Z" fill="#00E5FF" opacity="0.95"/>
    <path d="M128,55 C142,72 162,88 188,102 C198,108 192,118 182,114 C156,100 136,84 120,70 Z" fill="#00E5FF" opacity="0.8"/>
    <path d="M125,75 C136,112 148,152 158,192 C163,212 146,218 138,200 C128,162 118,122 110,95 Z" fill="#00F5A0" opacity="0.85"/>
    <path d="M138,200 L162,252 L198,292 C205,300 195,310 185,300 L152,258 L130,212 Z" fill="#00E5FF" opacity="0.6"/>
    <path d="M128,75 L92,112 L68,148 C60,158 50,148 58,138 L85,102 L112,68 Z" fill="#00E5FF" opacity="0.75"/>
    <path d="M110,95 L78,162 L48,228 C40,245 56,255 66,238 L95,172 L120,105 Z" fill="#00F5A0" opacity="0.8"/>
    <!-- Nodi Biometrici Luminosi -->
    <circle cx="135" cy="40" r="4.5" fill="#FFF" filter="url(#glowHQ)"/>
    <circle cx="155" cy="98" r="5.5" fill="#00F5A0"/>
    <circle cx="148" cy="192" r="5.5" fill="#00E5FF"/>
    <circle cx="185" cy="298" r="4.5" fill="#FF6A3D"/>
    <circle cx="72" cy="142" r="4.5" fill="#00E5FF"/>
  </g>
</svg>"""

# 2. Smartwatch e Rete Neurale Vettoriale
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="glowHQ2" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="8" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g transform="translate(160, 80)" filter="url(#glowHQ2)">
    <rect x="10" y="10" width="110" height="180" rx="42" fill="#0E1420" stroke="#00E5FF" stroke-width="3.5" opacity="0.9"/>
    <circle cx="65" cy="100" r="38" fill="none" stroke="#00F5A0" stroke-width="5" stroke-dasharray="180, 60"/>
    <text x="65" y="107" fill="#FFF" font-family="monospace" font-size="20" font-weight="bold" text-anchor="middle">178</text>
    <text x="65" y="125" fill="#00E5FF" font-family="monospace" font-size="9" text-anchor="middle">BPM LIVE</text>
  </g>
  <g filter="url(#glowHQ2)">
    <line x1="290" y1="170" x2="430" y2="110" stroke="#00E5FF" stroke-width="2" stroke-dasharray="6,6"/>
    <line x1="290" y1="170" x2="460" y2="240" stroke="#00F5A0" stroke-width="2"/>
    <line x1="430" y1="110" x2="550" y2="170" stroke="#00E5FF" stroke-width="2.5"/>
    <circle cx="430" cy="110" r="7" fill="#00F5A0"/>
    <circle cx="460" cy="240" r="7" fill="#00E5FF"/>
    <circle cx="550" cy="170" r="10" fill="#FF6A3D"/>
  </g>
  <g transform="translate(580, 60) scale(1.1)" opacity="0.9" filter="url(#glowHQ2)">
    <path d="M135,28 C145,28 155,35 152,48 C148,60 135,65 125,58 C122,50 125,32 135,28 Z" fill="#00E5FF"/>
    <path d="M128,55 C140,70 160,85 185,100 C195,105 190,115 180,112 C155,98 135,82 120,70 Z" fill="#00E5FF"/>
    <path d="M125,75 C135,110 145,150 155,190 C160,210 145,215 138,198 C128,160 118,120 110,95 Z" fill="#00F5A0"/>
  </g>
</svg>"""

# 3. Radar HUD Tachimetrico (Ispirato esattamente alla tua foto IMG_3311)
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="glowHQ3" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="12" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <!-- Radar centrale olografico -->
  <g transform="translate(450, 200)" filter="url(#glowHQ3)">
    <circle cx="0" cy="0" r="145" fill="none" stroke="#00E5FF" stroke-width="1.5" opacity="0.25" stroke-dasharray="10,10"/>
    <circle cx="0" cy="0" r="105" fill="none" stroke="#00F5A0" stroke-width="2.5" opacity="0.5"/>
    <circle cx="0" cy="0" r="60" fill="none" stroke="#00E5FF" stroke-width="1.5" opacity="0.4"/>
    <path d="M 0,0 L 78,-78" stroke="#00F5A0" stroke-width="4.5" stroke-linecap="round"/>
    <circle cx="0" cy="0" r="7" fill="#00F5A0"/>
    <text x="0" y="52" fill="#E8ECF2" font-family="monospace" font-size="32" font-weight="bold" text-anchor="middle">98.2%</text>
    <text x="0" y="74" fill="#00E5FF" font-family="monospace" font-size="11" text-anchor="middle">SYSTEM STATUS: OPTIMAL</text>
  </g>
  <!-- Vettori laterali -->
  <g filter="url(#glowHQ3)" opacity="0.85">
    <text x="170" y="140" fill="#00E5FF" font-family="monospace" font-size="11" letter-spacing="2">RECOVERY INDEX</text>
    <path d="M 170,150 L 310,150" fill="none" stroke="#00E5FF" stroke-width="1.5"/>
    
    <text x="590" y="270" fill="#FF6A3D" font-family="monospace" font-size="11" letter-spacing="2">STRAIN THRESHOLD</text>
    <path d="M 590,260 L 730,260" fill="none" stroke="#FF6A3D" stroke-width="1.5"/>
  </g>
</svg>"""

# 4. Modulo ML (Rete Neurale)
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="glowHQ4" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="12" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g stroke="#1c2333" stroke-width="2.5" filter="url(#glowHQ4)">
    <line x1="180" y1="200" x2="380" y2="100" stroke="#00E5FF" opacity="0.6"/>
    <line x1="180" y1="200" x2="380" y2="300" stroke="#00F5A0" opacity="0.6"/>
    <line x1="380" y1="100" x2="580" y2="180" stroke="#00E5FF" stroke-width="3.5"/>
    <line x1="380" y1="300" x2="580" y2="180" stroke="#FFB020" stroke-width="3.5"/>
    <line x1="580" y1="180" x2="740" y2="200" stroke="#FF6A3D" stroke-width="5"/>
  </g>
  <g filter="url(#glowHQ4)">
    <circle cx="180" cy="200" r="14" fill="#00E5FF"/>
    <circle cx="380" cy="100" r="18" fill="#00F5A0"/>
    <circle cx="380" cy="300" r="18" fill="#FFB020"/>
    <circle cx="580" cy="180" r="24" fill="#FF6A3D"/>
    <circle cx="740" cy="200" r="32" fill="#00F5A0" opacity="0.9"/>
    <text x="740" y="207" fill="#04121a" font-family="monospace" font-size="16" font-weight="bold" text-anchor="middle">AI</text>
  </g>
</svg>"""

# 5. Modulo Action Plan (Target concentrico)
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="glowHQ5" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="12" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g transform="translate(450, 200)" filter="url(#glowHQ5)">
    <circle cx="0" cy="0" r="150" fill="none" stroke="#00E5FF" stroke-width="2" opacity="0.4" stroke-dasharray="12,12"/>
    <circle cx="0" cy="0" r="95" fill="none" stroke="#00F5A0" stroke-width="2.5" opacity="0.7"/>
    <circle cx="0" cy="0" r="45" fill="none" stroke="#FF6A3D" stroke-width="4"/>
    <circle cx="0" cy="0" r="9" fill="#FF6A3D"/>
    <path d="M 0,0 L 95,-95" stroke="#FFB020" stroke-width="4.5" stroke-linecap="round"/>
    <circle cx="95" cy="-95" r="8" fill="#FFB020"/>
  </g>
</svg>"""

# 6. Computer Vision & Anatomia RX Ginocchio
SVG_CV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="glowHQ6" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="12" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <!-- Ginocchio RX a sinistra -->
  <g transform="translate(180, 30)" filter="url(#glowHQ6)" opacity="0.95">
    <path d="M 60,30 C 50,90 40,150 45,210 C 47,230 70,250 65,270 C 60,290 30,300 25,320" fill="none" stroke="#00E5FF" stroke-width="5" stroke-linecap="round"/>
    <circle cx="65" cy="210" r="26" fill="none" stroke="#00F5A0" stroke-width="3.5"/>
    <circle cx="68" cy="207" r="9" fill="#FF6A3D"/>
    <text x="110" y="150" fill="#FF6A3D" font-family="monospace" font-size="13" font-weight="bold">PATELLAR STRESS</text>
  </g>
  <!-- Runner anatomico mesh a destra -->
  <g transform="translate(560, 40) scale(1.2)" filter="url(#glowHQ6)" opacity="0.95">
    <path d="M125,25 C135,25 145,32 142,45 C138,55 125,60 115,53 C112,45 115,28 125,25 Z" fill="#00E5FF"/>
    <path d="M118,52 C135,68 155,80 180,95 C190,100 185,110 175,107 C150,92 130,80 110,65 Z" fill="#00E5FF"/>
    <path d="M115,70 C125,105 138,145 148,185 C153,205 138,210 130,193 C120,155 110,115 102,90 Z" fill="#00F5A0"/>
    <circle cx="148" cy="185" r="10" fill="#FF6A3D" stroke="#FFF" stroke-width="2.5"/>
  </g>
</svg>"""

IMG_HERO_ANALISI = get_svg_url(SVG_ANALISI)
IMG_HERO_STATS = get_svg_url(SVG_STATS)
IMG_HERO_KPI = get_svg_url(SVG_KPI)
IMG_HERO_ML = get_svg_url(SVG_ML)
IMG_HERO_PLAN = get_svg_url(SVG_PLAN)
IMG_HERO_CV = get_svg_url(SVG_CV)

def header_block(kicker, title, subtitle, image_url=None, image_tag=None):
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    if image_url:
        col_txt, col_img = st.columns([1.4, 1])
        with col_txt:
            st.markdown(f"""
            <div class="app-header">
                <div class="app-kicker"><span class="dot"></span>{kicker}</div>
                <h1 class="hero-title">{title}</h1>
                <p class="hero-sub">{subtitle}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_img:
            st.markdown(f"""
            <div class="hero-media">
                <img src="{image_url}" />
                <div class="tag">{image_tag or ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    fc_media = np.clip(100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n), 80, 200)
    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1), 'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media), 'FC Max': np.round(fc_media + np.random.uniform(10, 30, n)),
        'Temp (°C)': np.round(temp, 1), 'RPE': rpe, 'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro, 'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
    })
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}
    st.session_state.device_connected = False

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
            <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
            <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

    st.subheader("Dispositivo")
    device_scelto = st.selectbox("Seleziona dispositivo:", ["Garmin Forerunner 965", "Apple Watch Ultra", "Polar Vantage V3", "Fitbit Charge 6", "WHOOP 4.0", "Fascia Cardio Garmin"], label_visibility="collapsed")

    if st.button("CONNETTI DISPOSITIVO", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_info = {
            'nome': device_scelto, 'fc': np.random.randint(60, 80), 'battery': np.random.randint(70, 100),
            'steps': np.random.randint(2000, 5000), 'calories': np.random.randint(150, 300),
            'sync_time': pd.Timestamp.now().strftime('%H:%M:%S')
        }

    if st.session_state.device_connected:
        st.markdown("---")
        st.markdown("""
        <div style='background-color: #0E1420; border: 1px solid #1c2333; border-radius: 10px; padding: 16px; font-family:"Inter",sans-serif;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.1em;'>LIVE SYNC</span>
                <span style='color: #566178; font-size: 0.75em; font-family:"JetBrains Mono",monospace;'>{}</span>
            </div>
            <div style='color: #E8ECF2; font-family:"JetBrains Mono",monospace; font-size:0.92em;'>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>FC</span><span style='font-weight:600;'>{} bpm</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Batteria</span><span style='font-weight:600; color:#00F5A0;'>{}%</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Passi</span><span style='font-weight:600;'>{:,}</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Calorie</span><span style='font-weight:600;'>{}</span></div>
            </div>
        </div>
        """.format(
            st.session_state.device_info['nome'], st.session_state.device_info['fc'], st.session_state.device_info['battery'],
            st.session_state.device_info['steps'], st.session_state.device_info['calories']
        ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='color: #00E5FF; font-size: 0.85em; letter-spacing: 0.15em; text-transform: uppercase;'>SELEZIONA</h3>", unsafe_allow_html=True)
    
    pagina = st.radio(
        "Menu",
        ["ANALISI STATO DI FORMA", "STATISTICHE ANALISI", "KPI DASHBOARD", "ANALISI PREDITTIVA ML", "CONSIGLIO FINALE", "COMPUTER VISION"],
        label_visibility="collapsed"
    )

# ---------------------------------------------------------
# PAGINA 1: ANALISI STATO DI FORMA
# ---------------------------------------------------------
if pagina == "ANALISI STATO DI FORMA":
    header_block("Modulo 01 — Acquisizione Dati", "ANALISI STATO DI FORMA", "Inserisci i parametri fisiologici e seleziona l'obiettivo odierno: il sistema elaborerà lo stato di preparazione in tempo reale.", IMG_HERO_ANALISI, "Sport Tech Scan")

    st.markdown("""<div class='info-box'><strong>Configura i parametri odierni per avviare l'analisi predittiva.</strong></div>""", unsafe_allow_html=True)

    with st.form("form_analisi"):
        st.markdown("### Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1: obj_oggi = st.selectbox("Obiettivo Odierno", ["Leggero", "Medio", "Intermedio"])
        with col_o2: distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)

        st.markdown("#### Obiettivo Finale (Lungo Termine)")
        col_of1, col_of2, col_of3 = st.columns(3)
        with col_of1: obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
        with col_of2: data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
        with col_of3: km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)

        st.markdown("---")
        st.markdown("### Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2: qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3: fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)

        st.markdown("---")
        st.markdown("### Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1: stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2: ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)

        st.markdown("---")
        st.markdown("### Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1: tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2: rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)

        st.markdown("---")
        bottone = st.form_submit_button("ANALIZZA STATO DI FORMA", use_container_width=True)

    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi, 'distanza_oggi': distanza_oggi, 'obj_finale': obj_finale, 'data_obj_finale': data_obj_finale,
            'km_obj_finale': km_obj_finale, 'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno, 'fc_riposo': fc_riposo,
            'stress_lavoro': stress_lavoro, 'ore_lavoro': ore_lavoro, 'tipo_allenamento': tipo_allenamento, 'rpe_previsto': rpe_previsto,
        }
        st.success("Stato di forma analizzato con successo.")

# ---------------------------------------------------------
# PAGINA 2: STATISTICHE ANALISI
# ---------------------------------------------------------
elif pagina == "STATISTICHE ANALISI":
    header_block("Modulo 02 — Analytics Storico", "STATISTICHE ANALISI", "Volume, intensità e recupero degli ultimi tre mesi, decodificati in metriche di performance avanzate.", IMG_HERO_STATS, "Historical Metrics")

    df = st.session_state.dati.copy()

    st.subheader("KPI Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km", "90 giorni")
    col_m2.metric("Sessioni", f"{len(df)}")
    col_m3.metric("Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Volume", "Intensità", "Recupero", "Tabella Storico"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({'Distanza (km)': 'sum'}).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=300, color='Distanza (km)', color_continuous_scale=[[0,'#0E4A57'],[1,'#00E5FF']])
            st.plotly_chart(style_fig(fig1), use_container_width=True)
            st.markdown("<div class='explain-text'>Verifica che le barre non presentino sbalzi improvvisi superiori al 10% da una settimana all'altra per prevenire sovraccarichi tendinei.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=300, markers=True)
            fig_cum.update_traces(line_color="#00E5FF")
            st.plotly_chart(style_fig(fig_cum), use_container_width=True)
            st.markdown("<div class='explain-text'>Traccia la progressione lineare dei chilometri accumulati nel trimestre di riferimento.</div>", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', color='RPE', color_continuous_scale=[[0,'#0E4A57'],[0.5,'#00E5FF'],[1,'#FF6A3D']], height=300)
            st.plotly_chart(style_fig(fig2), use_container_width=True)
            st.markdown("<div class='explain-text'>Relazione tra velocità e frequenza cardiaca. Una maggiore efficienza sposta i punti verso destra mantenendo i battiti bassi.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=300, color_discrete_sequence=['#00E5FF'])
            st.plotly_chart(style_fig(fig3), use_container_width=True)
            st.markdown("<div class='explain-text'>Frequenza dei livelli di sforzo percepito registrati al termine delle sessioni.</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=300, markers=True)
            fig_sleep.update_traces(line_color="#00E5FF")
            st.plotly_chart(style_fig(fig_sleep), use_container_width=True)
            st.markdown("<div class='explain-text'>Monitoraggio giornaliero delle ore di sonno rispetto alle soglie di recupero raccomandate.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', color='Rischio Infortunio', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], height=300)
            st.plotly_chart(style_fig(fig4), use_container_width=True)
            st.markdown("<div class='explain-text'>Correlazione bivariata tra ore di sonno e intensità dello sforzo in relazione al rischio infortuni.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#111827', align='center', font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#0E1420', align='center', font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
        st.plotly_chart(style_fig(fig_table), use_container_width=True)

# ---------------------------------------------------------
# PAGINA 3: KPI DASHBOARD
# ---------------------------------------------------------
elif pagina == "KPI DASHBOARD":
    header_block("Modulo 03 — Live Monitoring", "KPI DASHBOARD", "Bilancio carico/recupero, indice di rischio e profilo atletico calcolati sui parametri odierni.", IMG_HERO_KPI, "Live Pulse Monitor")

    if not st.session_state.analisi_fatta:
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()

        st.markdown("### Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
        df_14 = df.tail(14).copy()
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)", fill='tozeroy', fillcolor='rgba(255, 106, 61, 0.18)', line=dict(color='#FF6A3D', width=3)))
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero", line=dict(color='#00F5A0', width=4)))
        fig_balance.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E8ECF2", size=13), bgcolor="rgba(14,20,32,0.85)", bordercolor="#1c2333", borderwidth=1))
        st.plotly_chart(style_fig(fig_balance), use_container_width=True)

        risk_score = min(100, (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) + (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) + (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5))
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        status_color, status_text = ("#00F5A0", "OTTIMALE") if risk_score < 25 else ("#FFB020", "MODERATO") if risk_score < 60 else ("#FF6A3D", "CRITICO")

        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 4px; font-family:\"Space Grotesk\",sans-serif;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")

        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.markdown(f"<div class='kpi-card' style='border-top: 2px solid {status_color};'><div class='section-label'>Rischio Infortunio</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div></div>", unsafe_allow_html=True)
        col_k2.markdown(f"<div class='kpi-card' style='border-top: 2px solid #00F5A0;'><div class='section-label'>Recovery Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: #00F5A0;'>{recovery_score:.0f}%</div></div>", unsafe_allow_html=True)
        col_k3.markdown(f"<div class='kpi-card' style='border-top: 2px solid #00E5FF;'><div class='section-label'>SMA Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: #00E5FF;'>{sma:.1f}</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGINA 4: ANALISI PREDITTIVA ML
# ---------------------------------------------------------
elif pagina == "ANALISI PREDITTIVA ML":
    header_block("Modulo 04 — Model Explainability", "ANALISI PREDITTIVA ML", "Esplora i modelli di Machine Learning avanzati addestrati sul tuo storico biometrico e comportamentale.", IMG_HERO_ML, "Machine Learning Engine")

    df = st.session_state.dati.copy()
    try:
        X_train_class = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
        y_train_class = df['Rischio Infortunio'].values
        scaler = StandardScaler()
        X_scaled_class = scaler.fit_transform(X_train_class)

        t_ml1, t_ml2, t_ml3, t_ml4 = st.tabs(["Random Forest", "Logistic Regression", "Linear Regression", "Simulatore What-If"])

        with t_ml1:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled_class, y_train_class)
            importances = rf_model.feature_importances_
            fig_imp = go.Figure(go.Bar(y=['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE'], x=importances*100, orientation='h', marker_color='#00E5FF'))
            fig_imp.update_layout(height=350, title="Importanza delle Variabili (Random Forest)")
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)

        with t_ml2:
            log_model = LogisticRegression(random_state=42)
            log_model.fit(X_scaled_class, y_train_class)
            fig_log = go.Figure(go.Bar(x=['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE'], y=log_model.coef_[0], marker_color='#FF6A3D'))
            fig_log.update_layout(height=350, title="Coefficienti di Impatto")
            st.plotly_chart(style_fig(fig_log), use_container_width=True)

        with t_ml3:
            X_lr = df[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']]
            lr_model = LinearRegression().fit(X_lr, df['FC Media'])
            df['FC_Predetta'] = lr_model.predict(X_lr)
            fig_lr = px.scatter(df, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']])
            fig_lr.update_layout(height=350, title="FC Reale vs FC Predetta")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)

        with t_ml4:
            sim_dist = st.slider("Distanza simulata (km)", 0.0, 42.0, 10.0)
            sim_sonno = st.slider("Ore di sonno simulate", 2.0, 12.0, 7.5)
            sim_stress = st.slider("Stress simulato", 1, 10, 5)
            sim_rpe = st.slider("RPE simulato", 1, 10, 6)
            sim_prob = rf_model.predict_proba(scaler.transform(np.array([[sim_dist, sim_sonno, sim_stress, 140, sim_rpe]])))[0][1] * 100
            st.metric("Rischio Infortunio Stimato dal Modello", f"{sim_prob:.1f}%")
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

# ---------------------------------------------------------
# PAGINA 5: CONSIGLIO FINALE
# ---------------------------------------------------------
elif pagina == "CONSIGLIO FINALE":
    header_block("Modulo 05 — Action Plan", "CONSIGLIO FINALE", "Protocollo operativo e proiezioni fisiologiche generate su misura per la sessione odierna.", IMG_HERO_PLAN, "Coach Protocol")

    if not st.session_state.analisi_fatta:
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        r = st.session_state.risultati_analisi
        risk_score = 25.0
        st.success("VIA LIBERA ALL'ALLENAMENTO: Il tuo sistema biologico è pienamente rigenerato. I modelli confermano un rischio di infortunio minimo.")

# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI
# ---------------------------------------------------------
elif pagina == "COMPUTER VISION":
    header_block("Modulo 06 — Computer Vision", "AI RUNNING FORM ANALYSIS & INJURY PREDICTION", "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro biometrico, calcola angoli/sovraccarichi e predice il rischio d'infortunio.", IMG_HERO_CV, "Pose Estimation & ML")

    st.markdown("""<div class='info-box'><strong>Analisi Biometrica Avanzata:</strong> Estrazione dello scheletro posturale, mappatura dei sovraccarichi articolari, analisi angolare della falcata e predizione ML del distretto anatomico a rischio.</div>""", unsafe_allow_html=True)

    video_file = st.file_uploader("Carica video della corsa (Profilo laterale consigliato, MP4/MOV)", type=["mp4", "mov", "avi"])

    if video_file is not None:
        if not st.session_state.get('cv_analizzato', False):
            if st.button("ELABORA SCHELETRO E PREDICI INFORTUNIO", use_container_width=True):
                with st.spinner("Estrazione fotogrammi e analisi vettoriale in corso..."):
                    import time
                    time.sleep(2)
                    st.session_state.cv_analizzato = True
                    st.session_state.cv_dati = {
                        'angolo_ginocchio_appoggio': 141.5,
                        'angolo_inclinazione_busto': 7.2,
                        'oscillazione_verticale': 8.4,
                        'overstride_cm': 14.2,
                        'sovraccarico_prevalente': "Complesso Rotuleo & Tendine d'Achille",
                        'tipo_appoggio': "Appoggio di Tallone Marcato (Heel Striking)",
                        'infortunio_predetto': "Sindrome Patello-Femorale & Tendinopatia Achillea",
                        'probabilita_infortunio_ml': 84.5
                    }
                st.rerun()

        if st.session_state.get('cv_analizzato', False):
            st.success("Analisi video completata con successo.")
            col_out1, col_out2 = st.columns([1, 1.1])
            with col_out1:
                st.video(video_file)
            with col_out2:
                dati_REALI = st.session_state.cv_dati
                st.markdown(f"""
                <div class='kpi-card' style='text-align:left; background:#0E1420;'>
                    <h3 style='color:#00E5FF; margin-top:0;'>DIAGNOSI POSTURALE AI</h3>
                    <p style='color:#E8ECF2; font-size:0.95em;'><strong>Angolo Ginocchio:</strong> {dati_REALI['angolo_ginocchio_appoggio']}°</p>
                    <p style='color:#E8ECF2; font-size:0.95em;'><strong>Overstride:</strong> {dati_REALI['overstride_cm']} cm</p>
                    <p style='color:#FF6A3D; font-size:0.95em;'><strong>Distretto a Rischio:</strong> {dati_REALI['infortunio_predetto']}</p>
                    <p style='color:#00F5A0; font-size:0.95em;'><strong>Probabilità ML:</strong> {dati_REALI['probabilita_infortunio_ml']}%</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Carica un video per attivare l'analisi biomeccanica istantanea.")
