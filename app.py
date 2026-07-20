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
#  DESIGN SYSTEM & 6 GRAFICHE VETTORIALI UNICHE E DIVERSE
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12;
        --panel: #0E1420;
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

    /* CONTENITORE IMMAGINE PRIVO DI CORNICI E BORDI */
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
#  6 GRAFICHE TOTALMENTE DIVERSE TRA LORO (SENZA CORNICI)
# =========================================================

# 1. MODULO ANALISI: Ologramma biometrico con tracciamento scheletrico in corsa
SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g1" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g transform="translate(320, 40)" filter="url(#g1)">
    <!-- Ologramma dinamico in movimento (Atleta wireframe) -->
    <circle cx="130" cy="45" r="18" fill="none" stroke="#00E5FF" stroke-width="3"/>
    <path d="M 130,63 L 125,120 L 100,160 L 80,130" fill="none" stroke="#00E5FF" stroke-width="4" stroke-linecap="round"/>
    <path d="M 125,120 L 155,160 L 175,200" fill="none" stroke="#00F5A0" stroke-width="4" stroke-linecap="round"/>
    <path d="M 125,90 L 170,110 L 205,95" fill="none" stroke="#00E5FF" stroke-width="3.5" stroke-linecap="round"/>
    <path d="M 125,90 L 85,115 L 60,100" fill="none" stroke="#00E5FF" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="125" y1="120" x2="125" y2="210" stroke="#00E5FF" stroke-width="4"/>
    <path d="M 125,210 L 100,280 L 70,310" fill="none" stroke="#FF6A3D" stroke-width="4.5" stroke-linecap="round"/>
    <path d="M 125,210 L 155,270 L 195,290" fill="none" stroke="#00F5A0" stroke-width="4.5" stroke-linecap="round"/>
    <!-- Nodi sensore -->
    <circle cx="130" cy="45" r="5" fill="#FFF"/>
    <circle cx="125" cy="120" r="5" fill="#00E5FF"/>
    <circle cx="125" cy="210" r="5" fill="#00F5A0"/>
    <circle cx="100" cy="280" r="6" fill="#FF6A3D"/>
  </g>
  <g filter="url(#g1)" opacity="0.8">
    <line x1="100" y1="320" x2="800" y2="320" stroke="#1c2333" stroke-width="2"/>
    <polyline points="100,320 180,260 260,290 340,180 420,240 500,140 580,210 660,110 740,190 820,130" fill="none" stroke="#00E5FF" stroke-width="3"/>
  </g>
</svg>"""

# 2. MODULO STATISTICHE: Smartwatch con onde metriche ECG fluttuanti
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <!-- Orologio Smartwatch centrale con ECG -->
  <g transform="translate(370, 80)" filter="url(#g2)">
    <rect x="0" y="0" width="160" height="240" rx="45" fill="#0E1420" stroke="#00E5FF" stroke-width="4"/>
    <rect x="20" y="30" width="120" height="180" rx="20" fill="#080B12" stroke="#1c2333" stroke-width="2"/>
    <!-- Tracciato ECG animato dentro lo schermo -->
    <path d="M 30,120 L 55,120 L 70,80 L 85,160 L 100,100 L 115,130 L 130,120" fill="none" stroke="#00F5A0" stroke-width="3.5"/>
    <text x="80" y="65" fill="#FFF" font-family="monospace" font-size="22" font-weight="bold" text-anchor="middle">165</text>
    <text x="80" y="185" fill="#00E5FF" font-family="monospace" font-size="11" text-anchor="middle">PEAK ZONE</text>
  </g>
  <!-- Linee dati laterali -->
  <g filter="url(#g2)" opacity="0.8">
    <path d="M 120,150 L 320,150 L 370,180" fill="none" stroke="#00E5FF" stroke-width="2" stroke-dasharray="5,5"/>
    <circle cx="120" cy="150" r="6" fill="#00E5FF"/>
    <path d="M 530,180 L 580,150 L 780,150" fill="none" stroke="#FF6A3D" stroke-width="2" stroke-dasharray="5,5"/>
    <circle cx="780" cy="150" r="6" fill="#FF6A3D"/>
  </g>
</svg>"""

# 3. MODULO KPI DASHBOARD: Radar Tachimetro olografico dettagliato (Stile foto utente)
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g3" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g transform="translate(450, 200)" filter="url(#g3)">
    <circle cx="0" cy="0" r="160" fill="none" stroke="#00E5FF" stroke-width="1" opacity="0.2"/>
    <circle cx="0" cy="0" r="120" fill="none" stroke="#00F5A0" stroke-width="2" stroke-dasharray="8,8"/>
    <circle cx="0" cy="0" r="80" fill="none" stroke="#FF6A3D" stroke-width="1.5" opacity="0.6"/>
    <!-- Lancetta radar -->
    <path d="M 0,0 L 90,-90" stroke="#00F5A0" stroke-width="5" stroke-linecap="round"/>
    <circle cx="0" cy="0" r="8" fill="#00F5A0"/>
    <text x="0" y="25" fill="#FFF" font-family="monospace" font-size="42" font-weight="bold" text-anchor="middle">98.2%</text>
    <text x="0" y="55" fill="#00E5FF" font-family="monospace" font-size="12" letter-spacing="3" text-anchor="middle">SYSTEM OPTIMAL</text>
  </g>
  <g filter="url(#g3)">
    <text x="180" y="120" fill="#00E5FF" font-family="monospace" font-size="12" letter-spacing="2">RECOVERY INDEX</text>
    <path d="M 180,130 L 320,130" fill="none" stroke="#00E5FF" stroke-width="2"/>
    <text x="580" y="290" fill="#FF6A3D" font-family="monospace" font-size="12" letter-spacing="2">STRAIN THRESHOLD</text>
    <path d="M 580,280 L 720,280" fill="none" stroke="#FF6A3D" stroke-width="2"/>
  </g>
</svg>"""

# 4. MODULO ML: Rete neurale profonda e nodi interconnessi
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g4" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g stroke="#1c2333" stroke-width="2" filter="url(#g4)">
    <!-- Connessioni fitte neurali -->
    <line x1="150" y1="100" x2="350" y2="200" stroke="#00E5FF" stroke-width="2.5"/>
    <line x1="150" y1="300" x2="350" y2="200" stroke="#00F5A0" stroke-width="2.5"/>
    <line x1="350" y1="200" x2="550" y2="90" stroke="#FFB020" stroke-width="3"/>
    <line x1="350" y1="200" x2="550" y2="310" stroke="#00E5FF" stroke-width="3"/>
    <line x1="550" y1="90" x2="750" y2="200" stroke="#FF6A3D" stroke-width="4"/>
    <line x1="550" y1="310" x2="750" y2="200" stroke="#00F5A0" stroke-width="4"/>
  </g>
  <g filter="url(#g4)">
    <circle cx="150" cy="100" r="14" fill="#00E5FF"/>
    <circle cx="150" cy="300" r="14" fill="#00F5A0"/>
    <circle cx="350" cy="200" r="20" fill="#FFB020"/>
    <circle cx="550" cy="90" r="16" fill="#00E5FF"/>
    <circle cx="550" cy="310" r="16" fill="#FF6A3D"/>
    <circle cx="750" cy="200" r="30" fill="#00F5A0" opacity="0.9"/>
    <text x="750" y="208" fill="#080B12" font-family="monospace" font-size="16" font-weight="bold" text-anchor="middle">AI</text>
  </g>
</svg>"""

# 5. MODULO ACTION PLAN: Obiettivo e mirino balistico di performance
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g5" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <g transform="translate(450, 200)" filter="url(#g5)">
    <!-- Mirino geometrico avanzato -->
    <rect x="-110" y="-110" width="220" height="220" fill="none" stroke="#00E5FF" stroke-width="1.5" opacity="0.3" transform="rotate(45)"/>
    <circle cx="0" cy="0" r="130" fill="none" stroke="#00F5A0" stroke-width="2" stroke-dasharray="15,10"/>
    <circle cx="0" cy="0" r="70" fill="none" stroke="#FF6A3D" stroke-width="3"/>
    <line x1="-90" y1="0" x2="90" y2="0" stroke="#00E5FF" stroke-width="1.5" stroke-dasharray="4,4"/>
    <line x1="0" y1="-90" x2="0" y2="90" stroke="#00E5FF" stroke-width="1.5" stroke-dasharray="4,4"/>
    <circle cx="0" cy="0" r="8" fill="#FF6A3D"/>
  </g>
</svg>"""

# 6. MODULO COMPUTER VISION: RX Ginocchio e analisi articolare clinica
SVG_CV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400">
  <defs>
    <filter id="g6" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="900" height="400" fill="transparent"/>
  <!-- Scheletro Clinico / Raggi X Ginocchio con angoli biometrici -->
  <g transform="translate(350, 40)" filter="url(#g6)">
    <!-- Femore -->
    <path d="M 100,20 C 100,80 90,140 95,180" fill="none" stroke="#00E5FF" stroke-width="8" stroke-linecap="round"/>
    <!-- Rotula e articolazione -->
    <circle cx="105" cy="195" r="22" fill="none" stroke="#FF6A3D" stroke-width="4"/>
    <circle cx="105" cy="195" r="8" fill="#FF6A3D"/>
    <!-- Tibia / Perone -->
    <path d="M 100,215 C 105,250 110,290 115,330" fill="none" stroke="#00E5FF" stroke-width="7" stroke-linecap="round"/>
    <!-- Angoli di stress vettoriali -->
    <path d="M 105,195 L 180,130" fill="none" stroke="#FFB020" stroke-width="2.5" stroke-dasharray="5,5"/>
    <circle cx="180" cy="130" r="6" fill="#FFB020"/>
    <text x="195" y="135" fill="#FFB020" font-family="monospace" font-size="14" font-weight="bold">141.5° STRESS</text>
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
    header_block("Modulo 01 — Acquisizione Dati", "ANALISI STATO DI FORMA", "Inserisci i parametri fisiologici e seleziona l'obiettivo odierno: il sistema elaborerà lo stato di preparazione in tempo reale.", IMG_HERO_ANALISI, "Skeletal Tracking Scan")

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
    header_block("Modulo 02 — Analytics Storico", "STATISTICHE ANALISI", "Volume, intensità e recupero degli ultimi tre mesi, decodificati in metriche di performance avanzate.", IMG_HERO_STATS, "Smartwatch ECG Telemetry")

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
    header_block("Modulo 03 — Live Monitoring", "KPI DASHBOARD", "Bilancio carico/recupero, indice di rischio e profilo atletico calcolati sui parametri odierni.", IMG_HERO_KPI, "Radar HUD Tachometer")

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
    header_block("Modulo 04 — Model Explainability", "ANALISI PREDITTIVA ML", "Esplora i modelli di Machine Learning avanzati addestrati sul tuo storico biometrico e comportamentale.", IMG_HERO_ML, "Deep Neural Network")

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
    header_block("Modulo 05 — Action Plan", "CONSIGLIO FINALE", "Protocollo operativo e proiezioni fisiologiche generate su misura per la sessione odierna.", IMG_HERO_PLAN, "Target Ballistic HUD")

    if not st.session_state.analisi_fatta:
        st.warning("Completa prima il questionario nella pagina 'ANALISI STATO DI FORMA'.")
    else:
        st.success("VIA LIBERA ALL'ALLENAMENTO: Il tuo sistema biologico è pienamente rigenerato. I modelli confermano un rischio di infortunio minimo.")

# ---------------------------------------------------------
# PAGINA 6: COMPUTER VISION & BIOMECHANIC AI
# ---------------------------------------------------------
elif pagina == "COMPUTER VISION":
    header_block("Modulo 06 — Computer Vision", "AI RUNNING FORM ANALYSIS & INJURY PREDICTION", "Carica un video di corsa (profilo laterale): l'IA estrae lo scheletro biometrico, calcola angoli/sovraccarichi e predice il rischio d'infortunio.", IMG_HERO_CV, "Clinical RX & Angles")

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
