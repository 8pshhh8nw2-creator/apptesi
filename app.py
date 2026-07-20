import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
import base64

warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUNAI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#  DESIGN SYSTEM — RUNAI
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
            radial-gradient(circle at 15% 0%, rgba(0,229,255,0.05) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(255,106,61,0.04) 0%, transparent 45%),
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
    .app-kicker .dot { width:6px; height:6px; border-radius:50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); display:inline-block; }

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
        margin-top: 6px; padding: 14px 16px; background: var(--panel); border-radius: 8px; border-left: 2px solid var(--line);
    }
    .explain-text strong { color: var(--text-dim); font-weight: 600; }
    .data-figure { font-family: 'JetBrains Mono', monospace; }

    /* WIDGETS E FORM */
    .stForm { background-color: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 26px; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
    }
    div[data-baseweb="popover"] { background-color: #131a29 !important; }
    div[data-baseweb="popover"] ul, div[data-baseweb="menu"], ul[role="listbox"] { background-color: #131a29 !important; }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #131a29 !important; color: var(--text) !important;
    }
    div[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover { background-color: #1c2740 !important; color: #ffffff !important; }
    .stSlider label, .stSelectSlider label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color: var(--text-dim) !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    }
    .stSlider [data-baseweb="slider"] div { color: var(--text) !important; }
    div[data-testid="stTickBar"] { color: var(--text-faint) !important; }
    .stSelectSlider [role="slider"] { background-color: var(--cyan) !important; }
    div[data-testid="stWidgetLabel"] p { color: var(--text-dim) !important; }

    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important; color: #04121a !important;
        border: none !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    /* SIDEBAR & MENU */
    section[data-testid="stSidebar"] { background-color: var(--bg) !important; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] > div { background-color: var(--bg) !important; }
    section[data-testid="stSidebar"] h3 { color: var(--text-dim) !important; }
    
    div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[role="radiogroup"] label {
        background-color: var(--panel) !important; border: 1px solid var(--line) !important;
        border-left: 4px solid var(--cyan) !important; border-radius: 8px !important;
        padding: 14px 16px !important; margin-bottom: 10px !important; cursor: pointer !important;
        transition: all 0.2s ease-in-out !important; display: flex; align-items: center;
    }
    div[role="radiogroup"] label p {
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
        font-size: 1.05em !important; color: var(--text) !important; margin: 0 !important; letter-spacing: 0.02em;
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

    .hero-media {
        border-radius: 16px; overflow: hidden; position: relative; margin-bottom: 6px; border: 1px solid var(--line);
        background: var(--panel);
    }
    .hero-media img { display:block; width: 100%; height: 220px; object-fit: cover; }
    .hero-media .tag {
        position:absolute; bottom:14px; left:14px; font-family:'JetBrains Mono', monospace; font-size:0.72em;
        letter-spacing:0.12em; color:#fff; background: rgba(8,11,18,0.85); padding: 5px 10px; border-radius:6px;
        border: 1px solid rgba(255,255,255,0.15); text-transform: uppercase;
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

# SVG Generator per Immagini Sport Tech create ad hoc
def get_svg_url(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

SVG_ANALISI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><path d="M50,200 L250,200 L300,80 L350,280 L400,150 L450,250 L500,200 L850,200" stroke="#00E5FF" stroke-width="4" fill="none" opacity="0.8"/><circle cx="300" cy="80" r="6" fill="#00F5A0"/><circle cx="350" cy="280" r="6" fill="#FF6A3D"/><g opacity="0.3"><line x1="0" y1="100" x2="900" y2="100" stroke="#1c2333" stroke-width="1"/><line x1="0" y1="300" x2="900" y2="300" stroke="#1c2333" stroke-width="1"/></g><text x="60" y="360" fill="#566178" font-family="monospace" font-size="18">BIOMETRIC // SCAN</text></svg>"""
SVG_STATS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><rect x="150" y="150" width="40" height="150" fill="#00E5FF" opacity="0.3"/><rect x="250" y="200" width="40" height="100" fill="#00E5FF" opacity="0.5"/><rect x="350" y="100" width="40" height="200" fill="#00F5A0" opacity="0.8"/><rect x="450" y="220" width="40" height="80" fill="#00E5FF" opacity="0.4"/><rect x="550" y="70" width="40" height="230" fill="#FFB020" opacity="0.9"/><rect x="650" y="180" width="40" height="120" fill="#00E5FF" opacity="0.6"/><path d="M170,150 L270,200 L370,100 L470,220 L570,70 L670,180" stroke="#fff" stroke-width="3" fill="none"/><circle cx="570" cy="70" r="5" fill="#FF6A3D"/><text x="150" y="360" fill="#566178" font-family="monospace" font-size="18">HISTORICAL // METRICS</text></svg>"""
SVG_KPI = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><path d="M300,300 A 150 150 0 1 1 600,300" fill="none" stroke="#1c2333" stroke-width="20"/><path d="M300,300 A 150 150 0 0 1 500,170" fill="none" stroke="#00F5A0" stroke-width="20"/><circle cx="450" cy="270" r="10" fill="#00E5FF"/><line x1="450" y1="270" x2="520" y2="150" stroke="#00E5FF" stroke-width="4"/><text x="400" y="330" fill="#E8ECF2" font-family="monospace" font-size="28" font-weight="bold">82.4%</text><text x="70" y="360" fill="#566178" font-family="monospace" font-size="18">REALTIME // DASHBOARD</text></svg>"""
SVG_ML = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="200" cy="200" r="8" fill="#00E5FF"/><circle cx="350" cy="100" r="12" fill="#00F5A0"/><circle cx="350" cy="300" r="12" fill="#FFB020"/><circle cx="550" cy="150" r="15" fill="#FF6A3D"/><circle cx="550" cy="250" r="10" fill="#00E5FF"/><circle cx="750" cy="200" r="20" fill="#00F5A0"/><line x1="200" y1="200" x2="350" y2="100" stroke="#1c2333" stroke-width="2"/><line x1="200" y1="200" x2="350" y2="300" stroke="#1c2333" stroke-width="2"/><line x1="350" y1="100" x2="550" y2="150" stroke="#00E5FF" stroke-width="2" stroke-dasharray="5,5"/><line x1="350" y1="300" x2="550" y2="150" stroke="#1c2333" stroke-width="2"/><line x1="350" y1="300" x2="550" y2="250" stroke="#00F5A0" stroke-width="2" stroke-dasharray="5,5"/><line x1="550" y1="150" x2="750" y2="200" stroke="#FF6A3D" stroke-width="3"/><line x1="550" y1="250" x2="750" y2="200" stroke="#00E5FF" stroke-width="2"/><text x="70" y="360" fill="#566178" font-family="monospace" font-size="18">PREDICTIVE // ENGINE</text></svg>"""
SVG_PLAN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400"><rect width="900" height="400" fill="#080B12"/><circle cx="450" cy="200" r="120" fill="none" stroke="#1c2333" stroke-width="2"/><circle cx="450" cy="200" r="80" fill="none" stroke="#1c2333" stroke-width="2"/><circle cx="450" cy="200" r="40" fill="#00E5FF" opacity="0.2"/><circle cx="450" cy="200" r="10" fill="#00F5A0"/><path d="M450,200 L550,100" stroke="#FFB020" stroke-width="3"/><circle cx="550" cy="100" r="6" fill="#FFB020"/><path d="M450,200 L300,250" stroke="#FF6A3D" stroke-width="3"/><circle cx="300" cy="250" r="6" fill="#FF6A3D"/><text x="70" y="360" fill="#566178" font-family="monospace" font-size="18">ACTION // PROTOCOL</text></svg>"""

IMG_HERO_ANALISI = get_svg_url(SVG_ANALISI)
IMG_HERO_STATS = get_svg_url(SVG_STATS)
IMG_HERO_KPI = get_svg_url(SVG_KPI)
IMG_HERO_ML = get_svg_url(SVG_ML)
IMG_HERO_PLAN = get_svg_url(SVG_PLAN)

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
                <span style='color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.1em;'>&#9679; LIVE</span>
                <span style='color: #566178; font-size: 0.75em; font-family:"JetBrains Mono",monospace;'>{}</span>
            </div>
            <div style='color: #E8ECF2; font-family:"JetBrains Mono",monospace; font-size:0.92em;'>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>FC</span><span style='font-weight:600;'>{} bpm</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Batteria</span><span style='font-weight:600; color:#00F5A0;'>{}%</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Passi</span><span style='font-weight:600;'>{:,}</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Calorie</span><span style='font-weight:600;'>{}</span></div>
            </div>
            <div style='color: #566178; font-size: 0.7em; margin-top: 12px; text-align: center; font-family:"JetBrains Mono",monospace;'>SYNC {}</div>
        </div>
        """.format(
            st.session_state.device_info['nome'], st.session_state.device_info['fc'], st.session_state.device_info['battery'],
            st.session_state.device_info['steps'], st.session_state.device_info['calories'], st.session_state.device_info['sync_time']
        ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='color: #00E5FF; font-size: 0.85em; letter-spacing: 0.15em; text-transform: uppercase;'>SELEZIONA</h3>", unsafe_allow_html=True)
    
    pagina = st.radio(
        "Menu",
        ["ANALISI STATO DI FORMA", "STATISTICHE ANALISI", "KPI DASHBOARD", "ANALISI PREDITTIVA ML", "CONSIGLIO FINALE"],
        label_visibility="collapsed"
    )

# ----------------- PAGINA 1: ANALISI STATO DI FORMA -----------------
if pagina == "ANALISI STATO DI FORMA":
    header_block(
        "Modulo 01 — Acquisizione Dati",
        "ANALISI STATO DI FORMA: Pre-Session",
        "Inserisci i parametri fisiologici e di carico odierni: il motore predittivo li userà per calcolare il tuo rischio infortunio in tempo reale.",
        IMG_HERO_ANALISI, "Biometric Input"
    )

    st.markdown("""
    <div class='info-box'>
    <strong>Compila il questionario per avviare il motore di predizione.</strong>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_analisi"):
        st.markdown("### Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            # Modifica: "Obiettivo Odierno" ora è una selectbox
            obj_oggi = st.selectbox("Obiettivo Odierno", ["Leggero", "Medio", "Intermedio"])
        with col_o2:
            distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)

        st.markdown("#### Obiettivo Finale (Lungo Termine)")
        col_of1, col_of2, col_of3 = st.columns(3)
        with col_of1:
            obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
        with col_of2:
            data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
        with col_of3:
            km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)

        st.markdown("---")
        st.markdown("### Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)

        st.markdown("---")
        st.markdown("### Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)

        st.markdown("---")
        st.markdown("### Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)

        st.markdown("---")
        bottone = st.form_submit_button("ANALIZZA PARAMETRI", use_container_width=True)

    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi, 'distanza_oggi': distanza_oggi, 'obj_finale': obj_finale, 'data_obj_finale': data_obj_finale,
            'km_obj_finale': km_obj_finale, 'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno, 'fc_riposo': fc_riposo,
            'stress_lavoro': stress_lavoro, 'ore_lavoro': ore_lavoro, 'tipo_allenamento': tipo_allenamento, 'rpe_previsto': rpe_previsto,
        }
        st.success("Analisi completata e caricata nel modello.")

        if obj_finale:
            giorni_rimasti = (pd.Timestamp(data_obj_finale) - pd.Timestamp.today()).days
            st.markdown(f"""
            <div class='info-box' style='border-left-color:#00F5A0;'>
            <strong>Obiettivo Finale impostato:</strong> {obj_finale} ({km_obj_finale:.1f} km) — mancano circa <strong>{max(giorni_rimasti,0)} giorni</strong> ({pd.Timestamp(data_obj_finale).strftime('%d/%m/%Y')}).
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 2: STATISTICHE ANALISI -----------------
elif pagina == "STATISTICHE ANALISI":
    header_block(
        "Modulo 02 — Analytics Storico",
        "STATISTICHE ANALISI",
        "Volume, intensità e recupero degli ultimi tre mesi, decodificati in pattern utilizzabili.",
        IMG_HERO_STATS, "Historical Load"
    )

    df = st.session_state.dati.copy()

    st.subheader("KPI Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km", "90 giorni")
    col_m2.metric("Sessioni", f"{len(df)}")
    col_m3.metric("Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")

    st.markdown("---")
    st.subheader("Analisi Dettagliata")

    tab1, tab2, tab3, tab4 = st.tabs(["Volume", "Intensità", "Recupero", "Tabella Storico"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({'Distanza (km)': 'sum'}).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=300, color='Distanza (km)', color_continuous_scale=[[0,'#0E4A57'],[1,'#00E5FF']])
            st.plotly_chart(style_fig(fig1), use_container_width=True)
            st.markdown("<div class='explain-text'>Verifica che le barre non facciano salti maggiori del 10% da una settimana all'altra. Un picco improvviso porta a infiammazioni tendinee.</div>", unsafe_allow_html=True)

            st.markdown("**Carico per Giorno della Settimana**")
            df['Giorno_Settimana'] = df['Giorno'].dt.day_name()
            df_day = df.groupby('Giorno_Settimana')['Distanza (km)'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig_day = px.bar(df_day, x='Giorno_Settimana', y='Distanza (km)', height=300, color_discrete_sequence=['#00E5FF'])
            st.plotly_chart(style_fig(fig_day), use_container_width=True)
            st.markdown("<div class='explain-text'>Visualizza la tua routine. Assicurati che ai giorni con barre alte seguano giorni con barre basse o assenti (recupero attivo).</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=300, markers=True)
            fig_cum.update_traces(line_color="#00E5FF")
            st.plotly_chart(style_fig(fig_cum), use_container_width=True)
            st.markdown("<div class='explain-text'>Una linea retta indica costanza. Una linea piatta indica stop o infortuni.</div>", unsafe_allow_html=True)

            record_km = df.loc[df['Distanza (km)'].idxmax()]
            record_vel = df.loc[df['Velocità (km/h)'].idxmax()]
            giorni_attivi = (df['Distanza (km)'] > 0).sum()
            streak = int((df['Distanza (km)'] > df['Distanza (km)'].mean()).astype(int).groupby((df['Distanza (km)'] <= df['Distanza (km)'].mean()).cumsum()).cumsum().max())

            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
                <h3 style='color:#FFB020; margin-bottom:15px;'>Bacheca Record — Ultimi 90 giorni</h3>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Corsa più lunga</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_km['Distanza (km)']:.1f} km</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Velocità massima</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_vel['Velocità (km/h)']:.1f} km/h</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Miglior striscia sopra media</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{streak} allenamenti</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Giorni con allenamento</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{giorni_attivi} / {len(df)}</strong>
                </div>
                <p style='color:#566178; font-size:0.8em; margin-top:12px; margin-bottom:0; font-family:"Inter",sans-serif;'>Ogni record battuto è un passo in più verso l'obiettivo finale.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', color='RPE', color_continuous_scale=[[0,'#0E4A57'],[0.5,'#00E5FF'],[1,'#FF6A3D']], height=300)
            st.plotly_chart(style_fig(fig2), use_container_width=True)
            st.markdown("<div class='explain-text'>Più i punti scendono verso il basso a destra, più il tuo cuore è efficiente (vai veloce faticando poco). I punti arancioni sono i lavori massimali.</div>", unsafe_allow_html=True)

            st.markdown("**Ripartizione Zone Cardiache**")
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Fondo Lento)', 'Z3 (Medio/Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df['Zone'] = pd.cut(df['FC Media'], bins=bins, labels=labels)
            zone_counts = df['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', hole=0.6, height=300, color_discrete_sequence=['#00E5FF','#00B8D4','#0E4A57','#FFB020','#FF6A3D'])
            st.plotly_chart(style_fig(fig_zones), use_container_width=True)
            st.markdown("<div class='explain-text'>Un allenamento sano prevede 80% in Z1/Z2 e 20% in Z4/Z5. Evita di rimanere intrappolato in Z3 (zona grigia), che stanca senza allenare efficacemente.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=300, color_discrete_sequence=['#00E5FF'])
            fig3.add_vline(x=3.5, line_dash="dash", line_color="#00F5A0")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig3), use_container_width=True)
            st.markdown("<div class='explain-text'>Mostra quante volte hai spinto al massimo (oltre la linea arancione) e quante volte hai recuperato (sotto la linea verde).</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=300, markers=True)
            fig_sleep.update_traces(line_color="#00E5FF")
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="#00F5A0")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_sleep), use_container_width=True)
            st.markdown("<div class='explain-text'>Cerca di stare sempre sopra la linea verde. Le discese verso la linea arancione corrispondono a cali di prestazione muscolare.</div>", unsafe_allow_html=True)

            st.markdown("**Debito di Sonno (Rolling 7gg)**")
            df['Debito'] = df['Ore Sonno'].apply(lambda x: max(0, 7.5 - x)).rolling(7).sum()
            fig_debt = px.area(df, x='Giorno', y='Debito', height=300, color_discrete_sequence=['#FF6A3D'])
            st.plotly_chart(style_fig(fig_debt), use_container_width=True)
            st.markdown("<div class='explain-text'>Quest'area è fatica accumulata. Se il debito supera le 5 ore in una settimana, il rischio di strappi o contratture decuplica.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', color='Rischio Infortunio', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], height=300)
            fig4.add_hline(y=7, line_dash="dash", line_color="#FFB020")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="#FFB020")
            st.plotly_chart(style_fig(fig4), use_container_width=True)
            st.markdown("<div class='explain-text'>Il quadrante in alto a sinistra (poco sonno, alto sforzo) è la 'zona critica'. Evita che i punti cadano in quell'area.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: 'ALTO' if x == 1 else 'OK')

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#111827', align='center', font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#0E1420', align='center', font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
        st.plotly_chart(style_fig(fig_table), use_container_width=True)

# ----------------- PAGINA 3: KPI DASHBOARD -----------------
elif pagina == "KPI DASHBOARD":
    header_block(
        "Modulo 03 — Live Monitoring",
        "KPI DASHBOARD: CRUSCOTTO PRESTAZIONALE",
        "Bilancio carico/recupero, rischio infortunio e profilo atletico calcolati sui parametri appena inseriti.",
        IMG_HERO_KPI, "Real-Time Dashboard"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario in 'Analisi Stato Di Forma' prima di accedere a questa dashboard.")
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
        st.markdown("""
        <div class='explain-text' style='margin-bottom: 25px;'>
        La linea verde rappresenta la tua "Batteria" (quanto dormi e recuperi). L'area arancione è quanto sforzo spremi dal tuo corpo. Finché la linea verde avvolge i picchi arancioni, sei in stato di Supercompensazione (migliori). Se l'arancione sta costantemente sopra il verde, stai andando in Overtraining.
        </div>
        """, unsafe_allow_html=True)

        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        if risk_score < 25:
            status_color, status_text = "#00F5A0", "OTTIMALE"
        elif risk_score < 60:
            status_color, status_text = "#FFB020", "MODERATO"
        else:
            status_color, status_text = "#FF6A3D", "CRITICO"

        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 4px; font-family:\"Space Grotesk\",sans-serif;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {status_color};'><div class='section-label'>Rischio Infortunio</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k2:
            rec_color = "#00F5A0" if recovery_score >= 75 else "#FFB020" if recovery_score >= 40 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {rec_color};'><div class='section-label'>Recovery Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k3:
            sma_color = "#00F5A0" if sma < 10 else "#FFB020" if sma < 15 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {sma_color};'><div class='section-label'>SMA Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#8792A3'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#E8ECF2"}, 'bar': {'color': status_color, 'thickness': 0.75}, 'bgcolor': "#111827", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': "rgba(0, 245, 160, 0.08)"}, {'range': [25, 60], 'color': "rgba(255, 176, 32, 0.08)"}, {'range': [60, 100], 'color': "rgba(255, 106, 61, 0.08)"}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Indicatore di Rischio:</strong> Calcola la probabilità probabilistica di un sovraccarico lesivo se svolgi la sessione impostata, bilanciando recupero e sforzo.</div>", unsafe_allow_html=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (%)'], fill='toself', name='Parametri',
                marker=dict(color=status_color), line=dict(color=status_color)
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')), height=360)
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Radar Fisiologico:</strong> Mappa l'impronta del tuo stato di forma odierno. Un'area troppo sbilanciata verso l'alto (Stress/RPE) senza un equivalente supporto del sonno segnala un pericolo biomeccanico.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Il Tuo Profilo Atletico AI")
        cv_sonno, cv_rpe = df['Ore Sonno'].std() / df['Ore Sonno'].mean(), df['RPE'].std() / df['RPE'].mean()
        consistenza = max(0, 100 - (cv_sonno + cv_rpe) * 100)

        if recovery_score >= 75 and sma < 10:
            archetipo, arch_col, arch_desc = "Il Bilanciato", "#00F5A0", "Gestisci sonno e carichi con grande equilibrio. Il tuo corpo lavora in supercompensazione costante: mantieni questa routine."
        elif r['stress_lavoro'] >= 7 and r['ore_sonno'] < 7:
            archetipo, arch_col, arch_desc = "Il Guerriero Stanco", "#FFB020", "Spingi forte nonostante stress e sonno limitato. Grande grinta, ma il conto arriva: pianifica un blocco di recupero prima possibile."
        elif sma >= 15:
            archetipo, arch_col, arch_desc = "L'Instancabile", "#FF6A3D", "Accumuli carico su carico. Ottimo motore, ma attenzione: senza pause il rischio di crollo fisico o mentale cresce rapidamente."
        else:
            archetipo, arch_col, arch_desc = "Il Costante", "#00E5FF", "Il tuo profilo è stabile e prevedibile: la base ideale su cui costruire progressi graduali e a basso rischio infortuni."

        col_arch1, col_arch2 = st.columns([1, 2])
        with col_arch1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 2px solid {arch_col}; display:flex; flex-direction:column; justify-content:center;'>
                <h3 style='color:{arch_col}; margin:5px 0; font-size:1.3em;'>{archetipo}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_arch2:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; height:100%;'>
                <p style='color:#B8C2D0; font-size:1.02em; margin-bottom:15px; font-family:"Inter",sans-serif;'>{arch_desc}</p>
                <p style='color:#8792A3; margin-bottom:5px; font-family:"Inter",sans-serif; font-size:0.9em;'>Indice di Consistenza (90gg)</p>
                <div style='background:#111827; border-radius:8px; overflow:hidden; height:20px;'>
                    <div style='background: linear-gradient(90deg, #00E5FF, #00F5A0); width:{min(consistenza,100):.0f}%; height:100%; text-align:right; padding-right:8px; color:#04121a; font-size:0.78em; font-weight:700; line-height:20px; font-family:"JetBrains Mono",monospace;'>{consistenza:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 4: ANALISI PREDITTIVA ML -----------------
elif pagina == "ANALISI PREDITTIVA ML":
    header_block(
        "Modulo 04 — Model Explainability",
        "ANALISI PREDITTIVA ML",
        "Modelli multi-livello addestrati sul tuo storico per prevedere performance, pattern e rischi.",
        IMG_HERO_ML, "Machine Learning Engine"
    )

    df = st.session_state.dati.copy()

    st.markdown("""
    <div class='info-box'>
    <h3>Cos'è il Machine Learning in questo contesto?</h3>
    <p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>Il sistema non usa formule matematiche rigide. Piuttosto, gli algoritmi "imparano" dai tuoi 90 giorni di dati storici identificando correlazioni nascoste (es. come il sonno impatta la tua frequenza cardiaca a una certa velocità). I modelli generano previsioni e segmentazioni altamente personalizzate sulla tua bio-dinamica unica.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Preprocessing per i modelli supervisionati (Rischio Infortunio)
        X_train_class = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
        y_train_class = df['Rischio Infortunio'].values
        scaler = StandardScaler()
        X_scaled_class = scaler.fit_transform(X_train_class)

        # Tab layout
        t_ml1, t_ml2, t_ml3, t_ml4, t_ml5, t_ml6 = st.tabs([
            "Random Forest", "Logistic Regression", "Linear Regression", "Cluster K-Means", "Stress Prediction", "Simulatore What-If"
        ])

        with t_ml1:
            st.markdown("### Random Forest Classifier (Previsione Infortunio)")
            st.markdown("<div class='explain-text'><strong>Come funziona:</strong> Il modello costruisce 100 'Alberi Decisionali', ognuno dei quali valuta i tuoi parametri in modo leggermente diverso. Il risultato finale è la media dei loro 'voti'. È eccellente per catturare relazioni non-lineari.</div>", unsafe_allow_html=True)
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
            rf_model.fit(X_scaled_class, y_train_class)
            
            c1, c2 = st.columns(2)
            with c1:
                feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']
                importances = rf_model.feature_importances_
                imp_data = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
                fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'))
                fig_imp.update_layout(height=350, yaxis=dict(autorange="reversed"), title="Importanza delle Variabili")
                st.plotly_chart(style_fig(fig_imp), use_container_width=True)
            with c2:
                y_pred_rf = rf_model.predict(X_scaled_class)
                cm = confusion_matrix(y_train_class, y_pred_rf)
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred: Sicuro', 'Pred: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 20, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False))
                fig_cm.update_layout(height=350, title="Matrice di Confusione (Test)")
                st.plotly_chart(style_fig(fig_cm), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Risultati:</strong> Il grafico a sinistra mostra quali parametri influenzano di più il 'voto' degli alberi. La matrice a destra indica la capacità del modello di non dare falsi allarmi sui dati storici.</div>", unsafe_allow_html=True)

        with t_ml2:
            st.markdown("### Logistic Regression (Probabilità Lineare di Infortunio)")
            st.markdown("<div class='explain-text'><strong>Come funziona:</strong> Modella la probabilità di infortunio usando un'equazione lineare. A differenza della 'Foresta', ci fornisce dei 'Pesi' (Coefficienti): quelli positivi aumentano il rischio, quelli negativi lo diminuiscono.</div>", unsafe_allow_html=True)
            
            log_model = LogisticRegression(random_state=42)
            log_model.fit(X_scaled_class, y_train_class)
            coefs = log_model.coef_[0]
            
            colors = ['#FF6A3D' if c > 0 else '#00F5A0' for c in coefs]
            fig_log = go.Figure(go.Bar(x=feature_names, y=coefs, marker_color=colors))
            fig_log.update_layout(height=400, title="Impatto Direzionale (Coefficienti LogReg)", yaxis_title="Impatto (Odds Logaritmico)")
            fig_log.add_hline(y=0, line_color="#E8ECF2", line_width=1)
            st.plotly_chart(style_fig(fig_log), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Risultati:</strong> Le barre verdi (valori negativi, come le Ore di Sonno) indicano i fattori 'Protettivi' che abbassano la probabilità di lesioni. Le barre arancioni spingono la probabilità verso l'alto (es. Stress e RPE elevato).</div>", unsafe_allow_html=True)

        with t_ml3:
            st.markdown("### Linear Regression (Previsione FC Media)")
            st.markdown("<div class='explain-text'><strong>Come funziona:</strong> Cerca di tracciare la linea retta migliore per predire una variabile continua. Qui lo usiamo per prevedere quanti battiti al minuto avrai in base al ritmo (Velocità), Temperatura e Distanza.</div>", unsafe_allow_html=True)
            
            X_lr = df[['Velocità (km/h)', 'Temp (°C)', 'Distanza (km)']]
            y_lr = df['FC Media']
            lr_model = LinearRegression()
            lr_model.fit(X_lr, y_lr)
            df['FC_Predetta'] = lr_model.predict(X_lr)
            
            fig_lr = px.scatter(df, x='FC Media', y='FC_Predetta', color='RPE', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']])
            fig_lr.add_shape(type="line", x0=df['FC Media'].min(), y0=df['FC Media'].min(), x1=df['FC Media'].max(), y1=df['FC Media'].max(), line=dict(color="#00F5A0", dash="dash"))
            fig_lr.update_layout(height=400, title="Battiti Reali vs Battiti Predetti (Regressione Lineare)", xaxis_title="FC Reale", yaxis_title="FC Predetta")
            st.plotly_chart(style_fig(fig_lr), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Risultati:</strong> La linea tratteggiata verde è la predizione perfetta. Più i punti (gli allenamenti) sono vicini alla linea, più il tuo cuore risponde in modo meccanico e prevedibile agli sforzi. I punti molto lontani indicano giorni in cui eri particolarmente stanco o particolarmente in forma rispetto alla tua norma termodinamica.</div>", unsafe_allow_html=True)

        with t_ml4:
            st.markdown("### Cluster Analysis (Segmentazione K-Means)")
            st.markdown("<div class='explain-text'><strong>Come funziona:</strong> Un algoritmo 'Non Supervisionato'. Non gli diciamo cosa cercare; gli passiamo solo le tue Frequenze Cardiache e Distanze e gli chiediamo di trovare 3 gruppi naturali di allenamento.</div>", unsafe_allow_html=True)
            
            X_clust = df[['Distanza (km)', 'FC Media']]
            km = KMeans(n_clusters=3, random_state=42)
            df['Cluster_ID'] = km.fit_predict(X_clust)
            
            # Map cluster IDs to qualitative names
            cluster_centers = km.cluster_centers_
            df['Cluster_Type'] = df['Cluster_ID'].apply(lambda x: f"Tipo {x+1}")
            
            fig_km = px.scatter(df, x='Distanza (km)', y='FC Media', color='Cluster_Type', color_discrete_sequence=['#00E5FF', '#FFB020', '#00F5A0'], size='RPE')
            fig_km.update_layout(height=400, title="I 3 Modelli di Allenamento 'Scoperti' dall'Intelligenza Artificiale")
            st.plotly_chart(style_fig(fig_km), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Risultati:</strong> L'AI ha classificato le tue sessioni. Solitamente scopre da sola queste tipologie: le sessioni corte/intense, i lunghi lenti e le sessioni di recupero. Ti aiuta a vedere se stai variando abbastanza gli stimoli (se i punti sono tutti in un solo cluster, il tuo allenamento è troppo monotono).</div>", unsafe_allow_html=True)

        with t_ml5:
            st.markdown("### Stress / Overload Prediction (Time Series Analisys)")
            st.markdown("<div class='explain-text'><strong>Come funziona:</strong> Calcola una media mobile dello stress sistemico (SMA = Stress * RPE / Sonno) ed estrapola la tendenza per individuare periodi di crollo imminente prima che avvengano.</div>", unsafe_allow_html=True)
            
            df_stress = df[['Giorno', 'SMA']].sort_values('Giorno').copy()
            df_stress['SMA_Rolling'] = df_stress['SMA'].rolling(7, min_periods=1).mean()
            
            fig_sp = px.area(df_stress, x='Giorno', y='SMA_Rolling', color_discrete_sequence=['#FF6A3D'])
            fig_sp.add_hline(y=15, line_dash="dash", line_color="#FFB020", annotation_text="Soglia di Allerta")
            fig_sp.update_layout(height=400, title="Tendenza Cumulativa dello Stress (Media 7gg)")
            st.plotly_chart(style_fig(fig_sp), use_container_width=True)
            
            st.markdown("<div class='explain-text'><strong>Risultati:</strong> Se l'area del grafico attraversa ripetutamente la linea d'allerta tratteggiata, il tuo sistema nervoso simpatico è sotto attacco continuo. Indica i momenti storici in cui eri (o sei) a un passo dal Burnout atletico.</div>", unsafe_allow_html=True)

        with t_ml6:
            st.markdown("### Simulatore What-If (Random Forest Live)")
            st.markdown("""<div class='info-box'><strong>Muovi le leve e osserva in tempo reale come cambia la previsione di infortunio.</strong> Questo sfrutta il modello Random Forest addestrato precedentemente.</div>""", unsafe_allow_html=True)
            
            base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6}

            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                sim_dist = st.slider("Distanza simulata (km)", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
                sim_sonno = st.slider("Ore di sonno simulate", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
            with col_sim2:
                sim_stress = st.slider("Stress simulato", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
                sim_rpe = st.slider("RPE simulato", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")

            sim_fc = 100 + sim_rpe * 10
            sim_input = np.array([[sim_dist, sim_sonno, sim_stress, sim_fc, sim_rpe]])
            
            # Predict using the trained rf_model
            sim_prob = rf_model.predict_proba(scaler.transform(sim_input))[0][1] * 100
            sim_color = "#FF6A3D" if sim_prob >= 60 else "#FFB020" if sim_prob >= 25 else "#00F5A0"

            col_simg1, col_simg2 = st.columns(2)
            with col_simg1:
                fig_sim_gauge = go.Figure(go.Indicator(mode="gauge+number", value=sim_prob, title={'text': "Rischio Simulato", 'font': {'color': '#8792A3'}}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color}, 'bgcolor': "#111827", 'borderwidth': 0}, number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}))
                fig_sim_gauge.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sim_gauge), use_container_width=True)
            with col_simg2:
                sonno_range = np.linspace(4, 10, 20)
                probs_range = [rf_model.predict_proba(scaler.transform(np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])))[0][1] * 100 for s in sonno_range]
                fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'}, title="Sensibilità: Rischio vs Ore di Sonno")
                fig_sens.update_traces(line_color="#00E5FF", line_width=3)
                fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#FF6A3D")
                fig_sens.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sens), use_container_width=True)

    except Exception as e:
        st.error(f"Errore caricamento modelli ML: {str(e)}")


# ----------------- PAGINA 5: CONSIGLIO FINALE -----------------
elif pagina == "CONSIGLIO FINALE":
    header_block(
        "Modulo 05 — Action Plan",
        "CONSIGLIO FINALE",
        "Protocollo pre/durante/post allenamento generato su misura per i parametri di oggi.",
        IMG_HERO_PLAN, "Coach Protocol"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario in 'Analisi Stato Di Forma' per generare il piano.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()

        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        distanza_target = r.get('distanza_oggi', 10.0)
        distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0

        if risk_score < 25: tit, col = "ALLENAMENTO INTENSO AUTORIZZATO", "#00F5A0"
        elif risk_score < 60: tit, col = "RECUPERO ATTIVO CONSIGLIATO", "#FFB020"
        else: tit, col = "RIPOSO OBBLIGATORIO", "#FF6A3D"

        st.markdown(f"""
        <div class='kpi-card' style='border: 1px solid {col}; background-color: rgba(0,0,0,0.35);'>
            <h2 style='color: {col}; margin: 0; border: none; font-size:1.6em;'>{tit}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .kpi-card-equal { background: #0E1420; border-radius: 12px; padding: 25px 20px; border: 1px solid #1c2333; height: 480px; display: flex; flex-direction: column; }
        .kpi-card-equal .kpi-equal-body { overflow-y: auto; flex-grow: 1; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar { width: 6px; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar-thumb { background: #1c2333; border-radius: 4px; }
        </style>
        """, unsafe_allow_html=True)

        tipo_all = r.get('tipo_allenamento', 'Easy Run')
        col_new1, col_new2, col_new3 = st.columns(3)
        
        with col_new1:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00E5FF;'>Distanza Consigliata</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{distanza_consigliata:.1f} km</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>su {distanza_target}km desiderati</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center; font-family:"Inter",sans-serif;'>Tipo allenamento: <strong style='color:#B8C2D0;'>{tipo_all}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new2:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:{col};'>Rischio Calcolato</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{risk_score:.0f}%</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>Probabilità Infortunio/Burnout</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center; font-family:"Inter",sans-serif;'>Recovery Score: <strong style='color:#B8C2D0;'>{recovery_score:.0f}%</strong> · SMA: <strong style='color:#B8C2D0;'>{sma:.1f}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new3:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00F5A0;'>Protocollo Coach Dettagliato</h3>
                <div class='kpi-equal-body' style='color:#B8C2D0; font-size:0.85em; text-align:left; font-family:"Inter",sans-serif;'>
                    <strong style='color:#00E5FF;'>PRE-ALLENAMENTO (T-90/-15 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>T-90': pasto leggero, ~{round(distanza_target * 3)}g carboidrati.</li>
                        <li>T-30': {round(distanza_target * 20)}ml di liquidi.</li>
                        <li>T-15': mobilità dinamica anche/caviglie, skip (5').</li>
                    </ul>
                    <strong style='color:#FFB020;'>DURANTE</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Sorso d'acqua ogni 20' se superi i 60'.</li>
                        <li>Cadenza target 170-180 spm, respiro controllato.</li>
                    </ul>
                    <strong style='color:#00F5A0;'>POST (0-30 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Entro 30': ~{round(distanza_target * 1.2) + 15}g proteine + ~{round(distanza_target * 4) + 20}g carboidrati.</li>
                        <li>Stretching statico gentile 8-10'.</li>
                        <li>Rullo miofasciale 5' su quadricipiti.</li>
                    </ul>
                    <strong style='color:#8b5cf6;'>SERALE</strong>
                    <ul style='margin-top:5px; padding-left:18px; margin-bottom:0;'>
                        <li>Punta a {max(r['ore_sonno'],7.5):.1f}h di sonno per recupero cellulare.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.subheader("Analisi Parametri vs Media (90 giorni)")
        media_sonno_90, media_stress_90, media_rpe_90 = df['Ore Sonno'].mean(), df['Stress Lavoro'].mean(), df['RPE'].mean()
        sonno_vs_media, stress_vs_media, rpe_vs_media = r['ore_sonno'] - media_sonno_90, r['stress_lavoro'] - media_stress_90, r['rpe_previsto'] - media_rpe_90

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            sb, sc = ("SOTTO MEDIA", "#FF6A3D") if sonno_vs_media < -0.5 else ("SOPRA MEDIA", "#00F5A0") if sonno_vs_media > 0.5 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{sc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{sb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['ore_sonno']:.1f}h</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_sonno_90:.1f}h</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Il tempo dedicato alla rigenerazione cellulare. Un deficit rallenta il recupero muscolare.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a2:
            stb, stc = ("SOTTO MEDIA", "#00F5A0") if stress_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if stress_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{stc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{stb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['stress_lavoro']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_stress_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Il carico cognitivo e nervoso accumulato. Uno stress alto alza il cortisolo e il rischio di infortuni.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a3:
            rpb, rpc = ("SOTTO MEDIA", "#00F5A0") if rpe_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if rpe_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{rpc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{rpb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['rpe_previsto']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_rpe_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Lo sforzo programmato per la sessione. Valori superiori alla media indicano un forte sovraccarico sistemico.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("<h2>Proiezione Fisiologica Odierna</h2>", unsafe_allow_html=True)

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            time_x = np.arange(0, 60, 5)
            hr_y = [r['fc_riposo'] + 20] + [r['fc_riposo'] + 70 + np.random.randint(-5, 5) for _ in range(10)] + [r['fc_riposo'] + 30]
            fig_pace = px.line(x=time_x, y=hr_y, title="1. Curva BPM Consigliata Oggi", labels={'x':'Minuti', 'y':'BPM'})
            fig_pace.update_traces(line_color="#FF6A3D")
            fig_pace.update_layout(height=300)
            st.plotly_chart(style_fig(fig_pace), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Dinamica Cardiaca:</strong> Segui questo profilo di riscaldamento iniziale e defaticamento per non sovraccaricare il sistema cardiovascolare. Il picco riflette il tuo RPE impostato.</div>", unsafe_allow_html=True)
        
        with g_col2:
            hours = ["+0h", "+6h", "+12h", "+24h", "+48h"]
            rec_y = [30, 55, 75, 95, 100] if risk_score < 50 else [15, 30, 50, 70, 90]
            fig_rec = px.bar(x=hours, y=rec_y, title="2. Tempo di Ricarica Glicogeno Stimato", labels={'x':'Ore Post-Workout', 'y':'% Energie'})
            fig_rec.update_traces(marker_color="#00F5A0")
            fig_rec.update_layout(height=300)
            st.plotly_chart(style_fig(fig_rec), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Sintesi Glucidica:</strong> Tempo stimato per riempire di nuovo le riserve di glicogeno muscolare dopo questa sessione. Assumi carboidrati nei primi 30' per velocizzare la barra al 100%.</div>", unsafe_allow_html=True)
            
        g_col3, g_col4 = st.columns(2)
        with g_col3:
            fig_acwr = go.Figure(data=[
                go.Bar(name='Carico Ultimi 7gg', x=['Carico'], y=[450], marker_color='#FFB020'),
                go.Bar(name='Media 28gg', x=['Carico'], y=[390], marker_color='#00E5FF')
            ])
            fig_acwr.update_layout(title="3. Bilancio Acuto vs Cronico (ACWR)", barmode='group', height=300)
            st.plotly_chart(style_fig(fig_acwr), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>ACWR (Acute:Chronic Workload Ratio):</strong> Compara il lavoro di questa settimana con il lavoro del mese. L'ideale è mantenere le barre simili (rapporto 1.0 - 1.3). Sopra 1.5 il rischio infortuni esplode.</div>", unsafe_allow_html=True)
        
        with g_col4:
            fig_pie2 = px.pie(values=[70, 20, 10], names=['Aerobico Base', 'Soglia Lattata', 'Anaerobico'], title="4. Ripartizione Energetica Richiesta", hole=0.6, color_discrete_sequence=['#00E5FF', '#FFB020', '#FF6A3D'])
            fig_pie2.update_layout(height=300)
            st.plotly_chart(style_fig(fig_pie2), use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Sistemi Energetici:</strong> Da quali serbatoi il tuo corpo prenderà l'energia. Oltre l'80% dovrebbe derivare dal sistema aerobico (ossigeno) per sviluppare resistenza senza produrre scorie lattacide.</div>", unsafe_allow_html=True)
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 🎯 Azioni Raccomandate in Base ai Risultati ML")
        
        if risk_score < 33:
            st.success("🟢 **PROGRESSIONE SICURA**: Il tuo corpo è perfettamente in grado di assorbire uno stimolo allenante oggi. Puoi permetterti di aumentare il volume o l'intensità del 5-10% rispetto a quanto fatto l'ultima settimana. La sessione proposta è pienamente approvata dal modello predittivo.")
        elif risk_score < 66:
            st.warning("🟡 **MANTENIMENTO**: Ti trovi in una zona di equilibrio precario, probabilmente a causa di un leggero deficit di sonno o accumulo di stress. Non inserire nuovi stimoli intensi (ripetute/salite). Mantieni il piano originario tagliando il 10% del volume totale, focalizzandoti solo sulla regolarità.")
        else:
            st.error("🔴 **RECUPERO ATTIVO (RISCHIO ALTO)**: I segnali del sistema indicano un elevato rischio di infortunio o sovraccarico se prosegui con il piano impostato. **Consiglio primario:** Converti la sessione odierna in mobilità, stretching leggero o passeggiata rigenerante. La tua priorità assoluta per le prossime 24h è re-idratazione e recupero del sonno profondo.")
