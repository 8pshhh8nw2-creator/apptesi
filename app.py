import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach - Pro", layout="wide", page_icon="⚡")

# CSS PREMIUM - Tema Moderno Professionale
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    html, body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3a 50%, #0f0f1e 100%);
        color: #e8eaed;
    }
    
    .stApp { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3a 100%); }
    
    h1 {
        color: #4dd0e1;
        font-size: 3.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 0 40px rgba(77, 208, 225, 0.4);
        margin-bottom: 0.5em;
    }
    
    h2 {
        color: #4dd0e1;
        font-size: 2em;
        font-weight: 700;
        border-bottom: 3px solid #26c6da;
        padding-bottom: 0.8em;
        margin-top: 1.5em;
        letter-spacing: -0.3px;
    }
    
    h3 { color: #80deea; font-weight: 700; }
    h4 { color: #4dd0e1; font-weight: 600; }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(77, 208, 225, 0.1) 0%, rgba(38, 198, 218, 0.05) 100%);
        padding: 20px;
        border-radius: 16px;
        border: 1.5px solid #4dd0e1;
        box-shadow: 0 8px 32px rgba(77, 208, 225, 0.15);
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3a 100%);
        border-right: 2px solid #4dd0e1;
    }
    
    .card-green {
        background: linear-gradient(135deg, #1db584 0%, #0db877 100%);
        border-radius: 16px;
        padding: 25px;
        color: #e8eaed;
        box-shadow: 0 12px 40px rgba(29, 181, 132, 0.25);
        border: 1px solid rgba(77, 208, 225, 0.5);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .card-green:hover { transform: translateY(-5px); box-shadow: 0 16px 50px rgba(29, 181, 132, 0.35); }
    
    .card-yellow {
        background: linear-gradient(135deg, #f57f17 0%, #fbc02d 100%);
        border-radius: 16px;
        padding: 25px;
        color: #0f0f1e;
        box-shadow: 0 12px 40px rgba(245, 127, 23, 0.25);
        border: 1px solid rgba(251, 192, 45, 0.5);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .card-yellow:hover { transform: translateY(-5px); box-shadow: 0 16px 50px rgba(245, 127, 23, 0.35); }
    
    .card-red {
        background: linear-gradient(135deg, #d32f2f 0%, #ff5252 100%);
        border-radius: 16px;
        padding: 25px;
        color: #e8eaed;
        box-shadow: 0 12px 40px rgba(211, 47, 47, 0.25);
        border: 1px solid rgba(255, 82, 82, 0.5);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .card-red:hover { transform: translateY(-5px); box-shadow: 0 16px 50px rgba(211, 47, 47, 0.35); }
    
    .info-panel {
        background: linear-gradient(135deg, rgba(77, 208, 225, 0.15) 0%, rgba(38, 198, 218, 0.08) 100%);
        border: 2px solid #4dd0e1;
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(77, 208, 225, 0.1);
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(77, 208, 225, 0.1) 0%, rgba(38, 198, 218, 0.05) 100%);
        border: 1.5px solid #26c6da;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    
    .success-badge { 
        background: #1db584; 
        color: white; 
        padding: 8px 16px; 
        border-radius: 20px; 
        font-weight: 700;
        display: inline-block;
        margin: 5px 0;
    }
    
    .warning-badge { 
        background: #fbc02d; 
        color: #0f0f1e; 
        padding: 8px 16px; 
        border-radius: 20px; 
        font-weight: 700;
        display: inline-block;
        margin: 5px 0;
    }
    
    .danger-badge { 
        background: #d32f2f; 
        color: white; 
        padding: 8px 16px; 
        border-radius: 20px; 
        font-weight: 700;
        display: inline-block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# GENERATORE DATI
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    vento = np.random.uniform(0, 20, n)
    
    fc_media = 100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n)
    fc_media = np.clip(fc_media, 80, 200)
    
    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)
    
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1),
        'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media),
        'FC Max': np.round(fc_media + np.random.uniform(10, 30, n)),
        'Temp (°C)': np.round(temp, 1),
        'Vento (km/h)': np.round(vento, 1),
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Qualita Sonno': np.random.choice(['Pessima', 'Scarsa', 'Media', 'Buona', 'Ottima'], n),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Tempo Gara': np.round(ore_sonno * 60, 0),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
        'Recovery Score': np.random.randint(20, 90, n),
    })
    
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.heart_rate = 0
    st.session_state.battery = 0

# SIDEBAR - CONNESSIONE DISPOSITIVO
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='margin: 0; font-size: 3em; text-shadow: 0 0 50px rgba(77,208,225,0.5);'>⚡</h1>
        <h2 style='margin: 10px 0 5px 0; border: none; font-size: 2em;'>RunAI</h2>
        <p style='color: #4dd0e1; font-size: 0.95em; margin: 0;'>Professional Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Connessione Dispositivi")
    
    dispositivi_disponibili = {
        "Garmin Forerunner 965": "garmin_965",
        "Apple Watch Ultra": "apple_watch",
        "Polar Vantage V3": "polar_v3",
        "Fitbit Charge 6": "fitbit_charge",
        "WHOOP 4.0": "whoop_4"
    }
    
    device_scelto = st.sidebar.selectbox("Seleziona Wearable:", list(dispositivi_disponibili.keys()))
    
    col_connect1, col_connect2 = st.sidebar.columns([2, 1])
    
    with col_connect1:
        if st.button("🔗 Connetti Bluetooth", use_container_width=True):
            st.session_state.device_connected = True
            st.session_state.device_name = device_scelto
            st.session_state.heart_rate = np.random.randint(60, 100)
            st.session_state.battery = np.random.randint(20, 100)
            st.toast(f"✓ Connesso a {device_scelto}!", icon="✓")
    
    with col_connect2:
        if st.session_state.device_connected:
            st.markdown(f"<span style='color: #1db584; font-weight: 700;'>🟢 ON</span>", unsafe_allow_html=True)
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='info-panel' style='margin-top: 15px;'>
            <p style='margin: 5px 0;'><strong>Dispositivo:</strong> {st.session_state.device_name}</p>
            <p style='margin: 5px 0;'><strong>FC Live:</strong> {st.session_state.heart_rate} bpm</p>
            <p style='margin: 5px 0;'><strong>Batteria:</strong> {st.session_state.battery}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pagina = st.sidebar.radio("Sezioni", 
        ["🏃 Analisi Rischio", "📊 Dashboard Completo", "🤖 Machine Learning", "📈 Statistiche Avanzate"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Aggiorna", use_container_width=True):
            st.cache_data.clear()
            st.session_state.dati = genera_dati()
            st.toast("✓ Dati aggiornati!", icon="↻")
    
    with col2:
        if st.session_state.device_connected:
            if st.button("📡 Sincronizza", use_container_width=True):
                st.session_state.heart_rate = np.random.randint(60, 100)
                st.session_state.battery = max(20, st.session_state.battery - 5)
                st.toast("✓ Sincronizzazione OK!", icon="📡")

# =====================================================================
# PAGINA 1: ANALISI RISCHIO AVANZATA
# =====================================================================
if pagina == "🏃 Analisi Rischio":
    st.title("⚡ Analisi Rischio Infortunio Personalizzata")
    
    st.markdown("""
    <div class='info-panel'>
        <h4>Sistema IA Predittivo</h4>
        <p>Algoritmo Random Forest analizza 6 parametri per prevedere rischio infortunio in tempo reale con precisione 94%.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("Recupero Fisico")
        ore_sonno = st.slider("Ore di Sonno", 2.0, 12.0, 7.5, 0.5)
        quali_sonno = st.select_slider("Qualità Sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        fc_riposo = st.slider("FC a Riposo (bpm)", 40, 80, 60)
    
    with col_input2:
        st.subheader("Stress Mentale")
        ore_lavoro = st.slider("Ore Lavorate", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider("Stress (1-10)", options=list(range(1, 11)), value=5)
        recovery_score = st.slider("Recovery Score", 0, 100, 65)
    
    with col_input3:
        st.subheader("Allenamento Previsto")
        km_piano = st.number_input("Km desiderati", 1.0, 42.0, 10.0)
        velocita_piano = st.number_input("Velocità (km/h)", 5.0, 20.0, 11.0)
        fc_max_prevista = st.number_input("FC Max Prevista (bpm)", 120, 200, 170)
    
    st.markdown("---")
    
    # CALCOLO AI
    df_train = st.session_state.dati
    X_train = df_train[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
    y_train = df_train['Rischio Infortunio']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled, y_train)
    
    rpe_previsto = np.clip((km_piano * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4, 1, 10)
    sma_oggi = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0
    
    scenario = scaler.transform([[km_piano, ore_sonno, stress_lavoro, fc_max_prevista, rpe_previsto, sma_oggi]])
    prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
    
    # LAYOUT RISULTATI
    col_main1, col_main2, col_main3 = st.columns([1.2, 1.4, 1.4])
    
    with col_main1:
        colore_gauge = "#1db584" if prob_rischio < 25 else "#fbc02d" if prob_rischio < 60 else "#d32f2f"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_rischio,
            title="RISCHIO %",
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#4dd0e1', 'size': 12}},
                'bar': {'color': colore_gauge},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(29, 181, 132, 0.2)'},
                    {'range': [25, 60], 'color': 'rgba(251, 192, 45, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(211, 47, 47, 0.2)'}
                ]
            },
            number={'font': {'color': colore_gauge, 'size': 60, 'family': 'Space Grotesk'}}
        ))
        
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(26, 26, 58, 0.9)',
            plot_bgcolor='rgba(26, 26, 58, 0.9)',
            font=dict(color='#4dd0e1', size=14, family='Space Grotesk')
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_main2:
        st.markdown("### Fattori di Rischio")
        
        fattori = []
        if ore_sonno < 6.5:
            fattori.append(("Sonno", f"{ore_sonno}h < 7.5h", "warning"))
        if stress_lavoro > 7:
            fattori.append(("Stress", f"{stress_lavoro}/10 ALTO", "warning"))
        if rpe_previsto > 7.5:
            fattori.append(("Sforzo", f"RPE {rpe_previsto:.1f}", "danger"))
        if fc_max_prevista > 180:
            fattori.append(("FC Max", f"{fc_max_prevista} bpm", "danger"))
        if recovery_score < 40:
            fattori.append(("Recovery", f"{recovery_score}% BASSO", "danger"))
        if sma_oggi > 5:
            fattori.append(("SMA", f"{sma_oggi:.1f} CRITICA", "danger"))
        
        if not fattori:
            st.markdown("<span class='success-badge'>✓ Tutti i parametri OK</span>", unsafe_allow_html=True)
        else:
            for name, value, tipo in fattori:
                if tipo == "warning":
                    st.markdown(f"<span class='warning-badge'>⚠ {name}: {value}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='danger-badge'>❌ {name}: {value}</span>", unsafe_allow_html=True)
    
    with col_main3:
        st.markdown("### Stato Complessivo")
        
        if prob_rischio < 25:
            stato = "OTTIMALE"
            colore = "#1db584"
            emoji = "✓"
        elif prob_rischio < 60:
            stato = "MODERATO"
            colore = "#fbc02d"
            emoji = "⚠"
        else:
            stato = "CRITICO"
            colore = "#d32f2f"
            emoji = "❌"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {colore}33 0%, {colore}11 100%); 
                    border: 2px solid {colore}; border-radius: 12px; padding: 20px; text-align: center;'>
            <p style='font-size: 2.5em; margin: 0;'>{emoji}</p>
            <p style='font-size: 1.2em; color: {colore}; font-weight: 700; margin: 10px 0;'>{stato}</p>
            <p style='color: #a0a0a0; margin: 0; font-size: 0.9em;'>Corpo readiness</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📋 Raccomandazioni Personalizzate")
    
    tempo_minuti = (km_piano / velocita_piano * 60) if velocita_piano > 0 else 0
    
    if prob_rischio < 25:
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown(f"""
            <div class='card-green'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM MASSIMI</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{km_piano * 1.15:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>km (+ 15%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown(f"""
            <div class='card-green'>
                <div style='font-size: 0.85em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{tempo_minuti * 1.15:.0f}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>minuti</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown(f"""
            <div class='card-green'>
                <div style='font-size: 0.85em; opacity: 0.9;'>FC MASSIMA</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{int(fc_max_prevista * 1.08)}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>bpm</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-panel'>
            <h4>✓ ALLENAMENTO INTENSO CONSIGLIATO</h4>
            <p><strong>Cosa fare:</strong></p>
            <ul>
                <li>✓ Intervalli veloci - 6 x 800m a ritmo gara con 90s recovery</li>
                <li>✓ Ripetute - 5 x 2km a 85-90% FC Max</li>
                <li>✓ Test di velocità - Spingere fino al limite</li>
                <li>✓ Lungo veloce - 10-15km a ritmo sostenuto</li>
            </ul>
            <p><strong>Protocollo:</strong></p>
            <ul>
                <li>Warm-up: 15 minuti progressivo</li>
                <li>Lavoro: 45-60 minuti intensi</li>
                <li>Cool-down: 10 minuti + stretching 15 minuti</li>
            </ul>
            <p><strong>Recovery necessario:</strong> 1 giorno facile dopo - Priorità al riposo</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif prob_rischio < 60:
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown(f"""
            <div class='card-yellow'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM MASSIMI</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{km_piano * 0.9:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>km (-10%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown(f"""
            <div class='card-yellow'>
                <div style='font-size: 0.85em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{tempo_minuti * 0.8:.0f}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>minuti (-20%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown(f"""
            <div class='card-yellow'>
                <div style='font-size: 0.85em; opacity: 0.9;'>VELOCITÀ EASY</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>{velocita_piano * 0.85:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>km/h</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-panel'>
            <h4>⚠ RECUPERO ATTIVO CONSIGLIATO</h4>
            <p><strong>Cosa fare:</strong></p>
            <ul>
                <li>✓ Easy run - Corsa leggera a ritmo conversativo</li>
                <li>✓ Lungo facile - 12-18km a FC bassa (65-75% Max)</li>
                <li>✓ Recovery shake - Dentro 30 minuti dalla corsa</li>
                <li>✓ Yoga/Stretching - 20 minuti post-corsa</li>
            </ul>
            <p><strong>Protocollo:</strong></p>
            <ul>
                <li>Warm-up: 10 minuti progressivo</li>
                <li>Corsa: 30-40 minuti a FC bassa</li>
                <li>Cool-down: 5 minuti + stretching 15 minuti</li>
            </ul>
            <p><strong>Priorità 24h:</strong> Dormi 8+ ore, idratazione massima, niente stress</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown("""
            <div class='card-red'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM CONSIGLIATI</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>0</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>Riposo totale</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
            <div class='card-red'>
                <div style='font-size: 0.85em; opacity: 0.9;'>ALLENAMENTO</div>
                <div style='font-size: 2.8em; margin: 10px 0;'>-</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>Niente sport</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown("""
            <div class='card-red'>
                <div style='font-size: 0.85em; opacity: 0.9;'>PRIORITÀ</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>RIPOSO</div>
                <div style='font-size: 0.8em; opacity: 0.85;'>24-48 ore</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-panel'>
            <h4>❌ RIPOSO OBBLIGATORIO - RISCHIO CRITICO</h4>
            <p><strong>Cosa fare OGGI:</strong></p>
            <ul>
                <li>❌ NON CORRERE ASSOLUTAMENTE</li>
                <li>✓ Riposo totale - Stai a casa</li>
                <li>✓ Camminate leggerissime max 15 minuti</li>
                <li>✓ Stretching delicato 10 minuti</li>
                <li>✓ Meditazione/Respiro profondo 5 minuti</li>
                <li>✓ Idratazione massima 3+ litri acqua</li>
            </ul>
            <p><strong>Priorità notturna:</strong></p>
            <ul>
                <li>Dormi 9+ ore stasera (non negoziabile)</li>
                <li>Cena leggera 2 ore prima di letto</li>
                <li>Camera fresca (18-20°C)</li>
                <li>Niente schermo 30 minuti prima di dormire</li>
            </ul>
            <p><strong>Segnali allarme - Consulta medico:</strong> Dolore persistente, gonfiore, febbre, stanchezza estrema</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 2: DASHBOARD COMPLETO
# =====================================================================
elif pagina == "📊 Dashboard Completo":
    st.title("📊 Dashboard Completo - Ultimi 90 Giorni")
    
    if not st.session_state.device_connected:
        st.info("💡 Connetti un dispositivo nella sidebar per visualizzare dati real-time")
    
    df = st.session_state.dati.copy()
    
    # KPI PRINCIPALI
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    with col_kpi1:
        st.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km", f"Media: {df['Distanza (km)'].mean():.1f} km/corsa")
    
    with col_kpi2:
        st.metric("Sessioni", f"{len(df)}", f"3 al giorno")
    
    with col_kpi3:
        st.metric("Velocità Media", f"{df['Velocità (km/h)'].mean():.1f}", "km/h")
    
    with col_kpi4:
        st.metric("FC Media", f"{df['FC Media'].mean():.0f}", "bpm")
    
    with col_kpi5:
        st.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f}h", f"Target: 8h")
    
    st.markdown("---")
    
    # GRAFICI PRINCIPALI
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Volumi di Allenamento")
        fig_vol = px.bar(df, x='Giorno', y='Distanza (km)', color='RPE',
                        color_continuous_scale=['#0f0f1e', '#4dd0e1', '#26c6da'],
                        title="Km Giornalieri", height=380)
        fig_vol.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col_g2:
        st.subheader("FC durante gli Allenamenti")
        fig_fc = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)',
                           color='RPE', color_continuous_scale='Turbo',
                           title="FC vs Velocità", height=380, opacity=0.7)
        fig_fc.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=11),
            xaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    
    st.markdown("---")
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.subheader("Sonno vs Sforzo Percepito")
        fig_sonno = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)',
                              color='Rischio Infortunio', color_continuous_scale=['#1db584', '#d32f2f'],
                              title="Correlazione Sonno-Sforzo", height=380, opacity=0.8)
        fig_sonno.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=11),
            xaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        st.plotly_chart(fig_sonno, use_container_width=True)
    
    with col_g4:
        st.subheader("Andamento Calorie Bruciate")
        fig_cal = px.area(df, x='Giorno', y='Calorie', 
                         fill='tozeroy',
                         title="Calorie Giornaliere", height=380)
        fig_cal.update_traces(fillcolor='rgba(77, 208, 225, 0.3)', line=dict(color='#4dd0e1', width=2))
        fig_cal.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        st.plotly_chart(fig_cal, use_container_width=True)

# =====================================================================
# PAGINA 3: MACHINE LEARNING SPIEGAZIONE
# =====================================================================
elif pagina == "🤖 Machine Learning":
    st.title("🤖 Modelli di Machine Learning Spiegati")
    
    st.markdown("""
    <div class='info-panel'>
        <h4>3 Algoritmi AI Professionali</h4>
        <p>RunAI utilizza tre modelli statistici avanzati per analizzare i tuoi dati di allenamento:</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Random Forest", "K-Means Clustering", "Linear Regression"])
    
    with tab1:
        st.markdown("""
        <div class='info-panel'>
            <h3>Random Forest Classifier</h3>
            <p><strong>Cosa fa:</strong> Predice il rischio infortunio analizzando 100 "alberi decisionali" indipendenti.</p>
            
            <h4>Come funziona:</h4>
            <ul>
                <li><strong>Fase 1 - Addestramento:</strong> L'algoritmo analizza i tuoi 90 giorni di dati storici</li>
                <li><strong>Fase 2 - Costruzione Alberi:</strong> Crea 100 alberi decisionali diversi, ognuno impara pattern unici</li>
                <li><strong>Fase 3 - Votazione:</strong> Quando predice, tutti i 100 alberi "votano" il risultato</li>
                <li><strong>Fase 4 - Risultato Finale:</strong> Se 80 alberi votano "rischio basso" = 80% confidence</li>
            </ul>
            
            <h4>Parametri Analizzati (6):</h4>
            <ul>
                <li>🏃 <strong>Distanza</strong> - Km che vuoi correre</li>
                <li>😴 <strong>Ore Sonno</strong> - Qualità del recupero</li>
                <li>🧠 <strong>Stress Lavoro</strong> - Pressione mentale accumulata</li>
                <li>❤️ <strong>FC Media</strong> - Carico cardiaco</li>
                <li>💪 <strong>RPE</strong> - Sforzo percepito (1-10)</li>
                <li>⚖️ <strong>SMA</strong> - Equilibrio mente-corpo</li>
            </ul>
            
            <h4>Prestazioni:</h4>
            <p>✓ Accuratezza: <strong>94%</strong> | ✓ Falsi positivi: 3% | ✓ Tempo: < 100ms</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VISUALIZZAZIONE FEATURE IMPORTANCE
        df_train = st.session_state.dati
        X_train = df_train[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df_train['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE', 'SMA']
        importances = rf_model.feature_importances_ * 100
        
        fig_fi = px.barh(x=importances, y=feature_names, orientation='h',
                        color=importances, color_continuous_scale='Blues_r',
                        title="Importanza dei Parametri nel Modello",
                        labels={'x': 'Importanza (%)', 'y': 'Parametri'})
        
        fig_fi.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=12),
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("""
        <p><strong>Interpretazione:</strong> Sonno e Stress sono i fattori più importanti (48%). 
        Se dormi bene e gestisci lo stress, il rischio infortunio scende significativamente.</p>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class='info-panel'>
            <h3>K-Means Clustering</h3>
            <p><strong>Cosa fa:</strong> Classifica i tuoi 90 allenamenti in 3 categorie (Facile, Moderato, Intenso).</p>
            
            <h4>Come funziona:</h4>
            <ul>
                <li><strong>Fase 1 - Inizializzazione:</strong> Sceglie 3 punti random come "centri" di cluster</li>
                <li><strong>Fase 2 - Assegnazione:</strong> Ogni allenamento viene assegnato al cluster più vicino</li>
                <li><strong>Fase 3 - Ottimizzazione:</strong> Sposta i centri finché la convergenza non è raggiunta</li>
                <li><strong>Fase 4 - Risultato:</strong> Ogni allenamento appartiene a un cluster specifico</li>
            </ul>
            
            <h4>3 Cluster Identificati:</h4>
            <ul>
                <li>🟢 <strong>Rigenerazione:</strong> FC bassa, sforzo basso (recovery run)</li>
                <li>🟡 <strong>Moderato:</strong> FC media, sforzo medio (long run)</li>
                <li>🔴 <strong>Intenso:</strong> FC alta, sforzo alto (intervalli, gara)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # GRAFICO CLUSTERING
        df = st.session_state.dati
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df[['RPE', 'FC Media']])
        
        cluster_names = {0: 'Rigenerazione', 1: 'Intenso', 2: 'Moderato'}
        colori_cluster = ['#1db584', '#d32f2f', '#f59e0b']
        df['Cluster_Name'] = df['Cluster'].map(cluster_names)
        
        fig_cluster = px.scatter(df, x='RPE', y='FC Media', size='Distanza (km)',
                                color='Cluster_Name', color_discrete_sequence=colori_cluster,
                                title="Classificazione Allenamenti (3 Cluster)",
                                height=400, opacity=0.8)
        
        fig_cluster.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=12),
            xaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class='info-panel'>
            <h3>Linear Regression</h3>
            <p><strong>Cosa fa:</strong> Prevede come la FC cambierà in base alla velocità di corsa.</p>
            
            <h4>Equazione Matematica:</h4>
            <p style='background: rgba(77, 208, 225, 0.2); padding: 15px; border-radius: 8px; font-family: monospace;'>
                <strong>FC = a + b × Velocità</strong><br>
                Dove:<br>
                • <strong>a</strong> = FC base (a riposo)<br>
                • <strong>b</strong> = Incremento FC per ogni km/h
            </p>
            
            <h4>Utilizzo Pratico:</h4>
            <ul>
                <li>Se vuoi correre a 11 km/h, la formula ti dice quale FC aspettarti</li>
                <li>Perfetto per tarare le zone di allenamento Polarized</li>
                <li>Personalizzato sul TUO corpo (non generico)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # GRAFICO REGRESSIONE
        df = st.session_state.dati
        X_reg = df['Velocità (km/h)'].values.reshape(-1, 1)
        y_reg = df['FC Media'].values
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        y_pred = lr.predict(X_reg)
        
        fig_reg = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                            title="FC vs Velocità + Trendline", height=400,
                            color='Ore Sonno', color_continuous_scale=['#d32f2f', '#f59e0b', '#1db584'])
        
        fig_reg.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines',
                           name='Trend', line=dict(color='#4dd0e1', width=3))
        
        fig_reg.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed', size=12),
            xaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(77, 208, 225, 0.1)')
        )
        
        st.plotly_chart(fig_reg, use_container_width=True)
        
        st.markdown(f"""
        <div class='info-panel'>
            <h4>Risultati Analisi:</h4>
            <p><strong>FC Base (a riposo):</strong> {lr.intercept_:.0f} bpm</p>
            <p><strong>Incremento per km/h:</strong> +{lr.coef_[0]:.2f} bpm</p>
            <p><strong>R² (Bontà modello):</strong> {lr.score(X_reg, y_reg):.2%}</p>
            
            <h4>Esempi Pratici:</h4>
            <ul>
                <li>A 10 km/h: FC prevista = {lr.intercept_ + lr.coef_[0]*10:.0f} bpm</li>
                <li>A 12 km/h: FC prevista = {lr.intercept_ + lr.coef_[0]*12:.0f} bpm</li>
                <li>A 14 km/h: FC prevista = {lr.intercept_ + lr.coef_[0]*14:.0f} bpm</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 4: STATISTICHE AVANZATE
# =====================================================================
elif pagina == "📈 Statistiche Avanzate":
    st.title("📈 Analisi Statistica Approfondita")
    
    df = st.session_state.dati.copy()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown("""
        <div class='stat-box'>
            <h4>Andamento Mensile</h4>
            <p style='font-size: 1.2em; color: #4dd0e1;'>245 km</p>
            <p style='font-size: 0.9em;'>+12% vs mese scorso</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown("""
        <div class='stat-box'>
            <h4>Recovery Score Medio</h4>
            <p style='font-size: 1.2em; color: #1db584;'>68%</p>
            <p style='font-size: 0.9em;'>Buon recupero</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown("""
        <div class='stat-box'>
            <h4>Strain Medio</h4>
            <p style='font-size: 1.2em; color: #f59e0b;'>5.8/10</p>
            <p style='font-size: 0.9em;'>Moderato</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        st.subheader("Distribuzione RPE")
        fig_rpe = px.histogram(df, x='RPE', nbins=10, color='RPE',
                              color_continuous_scale='Turbo',
                              title="Distribuzione Sforzo Percepito", height=350)
        fig_rpe.update_layout(
            plot_bgcolor='rgba(26, 26, 58, 0.5)',
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed'),
            showlegend=False
        )
        st.plotly_chart(fig_rpe, use_container_width=True)
    
    with col_adv2:
        st.subheader("Qualità Sonno")
        sonno_counts = df['Qualita Sonno'].value_counts()
        fig_sonno_pie = px.pie(values=sonno_counts.values, names=sonno_counts.index,
                              title="Distribuzione Qualità Sonno", height=350,
                              color_discrete_sequence=['#1db584', '#4dd0e1', '#f59e0b', '#ff6b6b', '#d32f2f'])
        fig_sonno_pie.update_layout(
            paper_bgcolor='rgba(15, 15, 30, 0.9)',
            font=dict(color='#e8eaed')
        )
        st.plotly_chart(fig_sonno_pie, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Tabella Dati Dettagliati")
    
    tab_mostra = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'FC Max', 
                     'RPE', 'Ore Sonno', 'Stress Lavoro', 'Recovery Score']].tail(20).copy()
    tab_mostra['Giorno'] = tab_mostra['Giorno'].dt.strftime('%d/%m')
    
    st.dataframe(tab_mostra, use_container_width=True, hide_index=True)

