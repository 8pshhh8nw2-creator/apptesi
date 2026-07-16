import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURAZIONE APP ---
st.set_page_config(page_title="RunAI Coach - Professional Analytics", layout="wide", page_icon="🏃")

# CSS Profesionale Avanzato
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    h1 {
        color: #06b6d4;
        font-size: 2.8em;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(6, 182, 212, 0.3);
        margin-bottom: 1.2em;
        background: linear-gradient(90deg, #06b6d4, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #0ea5e9;
        font-size: 1.8em;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 1em;
        border-bottom: 2px solid #0ea5e9;
        padding-bottom: 0.7em;
    }
    
    h3 {
        color: #06b6d4;
        font-weight: 600;
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #0ea5e9;
    }
    
    .stSidebar [data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #06b6d4 0%, #0ea5e9 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #0ea5e9;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.15);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(6, 182, 212, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 4px solid #10b981;
        color: #d1fae5;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-left: 4px solid #f59e0b;
        color: #fef3c7;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-left: 4px solid #ef4444;
        color: #fee2e2;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(3, 102, 214, 0.1) 100%);
        border-left: 4px solid #06b6d4;
        color: #cffafe;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .stAlert {
        border-radius: 12px;
        padding: 16px;
        font-size: 1.05em;
        border-left: 4px solid #06b6d4;
        background: rgba(6, 182, 212, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. GENERATORE DATI INTELLIGENTE ---
@st.cache_data
def genera_dati_intelligenti():
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
        'Temp (°C)': np.round(temp, 1),
        'Vento (km/h)': np.round(vento, 1),
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1)
    })
    
    # KPI
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['IITR'] = np.where(df['Distanza (km)'] > 0, (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'], 0)
    df['IDET'] = np.where(df['Velocità (km/h)'] > 0, (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'], 0)
    
    # Overtraining Risk
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati_intelligenti()
    st.session_state.connesso = False

# --- 3. NAVIGAZIONE ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <div style='font-size: 4em; margin-bottom: 10px;'>🏃‍♂️</div>
        <h2 style='margin: 0; color: #06b6d4; font-size: 1.8em;'>RunAI Coach</h2>
        <p style='color: #94a3b8; font-size: 0.9em; margin-top: 5px;'>Professional Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pagina = st.sidebar.radio(
        "Navigazione",
        ["Analisi Rischio", "Dashboard", "Statistiche"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Connetti", use_container_width=True):
            st.session_state.connesso = True
            st.toast("✓ Dispositivo connesso!", icon="✓")
    
    with col2:
        if st.button("Aggiorna", use_container_width=True):
            st.session_state.dati = genera_dati_intelligenti()
            st.toast("✓ Dati aggiornati!", icon="↻")

# =====================================================================
# PAGINA 1: ANALISI RISCHIO INFORTUNIO - PAGINA PRINCIPALE
# =====================================================================
if pagina == "Analisi Rischio":
    st.title("Analisi Stato di Forma & Rischio Infortunio")
    
    st.markdown("""
    <div class='info-box'>
        <h4 style='margin-top: 0;'>Sistema Intelligente di Previsione</h4>
        <p>Inserisci i tuoi dati di oggi e l'IA ti dirà il % di rischio infortunio, 
        i fattori critici, e esattamente quanti km e quanto tempo puoi correre in sicurezza.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # INPUT DATI
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("Parametri di Recupero")
        ore_sonno = st.slider("Ore di Sonno (notte scorsa)", 2.0, 12.0, 7.5, 0.5)
        ore_lavoro = st.slider("Ore Lavorate (oggi)", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider("Stress Mentale/Lavoro", options=list(range(1, 11)), value=5)
        temp_est = st.number_input("Temperatura Esterna (°C)", -5, 40, 22)
    
    with col_input2:
        st.subheader("Parametri di Allenamento")
        km_piano = st.number_input("Km che vuoi fare", 1.0, 42.0, 10.0)
        velocita_piano = st.number_input("Velocità Media Prevista (km/h)", 5.0, 20.0, 11.0)
        fc_prevista = st.number_input("FC Media Prevista (bpm)", 100, 200, 150)
    
    st.markdown("---")
    
    # CALCOLO MODELLO AI
    df_train = st.session_state.dati
    
    X_train = df_train[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
    y_train = df_train['Rischio Infortunio']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled, y_train)
    
    # Calcolo KPI odierni
    rpe_previsto = (km_piano * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe_previsto = np.clip(rpe_previsto, 1, 10)
    
    sma_oggi = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0
    
    # Predizione rischio
    scenario = scaler.transform([[km_piano, ore_sonno, stress_lavoro, fc_prevista, rpe_previsto, sma_oggi]])
    prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Risultati Analisi IA")
    
    # GAUGE RISCHIO
    col_gauge, col_info = st.columns([1.2, 1.8])
    
    with col_gauge:
        if prob_rischio < 25:
            colore = "#10b981"
        elif prob_rischio < 60:
            colore = "#f59e0b"
        else:
            colore = "#ef4444"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_rischio,
            title={'text': "Rischio Infortunio Oggi", 'font': {'color': '#e2e8f0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                'bar': {'color': colore},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [25, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ]
            },
            number={'font': {'color': '#e2e8f0', 'size': 50}}
        ))
        
        fig_gauge.update_layout(
            height=380,
            paper_bgcolor='rgba(15, 23, 42, 0.8)',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analisi fattori di rischio
        fattori_rischio = []
        
        if ore_sonno < 6.5:
            fattori_rischio.append(f"Sonno insufficiente ({ore_sonno}h)")
        if stress_lavoro > 7:
            fattori_rischio.append(f"Stress elevato ({stress_lavoro}/10)")
        if rpe_previsto > 7.5:
            fattori_rischio.append(f"Sforzo intenso (RPE {rpe_previsto:.1f})")
        if fc_prevista > 160:
            fattori_rischio.append(f"FC molto alta ({fc_prevista} bpm)")
        if sma_oggi > 5:
            fattori_rischio.append(f"Equilibrio critico (SMA {sma_oggi:.1f})")
        if temp_est > 28:
            fattori_rischio.append(f"Temperatura elevata ({temp_est}°C)")
        
        if not fattori_rischio:
            fattori_rischio.append("Nessun fattore critico rilevato")
        
        st.markdown(f"""
        **Fattori di Rischio Identificati:**
        """)
        
        for fattore in fattori_rischio:
            st.markdown(f"• {fattore}")
    
    st.markdown("---")
    st.subheader("Raccomandazioni Personalizzate")
    
    # RACCOMANDAZIONI BASATE SU RISCHIO
    if prob_rischio < 25:
        st.markdown("""
        <div class='success-box'>
            <h3 style='margin-top: 0;'>✓ STATO OTTIMALE - Basso Rischio</h3>
            <p style='font-size: 1.1em;'><strong>Probabilità Infortunio: Minima</strong></p>
            
            <p>Tutti gli indicatori sono verdi. Puoi allenarsi normalmente senza restrizioni.</p>
            
            <h4>PIANO DI ALLENAMENTO CONSIGLIATO:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Calcoli per allenamento
        tempo_allenamento = (km_piano / velocita_piano * 60)  # in minuti
        
        rpe_max_safe = 8.5
        fc_max_safe = int(fc_prevista * 1.05)  # Max +5%
        km_max_safe = km_piano * 1.15  # Max +15%
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM MASSIMI SICURI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{km_max_safe:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>Puoi arrivare fino a qui</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 0.85em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{tempo_allenamento:.0f}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>minuti (a questa velocità)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 0.85em; opacity: 0.9;'>FC MASSIMA SICURA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{fc_max_safe}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>bpm - Monitora sempre</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Come Correre Oggi:**
        - Km consigliati: {km_piano:.1f} km (puoi spingere fino a {km_max_safe:.1f} km)
        - Tempo di corsa: {tempo_allenamento:.0f} minuti a velocità {velocita_piano:.1f} km/h
        - Sforzo (RPE): {rpe_previsto:.1f}/10 (massimo 8.5)
        - FC Obiettivo: {fc_prevista:.0f} bpm (massimo {fc_max_safe} bpm)
        - Tipo allenamento: INTENSO - Puoi fare intervalli o ripetute
        - Recovery necessario: 1 giorno facile dopo
        
        **Protocollo:**
        - Warm-up: 10 min easy pace
        - Lavoro: Intervalli/ripetute a {velocita_piano:.1f} km/h
        - Cool-down: 5 min easy + stretching 10 min
        """)
    
    elif prob_rischio < 60:
        st.markdown(f"""
        <div class='warning-box'>
            <h3 style='margin-top: 0;'>⚠ ATTENZIONE - Rischio Moderato ({prob_rischio:.0f}%)</h3>
            <p style='font-size: 1.1em;'><strong>Puoi Correre ma con Restrizioni</strong></p>
            
            <p>Stai accumulando fatica. Non è critico ma devi essere prudente.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calcoli ridotti
        tempo_allenamento = (km_piano / velocita_piano * 60) * 0.8  # ridotto 20%
        km_max_safe = km_piano * 0.9  # ridotto 10%
        velocita_ridotta = velocita_piano * 0.85  # ridotta 15%
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM MASSIMI SICURI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{km_max_safe:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>Ridotto del 10% per sicurezza</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{tempo_allenamento:.0f}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>minuti (ridotto 20%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>VELOCITÀ CONSIGLIATA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{velocita_ridotta:.1f}</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>km/h (easy pace)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Come Correre Oggi:**
        - Km consigliati: {km_max_safe:.1f} km (massimo, di solito meno)
        - Tempo di corsa: {tempo_allenamento:.0f} minuti
        - Velocità: {velocita_ridotta:.1f} km/h (easy pace - a ritmo conversativo)
        - Sforzo (RPE): 4-5/10 (BASSO)
        - FC Obiettivo: {int(fc_prevista * 0.85)} - {int(fc_prevista * 0.90)} bpm
        - Tipo allenamento: RECUPERO ATTIVO
        
        **Cosa EVITARE:**
        - ❌ Niente intervalli
        - ❌ Niente ripetute
        - ❌ Niente sprint
        - ❌ Niente salite impegnative
        
        **Protocollo Sicuro:**
        - Warm-up: 5 min easy
        - Corsa: Tutta a ritmo basso e costante
        - Cool-down: 5 min easy + stretching 15 min
        - Recupero: Riposo totale il giorno dopo
        
        **Priorità Domani:**
        - Dormi 8+ ore stasera
        - Mantieni stress basso domani
        - Ricarica idratazione e nutrizione
        """)
    
    else:
        st.markdown(f"""
        <div class='danger-box'>
            <h3 style='margin-top: 0;'>🛑 RISCHIO CRITICO ({prob_rischio:.0f}%)</h3>
            <p style='font-size: 1.1em;'><strong>ALTISSIMO RISCHIO INFORTUNIO</strong></p>
            
            <p>I dati indicano una situazione critica di sovrallenamento.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>KM CONSIGLIATI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>0</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>RIPOSO TOTALE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>ATTIVITÀ CONSIGLIATA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>-</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>Niente allenamento</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);'>
                <div style='font-size: 0.85em; opacity: 0.9;'>PRIORITÀ</div>
                <div style='font-size: 2em; font-weight: 700; margin: 10px 0;'>RIPOSO</div>
                <div style='font-size: 0.8em; opacity: 0.8;'>24-48 ore minimo</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **COSA FARE OGGI:**
        
        🛑 **RIPOSO TOTALE - NON CORRERE**
        - Il tuo corpo è in pericolo di infortunio grave
        - Il sistema nervoso è sovrastimolato
        - Il rischio è {prob_rischio:.0f}% - INACCETTABILE
        
        **Azioni Immediate (Oggi):**
        - ✓ Stai a casa e riposa
        - ✓ Niente allenamento di alcun tipo
        - ✓ Camminate leggere max 20 minuti a passo lento
        - ✓ Stretching delicato 15 minuti
        - ✓ Respira profondamente (riduce stress)
        - ✓ Bevi 3+ litri di acqua
        
        **Priorità Ripresa:**
        - ✓ Dormi 9 ore stasera (non negoziabile)
        - ✓ Riduci stress lavoro domani
        - ✓ Mangia protein e carboidrati (recupero)
        - ✓ Niente allenamento per 24-48 ore
        
        **Domani (Se il rischio scende < 50%):**
        - Solo camminate 20 min facili
        - Ripresa graduale solo se tutti i parametri migliorano
        
        **⚠ Segnali di Allarme - Consulta Medico:**
        - Dolore persistente
        - Gonfiore o rigidità
        - Impossibilità di muoverti normalmente
        - Febbre o malessere generale
        """)
    
    st.markdown("---")
    st.subheader("Fattori Principali del Rischio")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        sma_valore = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0
        sma_stato = "Critico" if sma_valore > 5 else "Alto" if sma_valore > 3.5 else "OK"
        st.metric("SMA (Equilibrio Mente)", f"{sma_valore:.1f}", sma_stato)
    
    with col_kpi2:
        islr_valore = (ore_lavoro * stress_lavoro) / km_piano if km_piano > 0 else 0
        islr_stato = "Critico" if islr_valore > 5 else "Alto" if islr_valore > 3 else "OK"
        st.metric("ISLR (Impatto Lavoro)", f"{islr_valore:.1f}", islr_stato)
    
    with col_kpi3:
        fc_stato = "Critico" if fc_prevista > 160 else "Alto" if fc_prevista > 150 else "OK"
        st.metric("Frequenza Cardiaca", f"{fc_prevista:.0f} bpm", fc_stato)
    
    with col_kpi4:
        sonno_stato = "Critico" if ore_sonno < 6.5 else "Alto" if ore_sonno < 7.5 else "OK"
        st.metric("Qualità Sonno", f"{ore_sonno:.1f}h", sonno_stato)

# =====================================================================
# PAGINA 2: DASHBOARD
# =====================================================================
elif pagina == "Dashboard":
    st.title("Dashboard - Riepilogo Generale")
    
    if not st.session_state.connesso:
        st.warning("Connetti un dispositivo per visualizzare i dati.")
    else:
        df = st.session_state.dati.copy()
        
        st.subheader("Statistiche Ultimi 90 Giorni")
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        
        col_stat1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f}", "ultimi 90gg")
        col_stat2.metric("Media Velocità", f"{df['Velocità (km/h)'].mean():.1f}", "km/h")
        col_stat3.metric("FC Media", f"{df['FC Media'].mean():.0f}", "bpm")
        col_stat4.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f}", "ore")
        col_stat5.metric("Sessioni", len(df), "allenamenti")
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Timeline Volumi")
            
            fig_volume = px.bar(
                df,
                x='Giorno',
                y='Distanza (km)',
                color='RPE',
                color_continuous_scale=['#0f172a', '#06b6d4', '#0ea5e9'],
                title="Distanza Giornaliera",
                height=380
            )
            
            fig_volume.update_layout(
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='#e2e8f0', size=10),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col_g2:
            st.subheader("Sonno vs Sforzo")
            
            fig_scatter = px.scatter(
                df,
                x="Ore Sonno",
                y="RPE",
                size="Distanza (km)",
                color="Rischio Infortunio",
                color_continuous_scale=['#06b6d4', '#ef4444'],
                title="Correlazione Sonno-RPE",
                height=380,
                opacity=0.7
            )
            
            fig_scatter.update_layout(
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='#e2e8f0', size=10)
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================================
# PAGINA 3: STATISTICHE
# =====================================================================
elif pagina == "Statistiche":
    st.title("Analisi Dettagliata - Ultimi 90 Giorni")
    
    if not st.session_state.connesso:
        st.warning("Connetti un dispositivo per visualizzare i dati.")
    else:
        df = st.session_state.dati.copy()
        
        tab1, tab2, tab3 = st.tabs(["Trend", "Correlazioni", "Tabella Dati"])
        
        with tab1:
            st.subheader("Trend Allenamenti")
            
            fig_reg = px.scatter(
                df,
                x="Velocità (km/h)",
                y="FC Media",
                trendline="ols",
                trendline_color_override="#ef4444",
                size="RPE",
                color="Ore Sonno",
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                title="Velocità vs FC + Trendline",
                height=450,
                opacity=0.6
            )
            
            fig_reg.update_layout(
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='#e2e8f0', size=11)
            )
            
            st.plotly_chart(fig_reg, use_container_width=True)
        
        with tab2:
            st.subheader("Correlazioni Principali")
            
            fig_heat = px.scatter(
                df,
                x="Temp (°C)",
                y="FC Media",
                size="Distanza (km)",
                color="Velocità (km/h)",
                color_continuous_scale='Viridis',
                title="Effetto Temperatura su FC",
                height=450,
                opacity=0.7
            )
            
            fig_heat.update_layout(
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                font=dict(color='#e2e8f0', size=11)
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
        
        with tab3:
            st.subheader("Ultimi 15 Allenamenti")
            
            tabella = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'SMA', 'ISLR']].tail(15).copy()
            tabella['Giorno'] = tabella['Giorno'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(
                tabella,
                use_container_width=True,
                hide_index=True
            )

