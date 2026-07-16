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

# CSS Profesionale Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    body {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #f0f4f8;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    h1 {
        color: #00d9ff;
        font-size: 2.8em;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        margin-bottom: 1.2em;
    }
    
    h2 {
        color: #00bfff;
        font-size: 1.8em;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 1em;
        border-bottom: 2px solid #00bfff;
        padding-bottom: 0.7em;
    }
    
    h3 {
        color: #00d9ff;
        font-weight: 600;
    }
    
    h4 {
        color: #00bfff;
        font-weight: 600;
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid #00d9ff;
    }
    
    .stSidebar [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #00d9ff 0%, #00bfff 100%);
        color: #0a0e27;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stSidebar [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00d9ff 100%);
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1a1f3a 0%, #262d4a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
    }
    
    .metric-box-green {
        background: linear-gradient(135deg, #00d9ff 0%, #00bfff 100%);
        color: #0a0e27;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .metric-box-yellow {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #0a0e27;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(251, 191, 36, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .metric-box-red {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #fff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 5px solid #10b981;
        border-right: 1px solid #10b981;
        color: #a7f3d0;
        padding: 24px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-left: 5px solid #f59e0b;
        border-right: 1px solid #f59e0b;
        color: #fcd34d;
        padding: 24px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.15);
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-left: 5px solid #ef4444;
        border-right: 1px solid #ef4444;
        color: #fca5a5;
        padding: 24px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.2) 0%, rgba(0, 191, 255, 0.1) 100%);
        border-left: 5px solid #00d9ff;
        border-right: 1px solid #00d9ff;
        color: #a5f3fc;
        padding: 24px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0, 217, 255, 0.15);
    }
    
    .stAlert {
        border-radius: 12px;
        padding: 16px;
        font-size: 1.05em;
    }
    
    hr {
        border: none;
        border-top: 2px solid rgba(0, 217, 255, 0.3);
        margin: 30px 0;
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
    
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['IITR'] = np.where(df['Distanza (km)'] > 0, (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'], 0)
    df['IDET'] = np.where(df['Velocità (km/h)'] > 0, (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'], 0)
    
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
        <h2 style='margin: 0; color: #00d9ff; font-size: 1.8em;'>RunAI Coach</h2>
        <p style='color: #7dd3fc; font-size: 0.9em; margin-top: 5px;'>Professional Training Analytics</p>
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
            st.cache_data.clear()
            st.session_state.dati = genera_dati_intelligenti()
            st.toast("✓ Dati aggiornati!", icon="↻")

# =====================================================================
# PAGINA 1: ANALISI RISCHIO INFORTUNIO - PAGINA PRINCIPALE
# =====================================================================
if pagina == "Analisi Rischio":
    st.title("Analisi Stato di Forma & Rischio Infortunio")
    
    st.markdown("""
    <div class='info-box'>
        <h4 style='margin-top: 0; color: #00ffff;'>Sistema Intelligente di Previsione Rischio</h4>
        <p>Inserisci i tuoi dati di oggi e l'IA calcolerà il % di rischio infortunio, 
        identificherà i fattori critici, e ti dirà esattamente quanti km e quanto tempo puoi correre in sicurezza.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # INPUT DATI
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("Parametri di Recupero")
        ore_sonno = st.slider("Ore di Sonno (notte scorsa)", 2.0, 12.0, 7.5, 0.5)
        ore_lavoro = st.slider("Ore Lavorate (oggi)", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider("Stress Mentale/Lavoro (1-10)", options=list(range(1, 11)), value=5)
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
            title={'text': "Rischio Infortunio Oggi", 'font': {'color': '#a5f3fc'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#7dd3fc'},
                'bar': {'color': colore},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [25, 60], 'color': 'rgba(251, 191, 36, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ]
            },
            number={'font': {'color': colore, 'size': 50}}
        ))
        
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(26, 31, 58, 0.8)',
            plot_bgcolor='rgba(26, 31, 58, 0.8)',
            font=dict(color='#a5f3fc')
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analisi fattori di rischio
        fattori_rischio = []
        
        if ore_sonno < 6.5:
            fattori_rischio.append(f"❌ Sonno insufficiente ({ore_sonno}h - Ideale: 7.5-9h)")
        if stress_lavoro > 7:
            fattori_rischio.append(f"❌ Stress elevato ({stress_lavoro}/10 - Ideale: < 5)")
        if rpe_previsto > 7.5:
            fattori_rischio.append(f"❌ Sforzo intenso (RPE {rpe_previsto:.1f} - Limite: 7.5)")
        if fc_prevista > 160:
            fattori_rischio.append(f"❌ FC molto alta ({fc_prevista} bpm - Limite: 160)")
        if sma_oggi > 5:
            fattori_rischio.append(f"❌ Equilibrio critico (SMA {sma_oggi:.1f} - Limite: 3.5)")
        if temp_est > 28:
            fattori_rischio.append(f"⚠️ Temperatura elevata ({temp_est}°C - Monitora idratazione)")
        
        st.markdown(f"""
        **Fattori di Rischio Identificati ({len(fattori_rischio)}):**
        """)
        
        if fattori_rischio:
            for fattore in fattori_rischio:
                st.markdown(f"{fattore}")
        else:
            st.markdown("✓ **Nessun fattore critico rilevato - Tutti i parametri ottimali!**")
    
    st.markdown("---")
    st.subheader("Raccomandazioni Personalizzate")
    
    # RACCOMANDAZIONI BASATE SU RISCHIO
    if prob_rischio < 25:
        st.markdown("""
        <div class='success-box'>
            <h3 style='margin-top: 0; color: #10b981;'>✓ STATO OTTIMALE - Basso Rischio</h3>
            <p style='font-size: 1.1em; color: #86efac;'><strong>Probabilità Infortunio: Minima ({:.0f}%)</strong></p>
        </div>
        """.format(prob_rischio), unsafe_allow_html=True)
        
        tempo_allenamento = (km_piano / velocita_piano * 60)
        rpe_max_safe = 8.5
        fc_max_safe = int(fc_prevista * 1.05)
        km_max_safe = km_piano * 1.15
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box-green'>
                <div style='font-size: 0.9em;'>KM MASSIMI SICURI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{km_max_safe:.1f}</div>
                <div style='font-size: 0.85em;'>Puoi arrivare fino a qui</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box-green'>
                <div style='font-size: 0.9em;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{tempo_allenamento:.0f}</div>
                <div style='font-size: 0.85em;'>minuti (a questa velocità)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box-green'>
                <div style='font-size: 0.9em;'>FC MASSIMA SICURA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{fc_max_safe}</div>
                <div style='font-size: 0.85em;'>bpm - Monitora sempre</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='info-box'>
        <h4 style='margin-top: 0;'>Piano di Allenamento di Oggi</h4>
        
        **Parametri Allenamento:**
        - Km consigliati: **{km_piano:.1f} km** (puoi spingere fino a **{km_max_safe:.1f} km**)
        - Tempo di corsa: **{tempo_allenamento:.0f} minuti** a velocità **{velocita_piano:.1f} km/h**
        - Sforzo (RPE): **{rpe_previsto:.1f}**/10 (massimo **8.5**)
        - FC Obiettivo: **{fc_prevista:.0f}} bpm** (massimo **{fc_max_safe} bpm**)
        
        **Tipo di Allenamento:** INTENSO
        - Puoi fare intervalli, ripetute o test di velocità
        - Sessione ideale per migliorare la forma
        
        **Protocollo Esecuzione:**
        - ⏱️ Warm-up: 10 minuti easy pace
        - 💪 Lavoro: Intervalli/ripetute a {velocita_piano:.1f} km/h
        - 🧘 Cool-down: 5 minuti easy + stretching 10 minuti
        
        **Recovery Necessario:**
        - Domani: 1 giorno facile di recupero attivo
        - Idratazione: 500ml acqua ogni 20 minuti
        - Nutrizione: Proteine + carboidrati entro 30 minuti
        </div>
        """, unsafe_allow_html=True)
    
    elif prob_rischio < 60:
        st.markdown(f"""
        <div class='warning-box'>
            <h3 style='margin-top: 0; color: #f59e0b;'>⚠ ATTENZIONE - Rischio Moderato ({prob_rischio:.0f}%)</h3>
            <p style='font-size: 1.1em; color: #fbbf24;'><strong>Puoi Correre ma con Restrizioni Importanti</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        tempo_allenamento = (km_piano / velocita_piano * 60) * 0.8
        km_max_safe = km_piano * 0.9
        velocita_ridotta = velocita_piano * 0.85
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box-yellow'>
                <div style='font-size: 0.9em;'>KM MASSIMI SICURI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{km_max_safe:.1f}</div>
                <div style='font-size: 0.85em;'>Ridotto del 10% per sicurezza</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box-yellow'>
                <div style='font-size: 0.9em;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{tempo_allenamento:.0f}</div>
                <div style='font-size: 0.85em;'>minuti (ridotto 20%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box-yellow'>
                <div style='font-size: 0.9em;'>VELOCITÀ CONSIGLIATA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>{velocita_ridotta:.1f}</div>
                <div style='font-size: 0.85em;'>km/h (easy pace)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='warning-box'>
        <h4 style='margin-top: 0;'>Piano di Recupero Attivo</h4>
        
        **Parametri Ridotti per Sicurezza:**
        - Km consigliati: **{km_max_safe:.1f} km** (massimo - preferibilmente meno)
        - Tempo di corsa: **{tempo_allenamento:.0f} minuti**
        - Velocità: **{velocita_ridotta:.1f}} km/h** (easy pace - ritmo conversativo)
        - Sforzo (RPE): **4-5**/10 (BASSO)
        - FC Obiettivo: **{int(fc_prevista * 0.85)} - {int(fc_prevista * 0.90)} bpm**
        
        **Tipo di Allenamento:** RECUPERO ATTIVO
        - Solo corsa rigenerante a bassa intensità
        
        **❌ COSA EVITARE ASSOLUTAMENTE:**
        - ❌ Niente intervalli
        - ❌ Niente ripetute ad alta velocità
        - ❌ Niente sprint
        - ❌ Niente salite impegnative
        - ❌ Niente allenamenti intensi
        
        **Protocollo Sicuro:**
        - ⏱️ Warm-up: 5 minuti easy
        - 🏃 Corsa: Tutta a ritmo basso e costante
        - 🧘 Cool-down: 5 minuti easy + stretching 15 minuti
        - 😴 Recupero: Riposo totale il giorno dopo
        
        **Priorità per le Prossime 24-48 Ore:**
        - ✓ Dormi 8+ ore stasera (non negoziabile)
        - ✓ Mantieni stress basso domani
        - ✓ Ricarica idratazione completa (3+ litri acqua)
        - ✓ Nutrizione ricca di proteine e carboidrati
        - ✓ Niente allenamenti intensi per 48 ore
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class='danger-box'>
            <h3 style='margin-top: 0; color: #ef4444;'>🛑 RISCHIO CRITICO ({prob_rischio:.0f}%)</h3>
            <p style='font-size: 1.1em; color: #fca5a5;'><strong>ALTISSIMO RISCHIO INFORTUNIO - RIPOSO OBBLIGATORIO</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col_ric1, col_ric2, col_ric3 = st.columns(3)
        
        with col_ric1:
            st.markdown(f"""
            <div class='metric-box-red'>
                <div style='font-size: 0.9em;'>KM CONSIGLIATI</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>0</div>
                <div style='font-size: 0.85em;'>RIPOSO TOTALE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric2:
            st.markdown(f"""
            <div class='metric-box-red'>
                <div style='font-size: 0.9em;'>ATTIVITÀ CONSIGLIATA</div>
                <div style='font-size: 2.8em; font-weight: 700; margin: 10px 0;'>-</div>
                <div style='font-size: 0.85em;'>Niente allenamento</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ric3:
            st.markdown(f"""
            <div class='metric-box-red'>
                <div style='font-size: 0.9em;'>PRIORITÀ</div>
                <div style='font-size: 2em; font-weight: 700; margin: 10px 0;'>RIPOSO</div>
                <div style='font-size: 0.85em;'>24-48 ore minimo</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='danger-box'>
        <h4 style='margin-top: 0;'>PROTOCOLLO EMERGENZA - Riposo Immediato</h4>
        
        **SITUAZIONE CRITICA:**
        Il tuo corpo è in serio pericolo di infortunio grave o sovrallenamento estremo.
        Rischio: **{prob_rischio:.0f}%** - INACCETTABILE
        
        **🛑 AZIONI IMMEDIATE (OGGI - Non negoziabile):**
        - ❌ NON CORRERE ASSOLUTAMENTE
        - ✓ Stai a casa e riposa completamente
        - ✓ Niente allenamento di alcun tipo
        - ✓ Solo camminate leggerissime (max 10 minuti a passo lento)
        - ✓ Stretching delicato 10 minuti
        - ✓ Respira profondamente per 5 minuti (riduce cortisolo)
        - ✓ Bevi 3+ litri di acqua durante il giorno
        
        **🌙 PRIORITÀ RIPRESA (NOTTURNA):**
        - ✓ Dormi 9 ore stasera (non negoziabile)
        - ✓ Cena leggera 2h prima di letto
        - ✓ Camera fresca (18-20°C)
        - ✓ Niente schermo 30 minuti prima di dormire
        
        **📋 DOMANI - Se il rischio scende a < 50%:**
        - Solo camminate 15-20 minuti facili
        - Ripresa allenamenti solo se TUTTI i parametri migliorano
        - Aspetta almeno 48 ore prima di qualsiasi sforzo
        
        **⚠️ SEGNALI DI ALLARME - Consulta Medico Immediatamente:**
        - Dolore persistente o acuto
        - Gonfiore, rigidità muscolare inspiegata
        - Impossibilità di muoverti normalmente
        - Febbre > 37.5°C
        - Stanchezza estrema anche a riposo
        - Confusione mentale, irritabilità severa
        
        **🏥 Contatta il Medico se:**
        - Il rischio rimane > 70% per 48+ ore
        - Sviluppi sintomi fisici strani
        - Non riesci a recuperare sonno
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Dashboard KPI Odierni")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        sma_valore = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0
        sma_stato = "CRITICO" if sma_valore > 5 else "ALTO" if sma_valore > 3.5 else "✓ OK"
        colore_sma = "red" if sma_valore > 5 else "orange" if sma_valore > 3.5 else "green"
        st.metric("SMA (Equilibrio)", f"{sma_valore:.1f}", sma_stato)
    
    with col_kpi2:
        islr_valore = (ore_lavoro * stress_lavoro) / km_piano if km_piano > 0 else 0
        islr_stato = "CRITICO" if islr_valore > 5 else "ALTO" if islr_valore > 3 else "✓ OK"
        st.metric("ISLR (Lavoro)", f"{islr_valore:.1f}", islr_stato)
    
    with col_kpi3:
        fc_stato = "CRITICO" if fc_prevista > 160 else "ALTO" if fc_prevista > 150 else "✓ OK"
        st.metric("FC Prevista", f"{fc_prevista:.0f} bpm", fc_stato)
    
    with col_kpi4:
        sonno_stato = "CRITICO" if ore_sonno < 6.5 else "ALTO" if ore_sonno < 7.5 else "✓ OK"
        st.metric("Qualità Sonno", f"{ore_sonno:.1f}h", sonno_stato)

# =====================================================================
# PAGINA 2: DASHBOARD
# =====================================================================
elif pagina == "Dashboard":
    st.title("Dashboard - Riepilogo Generale")
    
    if not st.session_state.connesso:
        st.warning("Connetti un dispositivo per visualizzare i dati storici.")
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
                color_continuous_scale=['#0a0e27', '#00d9ff', '#00bfff'],
                title="Distanza Giornaliera e Sforzo",
                height=380
            )
            
            fig_volume.update_layout(
                plot_bgcolor='rgba(26, 31, 58, 0.5)',
                paper_bgcolor='rgba(10, 14, 39, 0.8)',
                font=dict(color='#a5f3fc', size=10),
                hovermode='x unified',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0, 217, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0, 217, 255, 0.1)')
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
                color_continuous_scale=['#00d9ff', '#ef4444'],
                title="Correlazione Sonno-Sforzo",
                height=380,
                opacity=0.7
            )
            
            fig_scatter.update_layout(
                plot_bgcolor='rgba(26, 31, 58, 0.5)',
                paper_bgcolor='rgba(10, 14, 39, 0.8)',
                font=dict(color='#a5f3fc', size=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================================
# PAGINA 3: STATISTICHE
# =====================================================================
elif pagina == "Statistiche":
    st.title("Analisi Dettagliata - Ultimi 90 Giorni")
    
    if not st.session_state.connesso:
        st.warning("Connetti un dispositivo per visualizzare i dati storici.")
    else:
        df = st.session_state.dati.copy()
        
        tab1, tab2, tab3 = st.tabs(["Velocità vs FC", "Temperatura", "Tabella Dati"])
        
        with tab1:
            st.subheader("Analisi Velocità vs Frequenza Cardiaca")
            
            X_plot = df['Velocità (km/h)'].values.reshape(-1, 1)
            y_plot = df['FC Media'].values
            
            lr = LinearRegression()
            lr.fit(X_plot, y_plot)
            y_pred = lr.predict(X_plot)
            
            df['FC_Trend'] = y_pred
            
            fig_reg = px.scatter(
                df,
                x="Velocità (km/h)",
                y="FC Media",
                size="RPE",
                color="Ore Sonno",
                color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                title="Velocità vs FC - Analisi Correlazione",
                height=450,
                opacity=0.6,
                labels={'Velocità (km/h)': 'Velocità Media (km/h)', 'FC Media': 'Frequenza Cardiaca (bpm)'}
            )
            
            fig_reg.add_scatter(
                x=df['Velocità (km/h)'],
                y=df['FC_Trend'],
                mode='lines',
                name='Trend',
                line=dict(color='#ef4444', width=3, dash='dash')
            )
            
            fig_reg.update_layout(
                plot_bgcolor='rgba(26, 31, 58, 0.5)',
                paper_bgcolor='rgba(10, 14, 39, 0.8)',
                font=dict(color='#a5f3fc', size=11),
                xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)')
            )
            
            st.plotly_chart(fig_reg, use_container_width=True)
            
            st.markdown(f"""
            **Interpretazione:**
            - Coefficiente: Per ogni +1 km/h, FC aumenta di ~{lr.coef_[0]:.2f} bpm
            - Intercetta: FC di base ~{lr.intercept_:.0f} bpm
            - R² Score: {lr.score(X_plot, y_plot):.2%}
            """)
        
        with tab2:
            st.subheader("Effetto Temperatura su Frequenza Cardiaca")
            
            fig_heat = px.scatter(
                df,
                x="Temp (°C)",
                y="FC Media",
                size="Distanza (km)",
                color="Velocità (km/h)",
                color_continuous_scale='Viridis',
                title="Temperatura vs FC",
                height=450,
                opacity=0.7,
                labels={'Temp (°C)': 'Temperatura (°C)', 'FC Media': 'Frequenza Cardiaca (bpm)'}
            )
            
            fig_heat.update_layout(
                plot_bgcolor='rgba(26, 31, 58, 0.5)',
                paper_bgcolor='rgba(10, 14, 39, 0.8)',
                font=dict(color='#a5f3fc', size=11),
                xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)')
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

