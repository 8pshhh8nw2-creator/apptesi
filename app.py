import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# =====================================================================
# 1. SETUP PREMIUM (WHOOP-STYLE UI)
# =====================================================================
st.set_page_config(page_title="AI Performance Coach", layout="wide", initial_sidebar_state="expanded")

# CSS Personalizzato: Tema Scuro, Font Minimalista, UI a Schede (Cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Reset generale e Sfondo Scuro */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000 !important;
        color: #F3F4F6 !important;
    }
    
    /* Nasconde header e footer di default di Streamlit per un look da vera App */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Stile delle metriche e delle schede (Cards) */
    div[data-testid="metric-container"] {
        background-color: #111827 !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid #1F2937;
    }
    
    /* Testi e Titoli */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    p { color: #9CA3AF !important; }
    
    /* Box delle spiegazioni dell'Intelligenza Artificiale */
    .ai-insight {
        background: linear-gradient(145deg, #1e1e1e, #121212);
        border-left: 4px solid #3B82F6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 30px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.05);
    }
    .ai-insight h4 { color: #60A5FA !important; margin-top: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;}
    .ai-insight p { color: #D1D5DB !important; font-size: 15px; margin-bottom: 0; line-height: 1.6;}
    
    /* Stile dei Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { color: #9CA3AF; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #3B82F6 !important;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. MOTORE DATI SIMULATO (PERFETTO PER I MODELLI AI)
# =====================================================================
@st.cache_data
def inizializza_database():
    np.random.seed(42)
    n = 150 # Dati storici
    
    tipi = np.random.choice(['Recovery', 'Strain', 'Overreach'], n, p=[0.5, 0.35, 0.15])
    
    distanza, velocita, fc_media, ore_sonno, stress, rpe = [], [], [], [], [], []
    
    for t in tipi:
        if t == 'Recovery':
            distanza.append(np.random.uniform(5, 8))
            velocita.append(np.random.uniform(9, 10.5))
            fc_media.append(np.random.uniform(115, 130))
            ore_sonno.append(np.random.uniform(7.5, 9.5))
            stress.append(np.random.uniform(1, 4))
            rpe.append(np.random.uniform(2, 4))
        elif t == 'Strain':
            distanza.append(np.random.uniform(10, 18))
            velocita.append(np.random.uniform(11, 14.5))
            fc_media.append(np.random.uniform(145, 160))
            ore_sonno.append(np.random.uniform(6.5, 8))
            stress.append(np.random.uniform(3, 6))
            rpe.append(np.random.uniform(5, 7))
        else: # Overreach
            distanza.append(np.random.uniform(15, 25))
            velocita.append(np.random.uniform(10, 12.5))
            fc_media.append(np.random.uniform(160, 185)) 
            ore_sonno.append(np.random.uniform(4, 6)) 
            stress.append(np.random.uniform(7, 10))
            rpe.append(np.random.uniform(8, 10))

    df = pd.DataFrame({
        'Distanza (km)': distanza, 'Velocità (km/h)': velocita, 'FC Media': fc_media,
        'Temp (°C)': np.random.uniform(10, 25, n), 'Vento (km/h)': np.random.uniform(0, 10, n),
        'RPE': rpe, 'Ore Sonno': ore_sonno, 'Stress Lavoro': stress, 'Ore Lavoro': np.random.uniform(6, 10, n)
    })
    
    # Formule matematiche sicure (.clip previene divisioni per zero)
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'].clip(lower=0.1)
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'].clip(lower=0.1)
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'].clip(lower=0.1)
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'].clip(lower=0.1)
    
    soglia = np.percentile(df['SMA'] + df['ISLR'], 75)
    df['Overload'] = np.where((df['SMA'] + df['ISLR']) > soglia, 1, 0)
    
    return df

if 'df' not in st.session_state:
    st.session_state.df = inizializza_database()

df = st.session_state.df

# =====================================================================
# 3. SIDEBAR (CONTROLLI E INSERIMENTO)
# =====================================================================
with st.sidebar:
    st.markdown("<h2 style='color: #FFFFFF; font-size: 24px;'>Analisi Biometrica</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; margin-bottom: 30px;'>Piattaforma di calcolo preventivo</p>", unsafe_allow_html=True)
    
    st.markdown("### SINCRONIZZAZIONE")
    st.button("Sincronizza Dispositivo Wearable", use_container_width=True)
    
    st.markdown("<hr style='border-color: #1F2937;'>", unsafe_allow_html=True)
    
    st.markdown("### DIARIO FISIOLOGICO")
    ore_sonno_input = st.number_input("Sonno Registrato (h)", 0.0, 12.0, 7.5, step=0.5)
    stress_input = st.slider("Carico Mentale Lavorativo", 1, 10, 5)
    
    st.markdown("### DATI SESSIONE (CARICO ESTERNO)")
    distanza_input = st.number_input("Distanza (km)", 1.0, 42.0, 12.0, step=0.5)
    velocita_input = st.number_input("Velocità (km/h)", 5.0, 20.0, 11.5, step=0.5)
    fc_input = st.number_input("FC Media (bpm)", 60, 200, 145)
    rpe_input = st.slider("Fatica Percepita (RPE)", 1, 10, 6)

# =====================================================================
# 4. ADDESTRAMENTO MODELLO AI
# =====================================================================
features_usate = ['Distanza (km)', 'Velocità (km/h)', 'Ore Sonno', 'Stress Lavoro', 'FC Media']
X = df[features_usate]
y = df['Overload']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

input_odierno = pd.DataFrame([[distanza_input, velocita_input, ore_sonno_input, stress_input, fc_input]], columns=features_usate)
prob_overload = rf_model.predict_proba(input_odierno)[0][1] * 100

# Metriche in stile WHOOP
recovery_score = 100 - prob_overload
strain_score = min(21.0, (distanza_input * 0.5) + (fc_input * 0.05) + (rpe_input * 0.5) + (stress_input * 0.2))

# =====================================================================
# 5. DASHBOARD PRINCIPALE
# =====================================================================
st.title("Cruscotto Predittivo")
st.markdown("Monitoraggio in tempo reale del bilanciamento tra carico allostatico e capacità di recupero.")

col1, col2, col3 = st.columns(3)

with col1:
    colore_rec = "#10B981" if recovery_score > 66 else "#F59E0B" if recovery_score > 33 else "#EF4444"
    fig_rec = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = recovery_score,
        number = {'suffix': "%", 'font': {'color': colore_rec, 'size': 50}},
        title = {'text': "RECOVERY (RECUPERO)", 'font': {'color': '#9CA3AF', 'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': colore_rec, 'thickness': 0.8},
            'bgcolor': "#1F2937",
            'bordercolor': "#000000"
        }
    ))
    fig_rec.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter"})
    st.plotly_chart(fig_rec, use_container_width=True)

with col2:
    fig_strain = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = strain_score,
        number = {'font': {'color': '#3B82F6', 'size': 50}},
        title = {'text': "DAY STRAIN (SFORZO)", 'font': {'color': '#9CA3AF', 'size': 14}},
        gauge = {
            'axis': {'range': [0, 21], 'visible': False},
            'bar': {'color': "#3B82F6", 'thickness': 0.8},
            'bgcolor': "#1F2937"
        }
    ))
    fig_strain.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter"})
    st.plotly_chart(fig_strain, use_container_width=True)

with col3:
    sma_odierno = (stress_input * rpe_input) / max(ore_sonno_input, 0.1)
    idet_odierno = (fc_input * 20) / max(velocita_input, 0.1) 
    
    st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #9CA3AF; font-size: 14px; text-align: center;'>KPI TESI</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: #FFFFFF; font-size: 36px; text-align: center; margin-bottom: 0;'>{sma_odierno:.1f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #9CA3AF; font-size: 12px; text-align: center;'>Indice Stress Mentale (SMA)</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: #FFFFFF; font-size: 36px; text-align: center; margin-bottom: 0;'>{idet_odierno:.0f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #9CA3AF; font-size: 12px; text-align: center;'>Degrado Fisiologico (IDET)</p>", unsafe_allow_html=True)

if recovery_score > 66:
    testo_ai = "Il tuo sistema nervoso centrale ha assorbito perfettamente l'ultimo allenamento. Sei in piena fase di supercompensazione: i parametri sono ottimali per affrontare carichi elevati."
elif recovery_score > 33:
    testo_ai = "Il corpo sta accumulando fatica strutturale o lo stress mentale lavorativo sta inibendo il recupero completo. Si raccomanda un allenamento di mantenimento."
else:
    testo_ai = "Rischio critico di Overtraining. I modelli rilevano una discrepanza anomala tra le ore di sonno, l'alta frequenza cardiaca e lo stress percepito. Il modello prescrive riposo passivo o recupero attivo leggero."

st.markdown(f"""
<div class='ai-insight'>
    <h4>Analisi AI (Random Forest)</h4>
    <p>{testo_ai}</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 6. ANALISI DEI DATI PROFONDA (TABS)
# =====================================================================
tab1, tab2, tab3 = st.tabs(["IDENTIFICAZIONE ZONE (CLUSTERING)", "TREND FISIOLOGICI (REGRESSIONE)", "PESO DELLE VARIABILI"])

with tab1:
    st.markdown("### Segmentazione Profili Allenamento")
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Ore Sonno', 'FC Media']])
    
    fig_cluster = px.scatter(
        df, x="Ore Sonno", y="FC Media", color=df['Cluster'].astype(str), size="Distanza (km)",
        color_discrete_sequence=["#10B981", "#3B82F6", "#EF4444"], template="plotly_dark",
        labels={"Ore Sonno": "Qualità Riposo (Ore)", "FC Media": "Impatto Cardiaco (BPM)", "color": "Gruppo"}
    )
    fig_cluster.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20))
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.markdown("""
    <div class='ai-insight'>
        <h4>Interpretazione del Cluster (K-Means)</h4>
        <p>L'Intelligenza Artificiale ha letto il tuo intero storico e ha diviso gli allenamenti in 3 gruppi distinti. Analizzando le distanze matematiche tra i punti, il modello rivela come dormire meno (spostandosi a sinistra nel grafico) porti sistematicamente il cuore a battere più velocemente (spostamento in alto), segnalando un costo metabolico più elevato a parità di chilometri.</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Decadimento dell'Efficienza Aerobica")
    
    # GRAFICO DI BASE
    fig_reg = px.scatter(
        df, x="Velocità (km/h)", y="FC Media",
        color_discrete_sequence=['#9CA3AF'], template="plotly_dark",
        labels={"Velocità (km/h)": "Velocità Espressa", "FC Media": "Costo Fisiologico (BPM)"}
    )
    
    # LA SOLUZIONE AL CRASH: Calcolo manuale della retta di regressione usando Numpy
    x_vals = df["Velocità (km/h)"]
    y_vals = df["FC Media"]
    m, b = np.polyfit(x_vals, y_vals, 1) # y = mx + b
    
    # Aggiungo la retta al grafico
    fig_reg.add_trace(go.Scatter(
        x=x_vals, 
        y=m*x_vals + b, 
        mode='lines', 
        name='Trend OLS', 
        line=dict(color='#EF4444', width=3)
    ))
    
    fig_reg.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20))
    st.plotly_chart(fig_reg, use_container_width=True)
    
    st.markdown("""
    <div class='ai-insight'>
        <h4>Interpretazione del Trend (Regressione Lineare)</h4>
        <p>La linea rossa rappresenta il tuo standard di efficienza. Se in una giornata il tuo puntino si posiziona molto al di sopra della retta rossa (battiti alti ma corsa lenta), significa che fattori esterni (stress lavorativo o calore termico) stanno boicottando la tua performance muscolare innescando il sovrallenamento.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Che cosa causa il tuo Overtraining?")
    
    importanza = pd.DataFrame({'Metrica': features_usate, 'Impatto': rf_model.feature_importances_}).sort_values('Impatto', ascending=True)
    
    fig_bar = px.bar(
        importanza, x='Impatto', y='Metrica', orientation='h', 
        template="plotly_dark", color_discrete_sequence=['#3B82F6']
    )
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20), xaxis_visible=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("""
    <div class='ai-insight'>
        <h4>Interpretazione del Rischio (Random Forest)</h4>
        <p>Questo grafico estrae la logica interna dell'algoritmo predittivo. Mostra in ordine di importanza quali variabili di carico interno o esterno influenzano negativamente la tua probabilità di entrare in sindrome da sovrallenamento. Permette al preparatore atletico di sapere esattamente su quale parametro intervenire.</p>
    </div>
    """, unsafe_allow_html=True)
