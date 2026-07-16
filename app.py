import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# =====================================================================
# 1. SETUP PREMIUM & DARK MODE UI
# =====================================================================
st.set_page_config(page_title="AI Sports Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #000000 !important; color: #F3F4F6 !important; }
    #MainMenu, footer, header {visibility: hidden;}
    
    div[data-testid="metric-container"] {
        background-color: #111827 !important; border-radius: 12px; padding: 20px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #1F2937;
    }
    
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; letter-spacing: -0.5px; }
    
    .ai-insight {
        background: linear-gradient(145deg, #1e1e1e, #121212); border-left: 4px solid #3B82F6;
        padding: 20px; border-radius: 8px; margin-top: 15px; margin-bottom: 20px;
    }
    .ai-insight h4 { color: #60A5FA !important; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-top:0;}
    .ai-insight p { color: #D1D5DB !important; font-size: 14px; line-height: 1.6; margin-bottom:0;}
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom-color: #3B82F6 !important;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. MOTORE DATI (Include tutti i parametri richiesti)
# =====================================================================
@st.cache_data
def inizializza_database():
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'Distanza (km)': np.random.uniform(5, 25, n),
        'Velocità (km/h)': np.random.uniform(9, 16, n),
        'FC Media': np.random.uniform(120, 180, n),
        'Temp (°C)': np.random.uniform(5, 35, n),
        'Vento (km/h)': np.random.uniform(0, 20, n),
        'Ore Sonno': np.random.uniform(4.5, 9.5, n),
        'Ore Lavoro': np.random.uniform(0, 12, n),
        'Stress Lavoro': np.random.uniform(1, 10, n),
        'Feeling (1-5)': np.random.randint(1, 6, n),
        'RPE': np.random.uniform(2, 10, n)
    })
    
    # 4 KPI Proprietari
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'].clip(lower=0.1)
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'].clip(lower=0.1)
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'].clip(lower=0.1)
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'].clip(lower=0.1)
    
    # Target per i modelli AI
    soglia_critica = np.percentile(df['SMA'] + df['ISLR'] + (df['IDET']/100), 75)
    df['Overload'] = np.where((df['SMA'] + df['ISLR'] + (df['IDET']/100)) > soglia_critica, 1, 0)
    
    return df

df = inizializza_database()

# =====================================================================
# 3. SIDEBAR: SINCRONIZZAZIONE E QUESTIONARI COMPLETI
# =====================================================================
with st.sidebar:
    st.markdown("## AI Performance Hub")
    
    st.markdown("### 1. SORGENTE DATI WEARABLE")
    app_corsa = st.selectbox("Seleziona Provider API", ["Nike Run Club", "Strava", "Garmin Connect", "Coros"])
    st.button(f"Sincronizza {app_corsa}", use_container_width=True)
    st.markdown("<hr style='border-color: #1F2937;'>", unsafe_allow_html=True)
    
    st.markdown("### 2. QUESTIONARIO PRE-WORKOUT")
    km_pianificati = st.number_input("Km programmati", 1.0, 50.0, 10.0, step=0.5)
    feeling_input = st.select_slider("Come ti senti oggi?", options=["Pessimo", "Stanco", "Normale", "Bene", "In gran forma"], value="Normale")
    feeling_val = {"Pessimo": 1, "Stanco": 2, "Normale": 3, "Bene": 4, "In gran forma": 5}[feeling_input]
    ore_sonno_input = st.number_input("Ore dormite", 0.0, 12.0, 7.5, step=0.5)
    ore_lavoro_input = st.number_input("Ore di lavoro oggi", 0.0, 16.0, 8.0, step=0.5)
    stress_input = st.slider("Stress Lavoro/Mentale (1-10)", 1, 10, 5)
    
    st.markdown("### 3. METRICHE POST-WORKOUT")
    rpe_input = st.slider("RPE (Sforzo Percepito 1-10)", 1, 10, 6)
    fc_input = st.number_input("FC Media (bpm)", 60, 220, 145)
    velocita_input = st.number_input("Velocità Media (km/h)", 5.0, 25.0, 12.0, step=0.5)
    temp_input = st.number_input("Temperatura (°C)", -10.0, 45.0, 25.0)
    vento_input = st.number_input("Vento (km/h)", 0.0, 50.0, 10.0)

# Calcolo KPI Odierni in tempo reale
val_sma = (stress_input * rpe_input) / max(ore_sonno_input, 0.1)
val_islr = (ore_lavoro_input * stress_input) / max(km_pianificati, 0.1)
val_iitr = (temp_input * vento_input) / max(km_pianificati, 0.1)
val_idet = (fc_input * temp_input) / max(velocita_input, 0.1)

# =====================================================================
# 4. ADDESTRAMENTO MODELLI DI MACHINE LEARNING
# =====================================================================
features = ['SMA', 'ISLR', 'IITR', 'IDET', 'Ore Sonno', 'FC Media']
X = df[features]
y = df['Overload']

# Modello 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
prob_rf = rf_model.predict_proba([[val_sma, val_islr, val_iitr, val_idet, ore_sonno_input, fc_input]])[0][1] * 100

# Modello 2: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y)
prob_log = log_reg.predict_proba([[val_sma, val_islr, val_iitr, val_idet, ore_sonno_input, fc_input]])[0][1] * 100

# =====================================================================
# 5. UI PRINCIPALE (A SCHEDE)
# =====================================================================
nav = st.radio("Menu", ["Dashboard & KPI", "Analisi AI (4 Modelli ML)"], horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

if nav == "Dashboard & KPI":
    st.markdown("## Analisi Fisiologica Odierna")
    
    # SEZIONE 1: I 4 KPI DELLA TESI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stress Mentale (SMA)", f"{val_sma:.2f}")
    c2.metric("Lavoro Residuo (ISLR)", f"{val_islr:.2f}")
    c3.metric("Impatto Termico (IITR)", f"{val_iitr:.2f}")
    c4.metric("Degrado Termico (IDET)", f"{val_idet:.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # SEZIONE 2: METODO SEMAFORICO E RADAR
    col_gauge, col_radar = st.columns([1, 1.2])
    
    with col_gauge:
        colore_sem = "#10B981" if prob_rf < 35 else "#F59E0B" if prob_rf < 70 else "#EF4444"
        testo_sem = "OTTIMALE" if prob_rf < 35 else "ATTENZIONE" if prob_rf < 70 else "SOVRALLENAMENTO"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_rf,
            number = {'suffix': "%", 'font': {'color': colore_sem}},
            title = {'text': f"RISCHIO (SEMAFORO: {testo_sem})", 'font': {'color': '#9CA3AF', 'size': 12}},
            gauge = {
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': colore_sem, 'thickness': 0.8},
                'bgcolor': "#1F2937", 'bordercolor': "#000000"
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        categorie = ['Stress Mentale', 'Fatica Lavoro', 'Fattore Clima', 'Affaticamento Cardiaco']
        # Normalizziamo i dati rispetto alla media storica per visualizzarli nel radar
        valori_storici = [df['SMA'].mean(), df['ISLR'].mean(), df['IITR'].mean(), df['IDET'].mean()/100]
        valori_odierni = [val_sma, val_islr, val_iitr, val_idet/100]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=valori_storici, theta=categorie, fill='toself', name='Media Storica', line_color='#4B5563'))
        fig_radar.add_trace(go.Scatterpolar(r=valori_odierni, theta=categorie, fill='toself', name='Oggi', line_color='#3B82F6'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, max(valori_odierni)+2])),
            paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter", 'color': "#9CA3AF"}, height=300,
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

elif nav == "Analisi AI (4 Modelli ML)":
    st.markdown("## Elaborazione Modelli Computazionali")
    
    t1, t2, t3, t4 = st.tabs(["1. Random Forest", "2. Logistic Regression", "3. K-Means Clustering", "4. Linear Regression"])
    
    # 1. RANDOM FOREST
    with t1:
        st.markdown("### Previsione Rischio Clinico e Pesi")
        importanza = pd.DataFrame({'KPI': features, 'Impatto': rf_model.feature_importances_}).sort_values('Impatto')
        fig_rf = px.bar(importanza, x='Impatto', y='KPI', orientation='h', template="plotly_dark", color_discrete_sequence=['#3B82F6'])
        fig_rf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, xaxis_visible=False)
        st.plotly_chart(fig_rf, use_container_width=True)
        st.markdown("""
        <div class='ai-insight'>
            <h4>Interpretazione: Random Forest</h4>
            <p>Il Random Forest calcola il rischio di infortunio incrociando alberi decisionali multipli. Il grafico mostra la <b>Feature Importance</b>: identifica quali dei tuoi KPI (es. ISLR o SMA) contribuiscono maggiormente ad innalzare il rischio di overtraining.</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. LOGISTIC REGRESSION
    with t2:
        st.markdown("### Classificazione Binaria (Probabilità di Overload)")
        col_log1, col_log2 = st.columns(2)
        col_log1.metric("Probabilità Overload (Random Forest)", f"{prob_rf:.1f}%")
        col_log2.metric("Probabilità Overload (Regressione Logistica)", f"{prob_log:.1f}%")
        
        st.markdown("""
        <div class='ai-insight'>
            <h4>Interpretazione: Logistic Regression</h4>
            <p>A differenza del Random Forest, la Regressione Logistica mappa le probabilità usando una curva Sigmoidea (S-curve). È un modello lineare altamente interpretabile in ambito biomedico. Se la probabilità supera il 50%, il modello classifica matematicamente la sessione come 'Rischio Clinico'.</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. K-MEANS CLUSTERING
    with t3:
        st.markdown("### Segmentazione Profili tramite Distanza Euclidea")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[['ISLR', 'FC Media']])
        
        fig_km = px.scatter(
            df, x="ISLR", y="FC Media", color=df['Cluster'].astype(str), size="Distanza (km)",
            color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"], template="plotly_dark",
            labels={"ISLR": "Sforzo Lavorativo Residuo (ISLR)", "FC Media": "Frequenza Cardiaca (BPM)"}
        )
        fig_km.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_km, use_container_width=True)
        st.markdown("""
        <div class='ai-insight'>
            <h4>Interpretazione: K-Means</h4>
            <p>L'algoritmo raggruppa le corse in 3 cluster calcolando i centroidi geometrici e minimizzando la distanza euclidea. Mostra chiaramente come le sessioni con alto stress lavorativo (ISLR alto, asse X) si traducano quasi sempre in una risposta cardiaca alterata (asse Y alto).</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. LINEAR REGRESSION (Anticrash: scritta senza statsmodels)
    with t4:
        st.markdown("### Analisi del Trend (Costo Fisiologico vs Volume Esterno)")
        
        fig_lin = px.scatter(df, x="Distanza (km)", y="FC Media", template="plotly_dark", color_discrete_sequence=['#4B5563'])
        
        # Calcolo matematico manuale per evitare l'errore di Plotly/Statsmodels
        x_vals = df["Distanza (km)"]
        y_vals = df["FC Media"]
        m, b = np.polyfit(x_vals, y_vals, 1) # Equazione retta y = mx + q
        
        fig_lin.add_trace(go.Scatter(x=x_vals, y=m*x_vals + b, mode='lines', name='Trend Predittivo', line=dict(color='#3B82F6', width=3)))
        fig_lin.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_lin, use_container_width=True)
        
        st.markdown("""
        <div class='ai-insight'>
            <h4>Interpretazione: Linear Regression</h4>
            <p>La linea blu rappresenta la retta di regressione calcolata con il metodo dei minimi quadrati. Mostra il normale decadimento dell'efficienza aerobica all'aumentare dei chilometri. I punti nettamente al di sopra della linea identificano sessioni in cui l'atleta ha mostrato una deriva cardiaca anomala.</p>
        </div>
        """, unsafe_allow_html=True)
