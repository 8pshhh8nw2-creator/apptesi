import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="AI Sports Analytics | Overtraining Prevention", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Design personalizzato per renderla una dashboard moderna ed elegante
st.markdown("""
<style>
    .reportview-container { background: #f8fafc; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        border-left: 5px solid #6366f1;
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
</style>
""", unsafe_allow_html=True)

# Inizializzazione sicura dello stato dell'applicazione
if 'synced' not in st.session_state:
    st.session_state.synced = False
if 'df_storico' not in st.session_state:
    st.session_state.df_storico = pd.DataFrame()

# --- 2. GENERATORE SIMULAZIONE DATI (Come da Tesi) ---
def genera_dati_atleta():
    np.random.seed(42)
    n = 100  # Database più robusto per il cloud
    df = pd.DataFrame({
        'Distanza (km)': np.random.uniform(5, 21, n),
        'Velocità (km/h)': np.random.uniform(10, 15, n),
        'FC Media': np.random.uniform(130, 175, n),
        'Temp (°C)': np.random.uniform(12, 36, n),
        'Vento (km/h)': np.random.uniform(0, 25, n),
        'RPE': np.random.randint(3, 11, n),
        'Ore Sonno': np.random.uniform(4.5, 9, n),
        'Ore Lavoro': np.random.uniform(0, 10, n),
        'Stress Lavoro': np.random.randint(1, 11, n)
    })
    # Calcolo KPI proprietari prevenendo divisioni per zero
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'].replace(0, 0.1)
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'].replace(0, 0.1)
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)'].replace(0, 0.1)
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)'].replace(0, 0.1)
    
    # Soglia di rischio allostatico (75° percentile)
    threshold = np.percentile(df['SMA'] + df['ISLR'] + (df['IDET']/100), 75)
    df['Overload'] = np.where((df['SMA'] + df['ISLR'] + (df['IDET']/100)) > threshold, 1, 0)
    return df

# --- 3. UI PRINCIPALE ---
st.title("⚡ AI Sports Analytics & Overtraining Prevention")
st.markdown("Piattaforma clinico-sportiva per l'analisi del carico allostatico basata sul bilanciamento tra Carico Esterno (Wearable) e Interno (Questionari).")
st.divider()

# --- 4. BARRA LATERALE PER INPUT DATI ---
with st.sidebar:
    st.header("🔄 1. Wearable Sync")
    app_corsa = st.selectbox("Seleziona Piattaforma", ["Strava API", "Garmin Connect", "Polar Flow", "Nike Run Club"])
    
    if st.button(f"Sincronizza Dati", use_container_width=True, type="primary"):
        with st.spinner("Connessione API e calcolo metriche..."):
            time.sleep(1.2)
            st.session_state.df_storico = genera_dati_atleta()
            st.session_state.synced = True
        st.success("✅ Dati Wearable Sincronizzati!")
    
    st.divider()
    st.header("📋 2. Diari Giornalieri")
    
    st.markdown("**Stile di Vita (Pre-Allenamento)**")
    km_pianificati = st.number_input("Km Programmati per oggi", 1.0, 50.0, 10.0, 0.5)
    ore_dormite = st.number_input("Ore di sonno effettive", 0.0, 12.0, 7.5, 0.5)
    ore_lavorate = st.number_input("Ore di lavoro affrontate", 0.0, 16.0, 8.0, 0.5)
    stress_lavoro = st.slider("Stress lavorativo percepito (1-10)", 1, 10, 5)
    
    st.markdown("**Sessione di Corsa (Post-Allenamento)**")
    rpe_odierno = st.slider("RPE (Fatica Percepita - Borg CR10)", 1, 10, 6)
    fc_odierna = st.number_input("Frequenza Cardiaca Media (bpm)", 60, 220, 145)
    velocita_odierna = st.number_input("Velocità Media (km/h)", 5.0, 25.0, 12.0)
    temp_odierna = st.number_input("Temperatura Esterna (°C)", -10.0, 45.0, 25.0)
    vento_odierno = st.number_input("Velocità del Vento (km/h)", 0.0, 50.0, 10.0)

# --- 5. LOGICA DI ELABORAZIONE ---
if not st.session_state.synced:
    st.warning("👈 **Sincronizzazione Richiesta:** Avvia la sincronizzazione dei dati dalla barra laterale per caricare i modelli predittivi.")
else:
    try:
        # Calcolo dei 4 KPI proprietari odierni
        val_sma = (stress_lavoro * rpe_odierno) / max(ore_dormite, 0.1)
        val_islr = (ore_lavorate * stress_lavoro) / max(km_pianificati, 0.1)
        val_iitr = (temp_odierna * vento_odierno) / max(km_pianificati, 0.1)
        val_idet = (fc_odierna * temp_odierna) / max(velocita_odierna, 0.1)

        st.subheader("📊 I tuoi KPI Innovativi (Stato Odierno)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧠 SMA (Stress Mentale)", f"{val_sma:.2f}")
        c2.metric("💼 ISLR (Sforzo Lavoro Residuo)", f"{val_islr:.2f}")
        c3.metric("🌡️ IITR (Impatto Termico)", f"{val_iitr:.2f}")
        c4.metric("🔥 IDET (Degrado Termico)", f"{val_idet:.0f}")

        st.divider()

        # MODELLO ML: RANDOM FOREST
        df = st.session_state.df_storico
        features = ['SMA', 'ISLR', 'IITR', 'IDET', 'Distanza (km)', 'FC Media']
        X = df[features]
        y = df['Overload']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Predizione probabilistica sulla sessione odierna
        sessione_odierna = np.array([[val_sma, val_islr, val_iitr, val_idet, km_pianificati, fc_odierna]])
        prob_overload = rf.predict_proba(sessione_odierna)[0][1] * 100

        st.subheader("🤖 Analisi Predittiva: Algoritmo Random Forest")
        
        col_gauge, col_text = st.columns([1, 1.5])
        with col_gauge:
            # Selezione colore semaforico
            color = "#10B981" if prob_overload < 35 else "#F59E0B" if prob_overload < 70 else "#EF4444"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", 
                value = prob_overload, 
                title = {'text': "Probabilità di Sovrallenamento (%)", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': color},
                    'steps' : [
                        {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.1)"}, 
                        {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.1)"}, 
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_text:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if prob_overload < 35:
                st.success("#### 🟢 SEMAFORO VERDE: Stato di Forma Ottimale\nIl carico allostatico cumulativo è ampiamente sostenibile. Il sistema nervoso autonomo mostra ottime capacità adattive. Puoi procedere con l'intensità programmata.")
            elif prob_overload < 70:
                st.warning("#### 🟡 SEMAFORO GIALLO: Fatica Strutturale Rilevata\nAttenzione: i tuoi KPI (specialmente lo stress lavorativo o la fatica mentale) indicano un recupero parziale. Si raccomanda di ridurre del 15% il volume della sessione odierna.")
            else:
                st.error("#### 🔴 SEMAFORO ROSSO: Rischio Clinico Overtraining\nI modelli predittivi evidenziano un sovraccarico critico causato dalla combinazione di stress psicofisico e degrado termico. L'algoritmo consiglia una sessione di riposo totale o scarico attivo.")

        st.divider()

        # SEZIONI DI DEEP DIVE (I GRAFICI DELLA TUA TESI)
        st.subheader("🔍 Approfondimento dei Modelli Computazionali")
        t1, t2, t3 = st.tabs(["🧩 Clustering K-Means", "📈 Regressione (Trend)", "🎯 Pesi del Modello"])
        
        with t1:
            st.markdown("**Segmentazione Automatica dei Profili di Corsa** (Minimizzazione Distanza Euclidea)")
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[['ISLR', 'FC Media']])
            nomi_cluster = {0: 'Sessioni Rigenerative', 1: 'Allenamenti Qualitativi', 2: 'Corse ad Elevato Stress'}
            df['Cluster Fisiologico'] = df['Cluster'].map(nomi_cluster)
            
            fig_scatter = px.scatter(
                df, x="ISLR", y="FC Media", color="Cluster Fisiologico", 
                size="Distanza (km)", hover_data=['SMA', 'IDET'],
                color_discrete_sequence=["#10B981", "#6366F1", "#EF4444"]
            )
            fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with t2:
            st.markdown("**Studio del Trend: Volume vs Frequenza Cardiaca**")
            fig_reg = px.scatter(df, x="Distanza (km)", y="FC Media", trendline="ols", 
                                 trendline_color_override="#EF4444", opacity=0.7)
            fig_reg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_reg, use_container_width=True)

        with t3:
            st.markdown("**Feature Importance della Random Forest (% di impatto sul sovrallenamento)**")
            importanza = pd.DataFrame({'Variabile': features, 'Importanza (%)': rf.feature_importances_ * 100}).sort_values(by='Importanza (%)')
            fig_bar = px.bar(importanza, x='Importanza (%)', y='Variabile', orientation='h', 
                             color='Importanza (%)', color_continuous_scale="Purples")
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Errore di esecuzione matematica: {e}")
