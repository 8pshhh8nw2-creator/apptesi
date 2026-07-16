import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time

# --- 1. CONFIGURAZIONE PAGINA E DESIGN PROFESSIONALE ---
st.set_page_config(page_title="AI Sports Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: #1e293b; }
    .reportview-container { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border-top: 4px solid #2563eb; }
    h1, h2, h3 { font-weight: 600; color: #0f172a; }
    .st-bb { background-color: transparent; }
    .explanation-box { background-color: #f1f5f9; padding: 15px; border-radius: 8px; border-left: 4px solid #64748b; font-size: 0.95em; line-height: 1.5; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# --- 2. GENERAZIONE DATI STRUTTURATI (Per far funzionare perfettamente l'IA) ---
@st.cache_data
def genera_dati_strutturati():
    np.random.seed(42)
    n = 120 # 4 mesi di dati
    
    # Creiamo 3 tipologie di sessioni distinte per aiutare l'algoritmo di Clustering e Random Forest
    tipi_sessione = np.random.choice(['Rigenerante', 'Qualità', 'Sovraccarico'], n, p=[0.4, 0.4, 0.2])
    
    distanza, velocita, fc_media, ore_sonno, stress_lavoro, rpe = [], [], [], [], [], []
    
    for tipo in tipi_sessione:
        if tipo == 'Rigenerante':
            distanza.append(np.random.uniform(5, 10))
            velocita.append(np.random.uniform(9, 11))
            fc_media.append(np.random.uniform(120, 135))
            ore_sonno.append(np.random.uniform(7.5, 9))
            stress_lavoro.append(np.random.randint(1, 4))
            rpe.append(np.random.randint(2, 5))
        elif tipo == 'Qualità':
            distanza.append(np.random.uniform(10, 21))
            velocita.append(np.random.uniform(12, 16))
            fc_media.append(np.random.uniform(145, 165))
            ore_sonno.append(np.random.uniform(6.5, 8))
            stress_lavoro.append(np.random.randint(3, 7))
            rpe.append(np.random.randint(5, 8))
        else: # Sovraccarico (Dati anomali per l'IA)
            distanza.append(np.random.uniform(15, 25))
            velocita.append(np.random.uniform(10, 13))
            fc_media.append(np.random.uniform(160, 180)) # Battito alto anche se lenti
            ore_sonno.append(np.random.uniform(4, 6)) # Poco sonno
            stress_lavoro.append(np.random.randint(7, 10)) # Molto stress
            rpe.append(np.random.randint(8, 10)) # Fatica estrema

    df = pd.DataFrame({
        'Data': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': distanza, 'Velocità (km/h)': velocita, 'FC Media': fc_media,
        'Temp (°C)': np.random.uniform(10, 28, n), 'Vento (km/h)': np.random.uniform(0, 15, n),
        'RPE': rpe, 'Ore Sonno': ore_sonno, 'Stress Lavoro': stress_lavoro, 'Ore Lavoro': np.random.uniform(6, 10, n)
    })
    
    # Calcolo KPI Proprietari Tesi
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    # Target per il Machine Learning (Soglia oggettiva di fatica sistemica)
    soglia_critica = np.percentile(df['SMA'] + df['ISLR'], 75)
    df['Stato Sovrallenamento'] = np.where((df['SMA'] + df['ISLR']) > soglia_critica, 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati_strutturati()

# --- 3. MENU DI NAVIGAZIONE AZIENDALE ---
st.sidebar.image("https://images.unsplash.com/photo-1552674605-15c8712306f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_container_width=True)
st.sidebar.title("Data-Driven Sports")
st.sidebar.markdown("Piattaforma Analitica")
pagina = st.sidebar.radio("Seleziona Modulo", ["1. Dashboard Inserimento", "2. Storico e Metriche", "3. Modelli AI Predittivi"])

st.sidebar.markdown("---")
st.sidebar.caption("Dispositivo Sincronizzato: Connessione Attiva")

# =====================================================================
# PAGINA 1: DASHBOARD INSERIMENTO E KPI ODIERNI
# =====================================================================
if pagina == "1. Dashboard Inserimento":
    st.title("Monitoraggio Giornaliero Atleta")
    st.markdown("<p style='color: #64748b; font-size: 1.1em;'>Inserimento dati di Carico Esterno (Wearable) e Carico Interno (Soggettivo).</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        st.subheader("Fattori di Stile di Vita")
        ore_sonno = st.number_input("Ore di Sonno (h)", 0.0, 12.0, 7.5, step=0.5)
        ore_lavoro = st.number_input("Ore di Lavoro (h)", 0.0, 16.0, 8.0, step=0.5)
        stress_lavoro = st.slider("Indice Stress Mentale (1-10)", 1, 10, 5)
    
    with col_input2:
        st.subheader("Metriche di Allenamento")
        km_fatti = st.number_input("Volume: Distanza (km)", 1.0, 50.0, 12.0)
        velocita = st.number_input("Intensità: Velocità (km/h)", 5.0, 22.0, 12.5)
        fc_media = st.number_input("Risposta: FC Media (bpm)", 80, 200, 150)
        rpe = st.slider("Sforzo Percepito RPE (1-10)", 1, 10, 6)
        temp = st.number_input("Ambiente: Temperatura (°C)", -5, 40, 20)

    # Calcolo KPI
    sma = (stress_lavoro * rpe) / max(ore_sonno, 0.1)
    islr = (ore_lavoro * stress_lavoro) / max(km_fatti, 0.1)
    idet = (fc_media * temp) / max(velocita, 0.1)

    st.markdown("---")
    st.subheader("Indicatori di Performance (KPI Tesi)")
    st.markdown("<div class='explanation-box'>I KPI sottostanti incrociano lo stress fisiologico con i carichi allostatici (lavoro, sonno, clima) per fornire un quadro completo della readiness dell'atleta.</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("SMA (Stress Mentale Totale)", f"{sma:.2f}")
    c2.metric("ISLR (Sforzo Lavoro Residuo)", f"{islr:.2f}")
    c3.metric("IDET (Degradazione Termica)", f"{idet:.0f}")

# =====================================================================
# PAGINA 2: STORICO E METRICHE
# =====================================================================
elif pagina == "2. Storico e Metriche":
    st.title("Analisi Longitudinale del Carico")
    st.markdown("<p style='color: #64748b;'>Visualizzazione dell'andamento storico tramite grafiche interattive avanzate.</p>", unsafe_allow_html=True)
    df = st.session_state.dati
    
    st.subheader("Trend dei Volumi e Percezione della Fatica")
    st.markdown("<div class='explanation-box'><strong>Come leggere il grafico:</strong> Le barre rappresentano i chilometri percorsi. Il colore indica lo sforzo percepito (RPE). Le barre molto scure indicano allenamenti percepiti come estenuanti.</div>", unsafe_allow_html=True)
    
    fig_vol = px.bar(df, x='Data', y='Distanza (km)', color='RPE', 
                     color_continuous_scale='Blues', template='plotly_white')
    fig_vol.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_vol, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Bilanciamento Fisiologico")
        fig_scatter = px.scatter(df, x="Ore Sonno", y="FC Media", size="Stress Lavoro", 
                                 color="Stato Sovrallenamento", color_continuous_scale=['#3b82f6', '#ef4444'],
                                 template='plotly_white', opacity=0.8)
        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("<div class='explanation-box'>Mostra la correlazione tra ore di riposo e frequenza cardiaca. Le bolle più grandi indicano maggior stress lavorativo; il colore rosso segnala lo stato di sovrallenamento registrato.</div>", unsafe_allow_html=True)

    with col_chart2:
        st.subheader("Distribuzione KPI Medi")
        categorie = ['SMA (Stress)', 'ISLR (Lavoro)', 'IDET (Termico/100)']
        valori = [df['SMA'].mean(), df['ISLR'].mean(), df['IDET'].mean()/100]
        
        fig_radar = go.Figure(data=go.Scatterpolar(r=valori, theta=categorie, fill='toself', 
                                                   marker_color='#2563eb', opacity=0.7))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template='plotly_white')
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("<div class='explanation-box'>L'area del poligono definisce l'impronta di carico allostatico standard dell'atleta. Maggior volume equivale a maggior stress cronico.</div>", unsafe_allow_html=True)

# =====================================================================
# PAGINA 3: INTELLIGENZA ARTIFICIALE PREDITTIVA
# =====================================================================
elif pagina == "3. Modelli AI Predittivi":
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.title("Motori di Intelligenza Artificiale")
    st.markdown("Elaborazione computazionale per la prevenzione clinico-sportiva degli infortuni.")
    
    df = st.session_state.dati
    
    # 1. RANDOM FOREST
    st.markdown("---")
    st.subheader("1. Prevenzione Sindrome da Sovrallenamento (Random Forest)")
    st.markdown("<div class='explanation-box'><strong>Funzionamento:</strong> L'algoritmo Random Forest crea un 'bosco' di alberi decisionali basato su decine di variabili incrociate (Sonno, FC, Stress, Km). Calcola la probabilità matematica che la sessione programmata porti a un collasso prestativo (Overtraining).</div>", unsafe_allow_html=True)
    
    X = df[['Distanza (km)', 'Velocità (km/h)', 'Ore Sonno', 'Stress Lavoro', 'FC Media']]
    y = df['Stato Sovrallenamento']
    
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X, y)
    
    # Predizione simulata su un caso critico per mostrare i grafici
    prob_overload = rf.predict_proba([[15, 11, 5, 8, 165]])[0][1] * 100
    
    col_gauge, col_rf_text = st.columns([1, 1])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_overload, title = {'text': "Rischio Sindrome Overtraining (%)"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1e293b"},
                'steps' : [{'range': [0, 35], 'color': "#cbd5e1"}, {'range': [35, 70], 'color': "#94a3b8"}, {'range': [70, 100], 'color': "#dc2626"}]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_rf_text:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Feature Importance")
        st.write("Quali fattori incidono di più sul rischio infortunio di questo atleta?")
        importanza = pd.DataFrame({'Variabile': X.columns, 'Importanza': rf.feature_importances_}).sort_values('Importanza')
        fig_bar = px.bar(importanza, x='Importanza', y='Variabile', orientation='h', template='plotly_white', color_discrete_sequence=['#2563eb'])
        fig_bar.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. K-MEANS CLUSTERING
    st.markdown("---")
    st.subheader("2. Segmentazione Profili Allenamento (K-Means Clustering)")
    st.markdown("<div class='explanation-box'><strong>Funzionamento:</strong> L'algoritmo non supervisionato calcola la distanza euclidea tra i punti dati per raggruppare automaticamente le sessioni in cluster omogenei. Rivela schemi nascosti tra lo stress mentale (ISLR) e l'impegno fisiologico (FC Media).</div>", unsafe_allow_html=True)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['ISLR', 'FC Media']])
    
    fig_cluster = px.scatter(df, x="ISLR", y="FC Media", color=df['Cluster'].astype(str), size="Distanza (km)",
                             color_discrete_sequence=["#10b981", "#ef4444", "#3b82f6"], template='plotly_white',
                             labels={'color': 'ID Cluster'})
    fig_cluster.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cluster, use_container_width=True)

    # 3. REGRESSIONE LINEARE
    st.markdown("---")
    st.subheader("3. Analisi del Trend (Modelli di Regressione)")
    st.markdown("<div class='explanation-box'><strong>Funzionamento:</strong> La linea di tendenza identifica la relazione matematica lineare (equazione y = mx + q) tra la velocità di corsa e la risposta cardiaca. Deviazioni significative da questa retta (punti molto sopra) possono indicare un imminente stato febbrile o un sovraccarico cardiaco.</div>", unsafe_allow_html=True)
    
    fig_reg = px.scatter(df, x="Velocità (km/h)", y="FC Media", trendline="ols", 
                         trendline_color_override="#dc2626", template='plotly_white', color_discrete_sequence=['#94a3b8'])
    fig_reg.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_reg, use_container_width=True)
