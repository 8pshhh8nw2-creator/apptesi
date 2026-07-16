import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURAZIONE APP ---
st.set_page_config(page_title="RunAI Coach", layout="wide", page_icon="📱")

st.markdown("""
<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e293b; }
    .css-1d391kg { background-color: #f8fafc; } /* Colore sfondo sidebar */
</style>
""", unsafe_allow_html=True)

# --- 2. GENERATORE DATI INTELLIGENTE (Correlazioni Reali per far funzionare l'IA) ---
def genera_dati_intelligenti():
    np.random.seed(42)
    n = 90 # Simuliamo 3 mesi di allenamenti (90 giorni)
    
    # Generiamo dati con una logica realistica
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    vento = np.random.uniform(0, 20, n)
    
    # La FC Media sale se vai veloce, fai tanta distanza e fa caldo
    fc_media = 100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n)
    
    # Lo sforzo percepito (RPE) sale se dormi poco, sei stressato e corri tanto
    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)
    
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': distanza, 'Velocità (km/h)': velocita, 'FC Media': fc_media,
        'Temp (°C)': temp, 'Vento (km/h)': vento, 'RPE': rpe,
        'Ore Sonno': ore_sonno, 'Stress Lavoro': stress_lavoro, 'Ore Lavoro': np.random.uniform(4, 10, n)
    })
    
    # I TUOI KPI (Semplificati per non dare errori)
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    # Condizione di Overtraining (L'IA impara da questo)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati_intelligenti()
    st.session_state.connesso = False

# --- 3. MENU DI NAVIGAZIONE LATERALE (Simula un'app multi-pagina) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=80)
st.sidebar.title("RunAI Coach")
pagina = st.sidebar.radio("Navigazione App", ["🏠 Home & Check-in", "📈 Le mie Statistiche", "🤖 Analisi AI & Previsioni"])

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Connetti Smartwatch (Garmin/Strava)", type="primary"):
    st.session_state.connesso = True
    st.sidebar.success("Dispositivo Connesso! Dati scaricati.")

# =====================================================================
# PAGINA 1: HOME (Inserimento dati semplici e stato di oggi)
# =====================================================================
if pagina == "🏠 Home & Check-in":
    st.title("Ciao Runner! 👋")
    st.write("Compila il tuo diario giornaliero per permettere all'Intelligenza Artificiale di calcolare il tuo stato di forma.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌙 Come hai dormito e lavorato?")
        ore_sonno = st.slider("Ore di sonno stanotte", 2.0, 12.0, 7.0, 0.5)
        ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider("Livello di Stress Mentale/Lavorativo", options=[1,2,3,4,5,6,7,8,9,10], value=5)
    
    with col2:
        st.subheader("🏃‍♂️ Il tuo Allenamento di Oggi")
        km_fatti = st.number_input("Chilometri percorsi (km)", 1.0, 42.0, 10.0)
        velocita = st.number_input("Velocità media (km/h)", 5.0, 20.0, 11.0)
        fc_media = st.number_input("Battito Cardiaco Medio (bpm)", 100, 200, 145)
        rpe = st.slider("Quanta fatica hai fatto? (1 = Zero, 10 = Morto)", 1, 10, 6)
        temp = st.number_input("Temperatura Esterna (°C)", -5, 40, 22)

    st.markdown("---")
    st.subheader("⚡ I Tuoi KPI Personalizzati (Oggi)")
    # Calcolo KPI
    sma = (stress_lavoro * rpe) / ore_sonno
    islr = (ore_lavoro * stress_lavoro) / km_fatti
    idet = (fc_media * temp) / velocita

    c1, c2, c3 = st.columns(3)
    c1.metric("Stress Mentale (SMA)", f"{sma:.1f}", "Equilibrio mente-corpo")
    c2.metric("Sforzo Lavoro (ISLR)", f"{islr:.1f}", "Impatto del lavoro sulla corsa")
    c3.metric("Fatica Fisica (IDET)", f"{idet:.0f}", "Sforzo cardiaco termico")

# =====================================================================
# PAGINA 2: STATISTICHE E GRAFICI (Tanti grafici colorati)
# =====================================================================
elif pagina == "📈 Le mie Statistiche":
    st.title("📊 Il tuo Diario Sportivo (Ultimi 90 giorni)")
    if not st.session_state.connesso:
        st.warning("Clicca su 'Connetti Smartwatch' nella barra laterale per vedere i dati veri!")
    else:
        df = st.session_state.dati
        
        # Grafico 1: Linea del tempo dei KM
        st.subheader("📏 Volumi di Allenamento")
        st.write("Guarda quanti km hai corso ogni giorno. I picchi indicano i tuoi 'lunghi'.")
        fig_line = px.bar(df, x='Giorno', y='Distanza (km)', color='RPE', color_continuous_scale='blues', title="Chilometri percorsi")
        st.plotly_chart(fig_line, use_container_width=True)
        
        col_grafici1, col_grafici2 = st.columns(2)
        
        with col_grafici1:
            # Grafico 2: Dispersione (Correlazione Sonno e Fatica)
            st.subheader("😴 Sonno vs Fatica")
            st.write("Chi dorme meno, fa più fatica a parità di km?")
            fig_scatter = px.scatter(df, x="Ore Sonno", y="RPE", size="Distanza (km)", color="Rischio Infortunio",
                                     color_continuous_scale='reds', title="Meno dormi, più il rischio si accende di rosso")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_grafici2:
            # Grafico 3: Radar Chart (Media dei tuoi KPI)
            st.subheader("🕸️ La tua ragnatela della fatica")
            st.write("Come si bilanciano le tue metriche principali.")
            categorie = ['Stress Mentale (SMA)', 'Sforzo Lavoro (ISLR)', 'Impatto Clima (IITR)', 'Affaticamento Cuore (IDET)']
            valori = [df['SMA'].mean()*10, df['ISLR'].mean()*10, df['IITR'].mean()*10, df['IDET'].mean()/10] # Normalizzati per grafico
            
            fig_radar = go.Figure(data=go.Scatterpolar(r=valori, theta=categorie, fill='toself', marker_color='indigo'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

# =====================================================================
# PAGINA 3: INTELLIGENZA ARTIFICIALE
# =====================================================================
elif pagina == "🤖 Analisi AI & Previsioni":
    st.title("🧠 Il tuo Allenatore Virtuale (Intelligenza Artificiale)")
    st.write("Questa pagina usa modelli matematici complessi (Machine Learning) tradotti in consigli semplici per te.")
    
    df = st.session_state.dati
    
    # --- 1. RANDOM FOREST (Semaforo Rischio) ---
    st.subheader("🚨 Previsione Infortuni e Stanchezza (Random Forest)")
    st.write("L'IA ha studiato i tuoi 90 allenamenti passati. Ora ti dice quanto sei a rischio oggi.")
    
    X = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']]
    y = df['Rischio Infortunio']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Facciamo finta che i dati di oggi siano medi
    prob_infortunio = rf.predict_proba([[10, 6, 8, 160, 8, 10]])[0][1] * 100
    
    c_gauge, c_testo = st.columns([1, 1.5])
    with c_gauge:
        colore = "green" if prob_infortunio < 30 else "orange" if prob_infortunio < 70 else "red"
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_infortunio, title = {'text': "Livello di Pericolo (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': colore},
                     'steps' : [{'range': [0, 30], 'color': "lightgreen"}, {'range': [30, 70], 'color': "moccasin"}, {'range': [70, 100], 'color': "lightcoral"}]}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with c_testo:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if prob_infortunio < 30:
            st.success("✅ **TUTTO OK!** Il tuo corpo ha recuperato bene. Puoi spingere e fare un allenamento intenso.")
        elif prob_infortunio < 70:
            st.warning("⚠️ **ATTENZIONE!** Inizi ad accumulare fatica o stress dal lavoro. Meglio fare una corsetta facile oggi.")
        else:
            st.error("🛑 **FERMATI!** I dati dicono che sei a fortissimo rischio infortunio o sovrallenamento. Stai a casa sul divano!")

    st.markdown("---")

    # --- 2. CLUSTERING (Raggruppamento Allenamenti) ---
    st.subheader("🧩 Come corri di solito? (K-Means Clustering)")
    st.write("L'IA ha raggruppato tutti i tuoi allenamenti in 3 grandi 'Famiglie' in base a quanto il tuo cuore ha faticato rispetto allo stress lavorativo.")
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Tipo Allenamento'] = kmeans.fit_predict(df[['ISLR', 'FC Media']])
    df['Tipo Allenamento'] = df['Tipo Allenamento'].map({0: 'Corsa Facile (Rigenerante)', 1: 'Allenamento Duro', 2: 'Allenamento sotto Stress Mentale'})
    
    fig_cluster = px.scatter(df, x="ISLR", y="FC Media", color="Tipo Allenamento", size="Distanza (km)",
                             color_discrete_sequence=["#10b981", "#ef4444", "#3b82f6"])
    st.plotly_chart(fig_cluster, use_container_width=True)

    # --- 3. REGRESSIONE LINEARE ---
    st.subheader("🔮 Quanto batterà il tuo cuore se corri veloce? (Regressione Lineare)")
    st.write("Questa retta rossa è la previsione dell'AI. Mostra come i battiti del tuo cuore si alzano man mano che aumenti la velocità.")
    
    fig_reg = px.scatter(df, x="Velocità (km/h)", y="FC Media", trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig_reg, use_container_width=True)
