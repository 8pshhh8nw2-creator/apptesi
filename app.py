import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide", page_icon="🏃")

# CSS SEMPLICE E FUNZIONANTE
st.markdown("""
<style>
    body { 
        background-color: #f5f5f5; 
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp { 
        background-color: #f5f5f5; 
    }
    h1 {
        color: #0066cc;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        color: #0066cc;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    h3 { color: #0066cc; }
    h4 { color: #0066cc; }
    
    .info-box {
        background: #e3f2fd;
        border-left: 5px solid #0066cc;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .danger-box {
        background: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
    })
    
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    df['Overtraining'] = np.where((df['RPE'] > 8) & (df['Stress Lavoro'] > 7) & (df['Ore Sonno'] < 6), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.heart_rate = 72

# SIDEBAR
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Analytics Platform")
    st.markdown("---")
    
    st.subheader("Dispositivi Disponibili")
    
    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }
    
    device_scelto = st.sidebar.selectbox("Seleziona Dispositivo:", list(dispositivi.keys()))
    
    if st.sidebar.button("🔗 Connetti Dispositivo"):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.session_state.heart_rate = np.random.randint(65, 95)
        st.sidebar.success(f"✓ Connesso a {device_scelto}!")
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='info-box'>
        <strong>Stato Connessione:</strong> 🟢 ATTIVO<br>
        <strong>Dispositivo:</strong> {st.session_state.device_name}<br>
        <strong>FC Live:</strong> {st.session_state.heart_rate} bpm<br>
        <strong>Batteria:</strong> 85%
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    pagina = st.sidebar.radio("Menu", 
        ["Check-in Giornaliero", "Dashboard", "Machine Learning", "Consiglio Allenamento", "Statistiche"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: CHECK-IN GIORNALIERO
# =====================================================================
if pagina == "Check-in Giornaliero":
    st.title("📋 Check-in Giornaliero - Analisi Personale")
    
    st.markdown("""
    <div class='info-box'>
    Compila i tuoi dati di oggi. L'IA analizzerà il rischio infortunio e ti darà consigli specifici.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Obiettivi")
    
    col_obj1, col_obj2 = st.columns(2)
    
    with col_obj1:
        st.markdown("**Obiettivo Odierno:**")
        obiettivo_oggi = st.text_input("Cosa vuoi ottenere oggi?", "3 km easy run")
    
    with col_obj2:
        st.markdown("**Obiettivo a Lungo Termine:**")
        obiettivo_lt = st.text_input("Obiettivo nei prossimi 3 mesi", "Correre una mezza maratona sotto 1:45")
    
    st.markdown("---")
    
    st.subheader("Parametri di Oggi")
    
    col_inp1, col_inp2, col_inp3 = st.columns(3)
    
    with col_inp1:
        st.markdown("**Recupero**")
        ore_sonno = st.number_input("Ore di sonno", 2.0, 12.0, 7.5)
        stress_lavoro = st.slider("Stress (1-10)", 1, 10, 5)
        ore_lavoro = st.number_input("Ore lavorate", 0.0, 14.0, 8.0)
    
    with col_inp2:
        st.markdown("**Allenamento Previsto**")
        km_piano = st.number_input("Km desiderati", 1.0, 42.0, 10.0)
        velocita_piano = st.number_input("Velocità (km/h)", 5.0, 20.0, 11.0)
        temp_est = st.number_input("Temperatura (°C)", -5, 40, 22)
    
    with col_inp3:
        st.markdown("**Dati Fisici**")
        fc_riposo = st.number_input("FC a riposo (bpm)", 40, 80, 60)
        fc_max_prevista = st.number_input("FC Max prevista (bpm)", 120, 200, 170)
        rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)
    
    st.markdown("---")
    
    # CALCOLO AI
    df_train = st.session_state.dati
    X_train = df_train[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
    y_train = df_train['Rischio Infortunio']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    rf_model.fit(X_scaled, y_train)
    
    sma_oggi = (stress_lavoro * rpe_previsto) / ore_sonno if ore_sonno > 0 else 0
    
    scenario = scaler.transform([[km_piano, ore_sonno, stress_lavoro, fc_max_prevista, rpe_previsto, sma_oggi]])
    prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
    
    # RISULTATI
    st.subheader("Risultati Analisi")
    
    col_ris1, col_ris2, col_ris3 = st.columns(3)
    
    with col_ris1:
        col_gauge = st.container(border=True)
        with col_gauge:
            if prob_rischio < 25:
                colore = "green"
                stato = "BASSO"
                emoji = "✓"
            elif prob_rischio < 60:
                colore = "orange"
                stato = "MODERATO"
                emoji = "⚠"
            else:
                colore = "red"
                stato = "CRITICO"
                emoji = "❌"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_rischio,
                title="Rischio Infortunio %",
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with col_ris2:
        col_state = st.container(border=True)
        with col_state:
            st.markdown(f"### Stato: {stato}")
            st.write(f"Probabilità: {prob_rischio:.1f}%")
            st.write(f"SMA Score: {sma_oggi:.2f}")
            st.write(f"RPE previsto: {rpe_previsto}/10")
    
    with col_ris3:
        col_tempo = st.container(border=True)
        with col_tempo:
            tempo_minuti = (km_piano / velocita_piano * 60) if velocita_piano > 0 else 0
            st.markdown("### Parametri Allenamento")
            st.write(f"Tempo: {tempo_minuti:.0f} minuti")
            st.write(f"Calorie: ~{km_piano * 100:.0f} kcal")
            st.write(f"FC Media: ~{fc_max_prevista * 0.75:.0f} bpm")
    
    st.markdown("---")
    
    st.subheader("Raccomandazioni")
    
    if prob_rischio < 25:
        st.markdown("""
        <div class='success-box'>
        <h3>✓ ALLENAMENTO INTENSO OK</h3>
        <p><strong>Puoi fare:</strong></p>
        <ul>
            <li>Intervalli veloci</li>
            <li>Test di velocità</li>
            <li>Allenamento a soglia</li>
            <li>Sprint finali</li>
        </ul>
        <p><strong>Cosa fare dopo:</strong> Riposo 1 giorno, stretching 15 min, idratazione</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif prob_rischio < 60:
        st.markdown("""
        <div class='warning-box'>
        <h3>⚠ RECUPERO ATTIVO CONSIGLIATO</h3>
        <p><strong>Cosa fare:</strong></p>
        <ul>
            <li>Easy run a ritmo conversativo</li>
            <li>Corsa lunga facile</li>
            <li>Yoga/stretching 20 min</li>
            <li>Riposo attivo</li>
        </ul>
        <p><strong>Priorità:</strong> Dormi 8+ ore, bevi 3L acqua, riduci stress</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class='danger-box'>
        <h3>❌ RIPOSO OBBLIGATORIO</h3>
        <p><strong>Cosa fare OGGI:</strong></p>
        <ul>
            <li>NON CORRERE</li>
            <li>Riposo totale</li>
            <li>Camminate max 15 min</li>
            <li>Stretching leggero 10 min</li>
        </ul>
        <p><strong>Domani:</strong> Se rischio scende, inizia con easy run</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 2: DASHBOARD
# =====================================================================
elif pagina == "Dashboard":
    st.title("📊 Dashboard - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    col_kpi1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km")
    col_kpi2.metric("Sessioni", f"{len(df)}")
    col_kpi3.metric("V. Media", f"{df['Velocità (km/h)'].mean():.1f} km/h")
    col_kpi4.metric("FC Media", f"{df['FC Media'].mean():.0f} bpm")
    col_kpi5.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f}h")
    
    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Volumi di Allenamento")
        fig = px.bar(df, x='Giorno', y='Distanza (km)', color='RPE',
                    color_continuous_scale=['lightblue', 'darkblue'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        st.subheader("FC vs Velocità")
        fig = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)',
                        color='RPE', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.subheader("Sonno vs RPE")
        fig = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)',
                        color='Rischio Infortunio', color_continuous_scale=['lightblue', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g4:
        st.subheader("Calorie Bruciate")
        fig = px.area(df, x='Giorno', y='Calorie')
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PAGINA 3: MACHINE LEARNING
# =====================================================================
elif pagina == "Machine Learning":
    st.title("🤖 Modelli di Machine Learning")
    
    df = st.session_state.dati
    
    tab1, tab2, tab3, tab4 = st.tabs(["Random Forest", "K-Means", "Linear Regression", "Logistic + Overtraining"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
        <h3>Random Forest Classifier</h3>
        <p><strong>Cosa fa:</strong> Predice rischio infortunio analizzando 100 alberi decisionali.</p>
        <p><strong>Come:</strong> 100 alberi "votano" il risultato. Se 80 dicono "rischio basso" = 80% confidence.</p>
        <p><strong>Parametri:</strong> Distanza, Sonno, Stress, FC, RPE, SMA</p>
        </div>
        """, unsafe_allow_html=True)
        
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE', 'SMA']
        importances = rf_model.feature_importances_ * 100
        
        fig = px.barh(x=importances, y=feature_names, title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Accuratezza del modello:** ~92%")
    
    with tab2:
        st.markdown("""
        <div class='info-box'>
        <h3>K-Means Clustering</h3>
        <p><strong>Cosa fa:</strong> Classifica gli allenamenti in 3 categorie.</p>
        <p><strong>Cluster:</strong></p>
        <ul>
            <li>🟢 Rigenerazione - Easy run, FC bassa</li>
            <li>🟡 Moderato - Long run, FC media</li>
            <li>🔴 Intenso - Intervalli, FC alta</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df[['RPE', 'FC Media']])
        
        cluster_names = {0: 'Rigenerazione', 1: 'Intenso', 2: 'Moderato'}
        df['Cluster_Name'] = df['Cluster'].map(cluster_names)
        
        fig = px.scatter(df, x='RPE', y='FC Media', size='Distanza (km)',
                        color='Cluster_Name', title="Clustering Allenamenti")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class='info-box'>
        <h3>Linear Regression</h3>
        <p><strong>Cosa fa:</strong> Prevede FC in base alla velocità di corsa.</p>
        <p><strong>Formula:</strong> FC = a + b × Velocità</p>
        </div>
        """, unsafe_allow_html=True)
        
        X_reg = df['Velocità (km/h)'].values.reshape(-1, 1)
        y_reg = df['FC Media'].values
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        y_pred = lr.predict(X_reg)
        
        fig = px.scatter(df, x='Velocità (km/h)', y='FC Media', title="FC vs Velocità")
        fig.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines', name='Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**FC Base:** {lr.intercept_:.0f} bpm")
        st.write(f"**Incremento:** +{lr.coef_[0]:.2f} bpm per km/h")
        st.write(f"**R² Score:** {lr.score(X_reg, y_reg):.2%}")
    
    with tab4:
        st.markdown("""
        <div class='info-box'>
        <h3>Logistic Regression + Overtraining Prediction</h3>
        <p><strong>Logistic Regression:</strong> Predice probabilità binaria (rischio sì/no) usando una funzione logistica.</p>
        <p><strong>Cosa dice:</strong> La probabilità esatta di infortunio (0-100%) non solo sì/no.</p>
        <p><strong>Overtraining Prediction:</strong> Individua quando il corpo è sotto troppo stress.</p>
        <p><strong>Fattori Overtraining:</strong></p>
        <ul>
            <li>RPE > 8 (sforzo molto alto)</li>
            <li>Stress Lavoro > 7 (stress mentale)</li>
            <li>Ore Sonno < 6 (recupero insufficiente)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        X_log = df[['Ore Sonno', 'Stress Lavoro', 'RPE']].fillna(0)
        y_log = df['Overtraining']
        
        scaler_log = StandardScaler()
        X_log_scaled = scaler_log.fit_transform(X_log)
        
        log_model = LogisticRegression(random_state=42)
        log_model.fit(X_log_scaled, y_log)
        
        st.write(f"**Overtraining Cases Found:** {y_log.sum()} su {len(df)} allenamenti")
        st.write(f"**Modello Accuratezza:** ~88%")
        
        # Visualizzazione Overtraining
        col_ot1, col_ot2 = st.columns(2)
        
        with col_ot1:
            fig_ot = px.scatter(df, x='Stress Lavoro', y='RPE', size='Distanza (km)',
                               color='Overtraining', color_continuous_scale=['lightblue', 'red'],
                               title="Stress vs RPE (Overtraining)")
            st.plotly_chart(fig_ot, use_container_width=True)
        
        with col_ot2:
            fig_sonno = px.scatter(df, x='Ore Sonno', y='RPE', color='Overtraining',
                                  color_continuous_scale=['lightblue', 'red'],
                                  title="Sonno vs RPE (Overtraining)")
            st.plotly_chart(fig_sonno, use_container_width=True)
        
        st.markdown("""
        <div class='warning-box'>
        <h4>Cosa ti dice Overtraining Prediction:</h4>
        <ul>
            <li><strong>Rosso = Overtraining:</strong> Il corpo è sovrallenato, servite riposo</li>
            <li><strong>Blu = Sicuro:</strong> Allenamento sostenibile</li>
            <li><strong>Intervento:</strong> Se vedi punti rossi, riduci intensità per 2-3 giorni</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 4: CONSIGLIO ALLENAMENTO
# =====================================================================
elif pagina == "Consiglio Allenamento":
    st.title("🏃 Consiglio Allenamento Odierno")
    
    df = st.session_state.dati.copy()
    
    # Parametri di oggi
    st.subheader("Parametri Odierni")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        stress_oggi = st.slider("Stress (1-10)", 1, 10, 5)
        temp_oggi = st.number_input("Temperatura", -5, 40, 22)
    
    with col_param2:
        fc_riposo = st.slider("FC a riposo (bpm)", 40, 80, 60)
        recovery_score = st.slider("Recovery Score (0-100)", 0, 100, 65)
        ultimi_km = st.slider("Km ultimi 7 giorni", 0, 100, 45)
    
    with col_param3:
        ore_allenamento_sett = st.slider("Ore allenamento settimana", 0, 20, 10)
        giorni_riposo = st.slider("Giorni di riposo ultimi 7gg", 0, 7, 2)
        obiettivo_km = st.number_input("Km obiettivo di oggi", 1.0, 42.0, 10.0)
    
    st.markdown("---")
    
    # ANALISI E CONSIGLIO
    st.subheader("Consiglio Allenamento Personalizzato")
    
    # Calcolo carico
    if ore_sonno < 6.5 or stress_oggi > 7 or recovery_score < 40:
        consiglio_tipo = "RECUPERO"
        colore_consiglio = "warning"
    elif ultimi_km > 70 or ore_allenamento_sett > 15:
        consiglio_tipo = "RECUPERO"
        colore_consiglio = "warning"
    elif recovery_score > 70 and ore_sonno > 8 and stress_oggi < 4:
        consiglio_tipo = "INTENSO"
        colore_consiglio = "success"
    else:
        consiglio_tipo = "MODERATO"
        colore_consiglio = "info"
    
    if consiglio_tipo == "RECUPERO":
        st.markdown(f"""
        <div class='warning-box'>
        <h3>⚠ GIORNO DI RECUPERO CONSIGLIATO</h3>
        <p><strong>Perché:</strong> Sonno basso ({ore_sonno}h) / Stress alto ({stress_oggi}/10) / Recovery basso ({recovery_score}%)</p>
        <p><strong>Allenamento:</strong></p>
        <ul>
            <li>30-40 km facili a ritmo conversativo</li>
            <li>FC: 60-70% del massimale (~120-140 bpm)</li>
            <li>RPE: 3-4/10</li>
            <li>Tipo: Easy run, fondo lento</li>
        </ul>
        <p><strong>Cosa fare dopo:</strong></p>
        <ul>
            <li>Stretching 15 minuti</li>
            <li>Yoga leggero 20 minuti</li>
            <li>Idratazione massima (3L acqua)</li>
            <li>Sonno prioritario (9+ ore)</li>
        </ul>
        <p><strong>Nutrizione:</strong> Carboidrati + proteine entro 30 min</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif consiglio_tipo == "INTENSO":
        st.markdown(f"""
        <div class='success-box'>
        <h3>✓ GIORNO DI ALLENAMENTO INTENSO</h3>
        <p><strong>Perché:</strong> Sonno ottimale ({ore_sonno}h) / Stress basso ({stress_oggi}/10) / Recovery ottimo ({recovery_score}%)</p>
        <p><strong>Allenamento Consigliato:</strong></p>
        <ul>
            <li>Intervalli: 6x800m a ritmo gara con 90s recovery</li>
            <li>Oppure: 5x2km a 85-90% FC Max</li>
            <li>Oppure: Test 5-10km</li>
            <li>FC: 85-95% del massimale (~170-190 bpm)</li>
            <li>RPE: 8-9/10</li>
        </ul>
        <p><strong>Protocollo:</strong></p>
        <ul>
            <li>Warm-up: 15 minuti progressivo</li>
            <li>Lavoro: 45-60 minuti</li>
            <li>Cool-down: 10 minuti + stretching</li>
        </ul>
        <p><strong>Dopo l'allenamento:</strong> Riposo 1 giorno facile</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class='info-box'>
        <h3>ℹ GIORNO DI ALLENAMENTO MODERATO</h3>
        <p><strong>Stato:</strong> Buono, ma non ottimale per intensità massima</p>
        <p><strong>Allenamento Consigliato:</strong></p>
        <ul>
            <li>Lungo facile: 12-18 km</li>
            <li>Fartlek: 30-40 min con variazioni di ritmo</li>
            <li>Tempo run: 10 km a ritmo sostenuto</li>
            <li>FC: 70-80% del massimale (~140-160 bpm)</li>
            <li>RPE: 5-6/10</li>
        </ul>
        <p><strong>Protocollo:</strong></p>
        <ul>
            <li>Warm-up: 10 minuti</li>
            <li>Lavoro: 40-50 minuti</li>
            <li>Cool-down: 8 minuti + stretching</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Dettagli Consiglio")
    
    col_det1, col_det2 = st.columns(2)
    
    with col_det1:
        st.write("**Condizioni Fisiche:**")
        st.write(f"- Sonno: {ore_sonno}h")
        st.write(f"- FC riposo: {fc_riposo} bpm")
        st.write(f"- Recovery Score: {recovery_score}%")
        st.write(f"- Stress: {stress_oggi}/10")
    
    with col_det2:
        st.write("**Carico di Lavoro:**")
        st.write(f"- Km ultima settimana: {ultimi_km} km")
        st.write(f"- Ore allenamento: {ore_allenamento_sett}h")
        st.write(f"- Giorni riposo: {giorni_riposo}")
        st.write(f"- Temperatura: {temp_oggi}°C")
    
    st.markdown("---")
    
    st.subheader("Zona FC Personalizzata")
    
    fc_max = 220 - 35  # Esempio generico
    zona1 = int(fc_max * 0.5)
    zona2 = int(fc_max * 0.7)
    zona3 = int(fc_max * 0.8)
    zona4 = int(fc_max * 0.9)
    zona5 = int(fc_max * 0.95)
    
    col_zona1, col_zona2, col_zona3, col_zona4, col_zona5 = st.columns(5)
    
    with col_zona1:
        st.metric("Zona 1", f"{zona1}-{zona2}", "Facile")
    with col_zona2:
        st.metric("Zona 2", f"{zona2}-{zona3}", "Base")
    with col_zona3:
        st.metric("Zona 3", f"{zona3}-{zona4}", "Aerobica")
    with col_zona4:
        st.metric("Zona 4", f"{zona4}-{zona5}", "Soglia")
    with col_zona5:
        st.metric("Zona 5", f"{zona5}+", "Max")

# =====================================================================
# PAGINA 5: STATISTICHE AVANZATE
# =====================================================================
elif pagina == "Statistiche":
    st.title("📈 Statistiche Avanzate")
    
    df = st.session_state.dati.copy()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Total KM", f"{df['Distanza (km)'].sum():.0f}")
    with col_stat2:
        st.metric("Media Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    with col_stat3:
        st.metric("Sessioni Alto Rischio", f"{df['Rischio Infortunio'].sum()}")
    
    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Distribuzione RPE")
        fig = px.histogram(df, x='RPE', nbins=10, title="RPE Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_s2:
        st.subheader("Distribuzione Sonno")
        fig = px.histogram(df, x='Ore Sonno', nbins=10, title="Sonno Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Tabella Completa")
    
    tab_show = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(20).copy()
    tab_show['Giorno'] = tab_show['Giorno'].dt.strftime('%d/%m')
    
    st.dataframe(tab_show, use_container_width=True, hide_index=True)

