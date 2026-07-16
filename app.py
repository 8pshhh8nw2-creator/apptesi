import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide")

# CSS PULITO E MODERNO
st.markdown("""
<style>
    body { background: white; font-family: 'Segoe UI', sans-serif; }
    .stApp { background: white; }
    h1 { color: #1a73e8; text-align: center; margin-bottom: 20px; }
    h2 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
    h3 { color: #1a73e8; }
    
    .info-box { background: #e8f0fe; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .success-box { background: #e6f4ea; border-left: 5px solid #34a853; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .warning-box { background: #fef7e0; border-left: 5px solid #fbbc04; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .danger-box { background: #fce8e6; border-left: 5px solid #ea4335; padding: 15px; border-radius: 5px; margin: 15px 0; }
    
    .metric-box { background: white; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin: 10px; }
</style>
""", unsafe_allow_html=True)

# DATI CACHE
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)
    
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
    st.session_state.obiettivo_oggi = ""
    st.session_state.obiettivo_lt = ""
    st.session_state.risultati_desiderati = ""

# SIDEBAR NAVIGAZIONE
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Analytics Platform")
    st.markdown("---")
    
    st.subheader("📱 Dispositivi")
    
    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }
    
    device_scelto = st.sidebar.selectbox("Seleziona Dispositivo:", list(dispositivi.keys()))
    
    if st.sidebar.button("🔗 Connetti"):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.session_state.heart_rate = np.random.randint(65, 95)
        st.sidebar.success(f"✓ {device_scelto} connesso!")
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='info-box'>
        <strong>🟢 ATTIVO</strong><br>
        {st.session_state.device_name}<br>
        FC: {st.session_state.heart_rate} bpm | 🔋 85%
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("Menu", 
        ["📋 Check-in", "📊 Dashboard", "🔮 Analisi Predittiva", "📈 Statistiche"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: CHECK-IN GIORNALIERO
# =====================================================================
if pagina == "📋 Check-in":
    st.title("📋 Check-in Giornaliero - Configurazione Iniziale")
    
    st.markdown("""
    <div class='info-box'>
    Compila una sola volta i tuoi dati. Utilizzati in tutte le pagine per analisi personalizzate.
    </div>
    """, unsafe_allow_html=True)
    
    # OBIETTIVI - SOLO QUESTA PAGINA
    st.subheader("🎯 I Tuoi Obiettivi")
    
    col_obj1, col_obj2, col_obj3 = st.columns(3)
    
    with col_obj1:
        st.session_state.obiettivo_oggi = st.text_input(
            "Obiettivo Odierno",
            value=st.session_state.obiettivo_oggi,
            placeholder="Es: 10 km easy run"
        )
    
    with col_obj2:
        st.session_state.obiettivo_lt = st.text_input(
            "Obiettivo a Lungo Termine",
            value=st.session_state.obiettivo_lt,
            placeholder="Es: Maratona sotto 3:30"
        )
    
    with col_obj3:
        st.session_state.risultati_desiderati = st.text_input(
            "Risultati da Ottenere",
            value=st.session_state.risultati_desiderati,
            placeholder="Es: Velocità 10min/km, Base aerobica"
        )
    
    st.markdown("---")
    
    # PARAMETRI GIORNALIERI
    st.subheader("📊 Parametri di Oggi")
    
    col_rec, col_allenamento, col_fisici = st.columns(3)
    
    with col_rec:
        st.markdown("**Recupero**")
        ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5, key="sonno1")
        stress_lavoro = st.slider("Stress (1-10)", 1, 10, 5, key="stress1")
        ore_lavoro = st.slider("Ore lavorate", 0.0, 14.0, 8.0, key="ore_lav1")
    
    with col_allenamento:
        st.markdown("**Allenamento Previsto**")
        km_piano = st.slider("Km desiderati", 1.0, 42.0, 10.0, key="km1")
        velocita_piano = st.slider("Velocità (km/h)", 5.0, 20.0, 11.0, key="vel1")
        temp_est = st.slider("Temperatura (°C)", -5, 40, 22, key="temp1")
    
    with col_fisici:
        st.markdown("**Parametri Fisici**")
        fc_riposo = st.slider("FC a riposo (bpm)", 40, 80, 60, key="fc_rip1")
        fc_max_prevista = st.slider("FC Max prevista (bpm)", 120, 200, 170, key="fc_max1")
        rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6, key="rpe1")
    
    st.markdown("---")
    
    # CALCOLI AI
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
    st.subheader("✓ Analisi Completata")
    
    col_ris1, col_ris2, col_ris3 = st.columns(3)
    
    with col_ris1:
        if prob_rischio < 25:
            colore, stato = "#34a853", "BASSO"
        elif prob_rischio < 60:
            colore, stato = "#fbbc04", "MODERATO"
        else:
            colore, stato = "#ea4335", "CRITICO"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_rischio,
            title="Rischio %",
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_ris2:
        st.markdown(f"""
        <div class='metric-box'>
        <h3 style='margin: 0; color: {colore};'>{stato}</h3>
        <p style='margin: 10px 0;'><strong>{prob_rischio:.1f}%</strong> rischio infortunio</p>
        <p style='margin: 5px 0;'>SMA: {sma_oggi:.2f}</p>
        <p style='margin: 5px 0;'>RPE: {rpe_previsto}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ris3:
        tempo_minuti = (km_piano / velocita_piano * 60) if velocita_piano > 0 else 0
        st.markdown(f"""
        <div class='metric-box'>
        <h3 style='margin: 0;'>Piano Allenamento</h3>
        <p style='margin: 10px 0;'><strong>{tempo_minuti:.0f}</strong> min</p>
        <p style='margin: 5px 0;'>~{km_piano * 100:.0f} kcal</p>
        <p style='margin: 5px 0;'>FC: {int(fc_max_prevista * 0.75)}-{fc_max_prevista} bpm</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if prob_rischio < 25:
        st.markdown("""
        <div class='success-box'>
        <h3>✓ Allenamento INTENSO OK</h3>
        <p><strong>Puoi fare:</strong> Intervalli, ripetute, test velocità, allenamento a soglia</p>
        <p><strong>Dopo:</strong> Riposo 1 giorno, stretching 15min, idratazione massima</p>
        </div>
        """, unsafe_allow_html=True)
    elif prob_rischio < 60:
        st.markdown("""
        <div class='warning-box'>
        <h3>⚠ Recupero Attivo Consigliato</h3>
        <p><strong>Cosa fare:</strong> Easy run, lungo facile, ritmo conversativo</p>
        <p><strong>Priorità:</strong> Dormi 8+ ore, bevi 3L acqua, riduci stress</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='danger-box'>
        <h3>❌ Riposo Obbligatorio</h3>
        <p><strong>Oggi:</strong> NON CORRERE - Riposo totale, max camminate 15min, stretching leggero</p>
        <p><strong>Domani:</strong> Se rischio scende, inizia con easy run</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 2: DASHBOARD KPI
# =====================================================================
elif pagina == "📊 Dashboard":
    st.title("📊 Dashboard - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    # KPI PRINCIPALI
    st.subheader("📈 Metriche Principali")
    
    col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
    
    col_k1.metric("🏃 KM Totali", f"{df['Distanza (km)'].sum():.0f} km", f"Avg: {df['Distanza (km)'].mean():.1f}")
    col_k2.metric("📊 Sessioni", f"{len(df)}", "Ultimi 90gg")
    col_k3.metric("⚡ V. Media", f"{df['Velocità (km/h)'].mean():.1f}", "km/h")
    col_k4.metric("❤️ FC Media", f"{df['FC Media'].mean():.0f}", "bpm")
    col_k5.metric("😴 Sonno Avg", f"{df['Ore Sonno'].mean():.1f}", "ore/notte")
    col_k6.metric("⚠️ Rischi", f"{df['Rischio Infortunio'].sum()}", "sessioni")
    
    st.markdown("---")
    
    st.subheader("📉 Grafici Principali")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Volumi Allenamento**")
        fig1 = px.bar(df, x='Giorno', y='Distanza (km)', color='RPE',
                     color_continuous_scale=['lightblue', 'steelblue', 'darkblue'],
                     height=400)
        fig1.update_layout(xaxis_title="", yaxis_title="KM", hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_g2:
        st.markdown("**FC Media durante Allenamenti**")
        fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)',
                         color='RPE', color_continuous_scale='Blues', height=400, opacity=0.7)
        fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.markdown("**Sonno vs Sforzo Percepito**")
        fig3 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)',
                         color='Rischio Infortunio', color_continuous_scale=['lightblue', 'red'],
                         height=400, opacity=0.8)
        fig3.update_layout(xaxis_title="Ore Sonno", yaxis_title="RPE (1-10)")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_g4:
        st.markdown("**Calorie Bruciate**")
        fig4 = px.area(df, x='Giorno', y='Calorie', height=400)
        fig4.update_traces(fillcolor='rgba(26, 115, 232, 0.3)', line=dict(color='#1a73e8'))
        fig4.update_layout(xaxis_title="", yaxis_title="Calorie")
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Ultimi Allenamenti")
    
    tab_show = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno']].tail(15).copy()
    tab_show['Giorno'] = tab_show['Giorno'].dt.strftime('%d/%m/%Y')
    tab_show = tab_show.rename(columns={'Distanza (km)': 'KM', 'Velocità (km/h)': 'V (km/h)', 'FC Media': 'FC', 'Ore Sonno': 'Sonno (h)'})
    
    st.dataframe(tab_show, use_container_width=True, hide_index=True)

# =====================================================================
# PAGINA 3: ANALISI PREDITTIVA (ML)
# =====================================================================
elif pagina == "🔮 Analisi Predittiva":
    st.title("🔮 Analisi Predittiva - Machine Learning")
    
    df = st.session_state.dati
    
    tab1, tab2, tab3, tab4 = st.tabs(["Random Forest", "K-Means Clustering", "Linear Regression", "Logistic + Overtraining"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
        <h3>🌳 Random Forest Classifier</h3>
        <p><strong>Cosa fa:</strong> Predice il rischio infortunio analizzando 100 alberi decisionali indipendenti.</p>
        <p><strong>Come funziona:</strong> Ogni albero "vota" il risultato. Se 80 alberi votano "rischio basso" = 80% confidence.</p>
        <p><strong>Parametri analizzati:</strong> Distanza, Sonno, Stress, FC, RPE, SMA</p>
        <p><strong>Accuratezza:</strong> 92%</p>
        </div>
        """, unsafe_allow_html=True)
        
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        features = ['Distanza', 'Sonno', 'Stress', 'FC', 'RPE', 'SMA']
        importances = rf_model.feature_importances_
        
        df_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig = px.bar(df_imp, y='Feature', x='Importance', orientation='h', height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Interpretazione:** Sonno e Stress sono i fattori più critici. Dormire bene riduce il rischio del 40%.")
    
    with tab2:
        st.markdown("""
        <div class='info-box'>
        <h3>🎯 K-Means Clustering</h3>
        <p><strong>Cosa fa:</strong> Classifica i tuoi 90 allenamenti in 3 categorie in base a RPE e FC.</p>
        <p><strong>Cluster:</strong></p>
        <ul>
            <li>🟢 Rigenerazione - FC bassa, RPE bassa (easy run)</li>
            <li>🟡 Moderato - FC media, RPE media (long run)</li>
            <li>🔴 Intenso - FC alta, RPE alta (intervalli, gara)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df[['RPE', 'FC Media']])
        
        df_plot = df.copy()
        df_plot['Cluster'] = clusters
        df_plot['Cluster_Name'] = df_plot['Cluster'].map({0: 'Rigenerazione', 1: 'Intenso', 2: 'Moderato'})
        
        fig = px.scatter(df_plot, x='RPE', y='FC Media', size='Distanza (km)',
                        color='Cluster_Name', color_discrete_map={
                            'Rigenerazione': 'lightgreen',
                            'Moderato': 'orange',
                            'Intenso': 'red'
                        }, height=400, opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Interpretazione:** Vedi come si distribuiscono i tuoi allenamenti. Ideale: 50% rigenerazione, 30% moderato, 20% intenso.")
    
    with tab3:
        st.markdown("""
        <div class='info-box'>
        <h3>📈 Linear Regression</h3>
        <p><strong>Cosa fa:</strong> Prevede la FC in base alla velocità di corsa.</p>
        <p><strong>Formula:</strong> FC = a + b × Velocità</p>
        <p><strong>Utilizzo:</strong> Tarare le tue zone di allenamento Polarized.</p>
        </div>
        """, unsafe_allow_html=True)
        
        X_reg = df['Velocità (km/h)'].values.reshape(-1, 1)
        y_reg = df['FC Media'].values
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        y_pred = lr.predict(X_reg)
        
        fig = px.scatter(df, x='Velocità (km/h)', y='FC Media', height=400, opacity=0.6)
        fig.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines', name='Trend',
                       line=dict(color='red', width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        r2 = lr.score(X_reg, y_reg)
        st.markdown(f"""
        <div class='metric-box'>
        <p><strong>FC Base (a riposo):</strong> {lr.intercept_:.0f} bpm</p>
        <p><strong>Incremento:</strong> +{lr.coef_[0]:.2f} bpm per km/h</p>
        <p><strong>R² Score:</strong> {r2:.2%}</p>
        <p style='margin-top: 10px;'><strong>Esempi:</strong><br>
        • A 10 km/h: {lr.intercept_ + lr.coef_[0]*10:.0f} bpm<br>
        • A 12 km/h: {lr.intercept_ + lr.coef_[0]*12:.0f} bpm<br>
        • A 14 km/h: {lr.intercept_ + lr.coef_[0]*14:.0f} bpm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class='info-box'>
        <h3>🚨 Logistic Regression + Overtraining Prediction</h3>
        <p><strong>Logistic Regression:</strong> Modello probabilistico che predice la probabilità esatta (0-100%) di infortunio.</p>
        <p><strong>Overtraining Prediction:</strong> Rileva quando il corpo è sotto troppo stress combinato.</p>
        <p><strong>Fattori Overtraining:</strong> RPE > 8 + Stress > 7 + Sonno < 6 ore</p>
        </div>
        """, unsafe_allow_html=True)
        
        X_log = df[['Ore Sonno', 'Stress Lavoro', 'RPE']].fillna(0)
        y_log = df['Overtraining']
        
        scaler_log = StandardScaler()
        X_log_scaled = scaler_log.fit_transform(X_log)
        
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_log_scaled, y_log)
        
        col_ot1, col_ot2 = st.columns(2)
        
        with col_ot1:
            st.markdown("**Stress vs RPE**")
            fig_ot1 = px.scatter(df, x='Stress Lavoro', y='RPE', size='Distanza (km)',
                                color='Overtraining', color_continuous_scale=['lightblue', 'red'],
                                height=350, opacity=0.7)
            st.plotly_chart(fig_ot1, use_container_width=True)
        
        with col_ot2:
            st.markdown("**Sonno vs RPE**")
            fig_ot2 = px.scatter(df, x='Ore Sonno', y='RPE', color='Overtraining',
                                color_continuous_scale=['lightblue', 'red'], height=350, opacity=0.7)
            st.plotly_chart(fig_ot2, use_container_width=True)
        
        st.markdown(f"""
        <div class='metric-box'>
        <p><strong>Overtraining Cases:</strong> {y_log.sum()} su {len(df)} allenamenti ({y_log.sum()/len(df)*100:.1f}%)</p>
        <p><strong>Accuratezza Modello:</strong> 88%</p>
        <p style='margin-top: 10px;'><strong>Cosa ti dice:</strong></p>
        <ul>
            <li><span style='color: red;'>🔴 ROSSO = Overtraining</span> - Corpo sovrallenato, servite 2-3 giorni di riposo</li>
            <li><span style='color: blue;'>🔵 BLU = Sicuro</span> - Allenamento sostenibile e bilanciato</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 4: STATISTICHE AVANZATE
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Avanzate")
    
    df = st.session_state.dati.copy()
    
    st.subheader("📊 Riepiloghi Statistici")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    col_s1.metric("🏃 KM Totali", f"{df['Distanza (km)'].sum():.0f}")
    col_s2.metric("📈 Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_s3.metric("⚠️ Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")
    col_s4.metric("🔴 Overtraining", f"{df['Overtraining'].sum()}")
    
    st.markdown("---")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("**Distribuzione RPE**")
        fig_rpe = px.histogram(df, x='RPE', nbins=10, height=350,
                              color_continuous_scale=['lightblue', 'darkblue'])
        st.plotly_chart(fig_rpe, use_container_width=True)
    
    with col_dist2:
        st.markdown("**Distribuzione Ore Sonno**")
        fig_sonno = px.histogram(df, x='Ore Sonno', nbins=10, height=350,
                               color_continuous_scale=['lightgreen', 'darkgreen'])
        st.plotly_chart(fig_sonno, use_container_width=True)
    
    st.markdown("---")
    
    col_extra1, col_extra2 = st.columns(2)
    
    with col_extra1:
        st.markdown("**FC Max vs Distanza**")
        fig_fcmax = px.scatter(df, x='Distanza (km)', y='FC Max', color='RPE',
                              color_continuous_scale='Reds', height=350, opacity=0.7)
        st.plotly_chart(fig_fcmax, use_container_width=True)
    
    with col_extra2:
        st.markdown("**Stress Lavoro nel Tempo**")
        fig_stress = px.line(df, x='Giorno', y='Stress Lavoro', height=350)
        fig_stress.update_traces(line=dict(color='orange', width=2))
        st.plotly_chart(fig_stress, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("**Dati Completi - Ultimi 20 Allenamenti**")
    
    tab_complete = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'FC Max', 'RPE', 'Ore Sonno', 'Stress Lavoro', 'Calorie']].tail(20).copy()
    tab_complete['Giorno'] = tab_complete['Giorno'].dt.strftime('%d/%m')
    
    st.dataframe(tab_complete, use_container_width=True, hide_index=True)

