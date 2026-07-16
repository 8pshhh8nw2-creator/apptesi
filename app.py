import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide", page_icon="🏃")

# CSS BELLISSIMO - Tema Moderno Elegante
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ecf0f1;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    h1 {
        color: #00d4ff;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 1em;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    h2 {
        color: #00bfff;
        font-size: 1.8em;
        font-weight: 700;
        border-bottom: 3px solid #00d4ff;
        padding-bottom: 0.5em;
        margin-top: 1.5em;
    }
    
    h3 {
        color: #00e5ff;
        font-weight: 700;
    }
    
    h4 {
        color: #00d4ff;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #0f5d3a 0%, #1a8f4a 100%);
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        color: #d1fae5;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #6b3a1a 0%, #b85c1a 100%);
        border: 2px solid #f59e0b;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
        color: #fef3c7;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #6b1a1a 0%, #b82a2a 100%);
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        color: #fecaca;
    }
    
    .info-box {
        background: linear-gradient(135deg, #0a3a52 0%, #0f5f7a 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        color: #a5f3fc;
    }
    
    .metric-green {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        border: 2px solid #d1fae5;
    }
    
    .metric-yellow {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
        border: 2px solid #fcd34d;
    }
    
    .metric-red {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        border: 2px solid #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# DATI
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
        'Temp (°C)': np.round(temp, 1),
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1)
    })
    
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.connesso = False

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='margin: 0; font-size: 2.5em;'>🏃</h1>
        <h2 style='margin: 10px 0 0 0; border: none; font-size: 1.5em;'>RunAI Coach</h2>
        <p style='color: #00d4ff; margin-top: 5px;'>Professional Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("Menu", ["Analisi Rischio", "Dashboard", "Statistiche"], label_visibility="collapsed")
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Connetti", use_container_width=True):
            st.session_state.connesso = True
    with col2:
        if st.button("Aggiorna", use_container_width=True):
            st.cache_data.clear()
            st.session_state.dati = genera_dati()

# =====================================================================
# PAGINA 1: ANALISI RISCHIO
# =====================================================================
if pagina == "Analisi Rischio":
    st.title("Analisi Stato di Forma & Rischio Infortunio")
    
    st.markdown("""
    <div class='info-box'>
        <h4>Sistema Intelligente di Previsione</h4>
        <p>Inserisci i tuoi dati e l'IA calcola il % di rischio infortunio con consigli specifici: KM, TEMPO e VELOCITÀ.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("Recupero")
        ore_sonno = st.slider("Ore di Sonno", 2.0, 12.0, 7.5, 0.5)
        ore_lavoro = st.slider("Ore Lavorate", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider("Stress (1-10)", options=list(range(1, 11)), value=5)
    
    with col_input2:
        st.subheader("Allenamento Previsto")
        km_piano = st.number_input("Km desiderati", 1.0, 42.0, 10.0)
        velocita_piano = st.number_input("Velocità (km/h)", 5.0, 20.0, 11.0)
        fc_prevista = st.number_input("FC prevista (bpm)", 100, 200, 150)
    
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
    
    scenario = scaler.transform([[km_piano, ore_sonno, stress_lavoro, fc_prevista, rpe_previsto, sma_oggi]])
    prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
    
    st.markdown("---")
    st.subheader("RISULTATI ANALISI")
    
    col_gauge, col_consigli = st.columns([1.2, 1.8])
    
    with col_gauge:
        colore_gauge = "#10b981" if prob_rischio < 25 else "#f59e0b" if prob_rischio < 60 else "#ef4444"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_rischio,
            title="Rischio Infortunio %",
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colore_gauge},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.3)'},
                    {'range': [25, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                ]
            },
            number={'font': {'color': colore_gauge, 'size': 50}}
        ))
        
        fig_gauge.update_layout(
            height=380,
            paper_bgcolor='rgba(22, 33, 62, 0.9)',
            plot_bgcolor='rgba(22, 33, 62, 0.9)',
            font=dict(color='#ecf0f1')
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_consigli:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        fattori = []
        if ore_sonno < 6.5:
            fattori.append(f"Sonno insufficiente ({ore_sonno}h)")
        if stress_lavoro > 7:
            fattori.append(f"Stress elevato ({stress_lavoro}/10)")
        if rpe_previsto > 7.5:
            fattori.append(f"Sforzo intenso (RPE {rpe_previsto:.1f})")
        if fc_prevista > 160:
            fattori.append(f"FC alta ({fc_prevista} bpm)")
        if sma_oggi > 5:
            fattori.append(f"SMA critica ({sma_oggi:.1f})")
        
        st.markdown(f"**Fattori di Rischio:** {len(fattori)}")
        for f in fattori:
            st.markdown(f"• {f}")
    
    st.markdown("---")
    st.subheader("RACCOMANDAZIONI PERSONALIZZATE")
    
    # VERDE
    if prob_rischio < 25:
        tempo_minuti = (km_piano / velocita_piano * 60)
        km_max = km_piano * 1.15
        fc_max = int(fc_prevista * 1.05)
        
        st.markdown("""
        <div class='success-box'>
            <h3>✓ STATO OTTIMALE - Rischio Basso</h3>
            <p>Tutti gli indicatori sono verdi. Puoi allenarti normalmente.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown(f"""
            <div class='metric-green'>
                <div style='font-size: 0.9em; opacity: 0.9;'>KM MASSIMI</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{km_max:.1f}</div>
                <div style='font-size: 0.85em;'>km sicuri</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown(f"""
            <div class='metric-green'>
                <div style='font-size: 0.9em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{tempo_minuti:.0f}</div>
                <div style='font-size: 0.85em;'>minuti</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown(f"""
            <div class='metric-green'>
                <div style='font-size: 0.9em; opacity: 0.9;'>FC MASSIMA</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{fc_max}</div>
                <div style='font-size: 0.85em;'>bpm</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Piano Allenamento:**
        - {km_piano:.1f} km a {velocita_piano:.1f} km/h
        - Tempo: {tempo_minuti:.0f} minuti
        - RPE: {rpe_previsto:.1f}/10
        - Tipo: INTENSO - Intervalli, ripetute
        - Recovery: 1 giorno easy dopo
        """)
    
    # GIALLO
    elif prob_rischio < 60:
        tempo_minuti = (km_piano / velocita_piano * 60) * 0.8
        km_max = km_piano * 0.9
        velocita_ridotta = velocita_piano * 0.85
        
        st.markdown("""
        <div class='warning-box'>
            <h3>⚠ ATTENZIONE - Rischio Moderato</h3>
            <p>Puoi correre ma con restrizioni importanti.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown(f"""
            <div class='metric-yellow'>
                <div style='font-size: 0.9em; opacity: 0.9;'>KM MASSIMI</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{km_max:.1f}</div>
                <div style='font-size: 0.85em;'>km (-10%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown(f"""
            <div class='metric-yellow'>
                <div style='font-size: 0.9em; opacity: 0.9;'>TEMPO MASSIMO</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{tempo_minuti:.0f}</div>
                <div style='font-size: 0.85em;'>minuti (-20%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown(f"""
            <div class='metric-yellow'>
                <div style='font-size: 0.9em; opacity: 0.9;'>VELOCITÀ</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>{velocita_ridotta:.1f}</div>
                <div style='font-size: 0.85em;'>km/h (easy)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Piano Recupero Attivo:**
        - {km_max:.1f} km a {velocita_ridotta:.1f} km/h (ritmo conversativo)
        - Tempo: {tempo_minuti:.0f} minuti
        - RPE: 4-5/10 (BASSO)
        - Tipo: RECUPERO - Solo easy run
        - ❌ EVITA: Intervalli, sprint, salite
        - Recovery: Riposo totale il giorno dopo
        
        **Priorità 24-48h:**
        - Dormi 8+ ore stasera
        - Mantieni stress basso
        - Bevi 3+ litri acqua
        """)
    
    # ROSSO
    else:
        st.markdown("""
        <div class='danger-box'>
            <h3>🛑 RISCHIO CRITICO - RIPOSO OBBLIGATORIO</h3>
            <p>Il tuo corpo è in serio pericolo. Non correre oggi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown("""
            <div class='metric-red'>
                <div style='font-size: 0.9em; opacity: 0.9;'>KM CONSIGLIATI</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>0</div>
                <div style='font-size: 0.85em;'>Riposo totale</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
            <div class='metric-red'>
                <div style='font-size: 0.9em; opacity: 0.9;'>ALLENAMENTO</div>
                <div style='font-size: 2.5em; margin: 10px 0;'>-</div>
                <div style='font-size: 0.85em;'>Niente sport</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec3:
            st.markdown("""
            <div class='metric-red'>
                <div style='font-size: 0.9em; opacity: 0.9;'>PRIORITÀ</div>
                <div style='font-size: 2em; margin: 10px 0;'>RIPOSO</div>
                <div style='font-size: 0.85em;'>24-48 ore</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **AZIONI IMMEDIATE:**
        - ❌ NON CORRERE
        - ✓ Stai a casa
        - ✓ Dormi 9+ ore stasera
        - ✓ Bevi 3+ litri acqua
        - ✓ Solo stretching leggero
        
        **SEGNALI ALLARME - Consulta medico:**
        - Dolore persistente
        - Gonfiore/rigidità
        - Febbre > 37.5°C
        - Stanchezza estrema
        """)

elif pagina == "Dashboard":
    st.title("Dashboard - Ultimi 90 Giorni")
    
    if not st.session_state.connesso:
        st.warning("Connetti dispositivo per dati storici")
    else:
        df = st.session_state.dati.copy()
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        col_stat1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f}")
        col_stat2.metric("V. Media", f"{df['Velocità (km/h)'].mean():.1f} km/h")
        col_stat3.metric("FC Media", f"{df['FC Media'].mean():.0f} bpm")
        col_stat4.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f}h")
        col_stat5.metric("Sessioni", f"{len(df)}")
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_vol = px.bar(df, x='Giorno', y='Distanza (km)', color='RPE', 
                            color_continuous_scale=['#16213e', '#00d4ff', '#00e5ff'],
                            title="Timeline Distanza", height=350)
            fig_vol.update_layout(plot_bgcolor='rgba(22,33,62,0.5)', paper_bgcolor='rgba(26,26,46,0.8)', 
                                 font=dict(color='#ecf0f1'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col_g2:
            fig_scat = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', 
                                 color='Rischio Infortunio', color_continuous_scale=['#00d4ff', '#ef4444'],
                                 title="Sonno vs RPE", height=350, opacity=0.7)
            fig_scat.update_layout(plot_bgcolor='rgba(22,33,62,0.5)', paper_bgcolor='rgba(26,26,46,0.8)', 
                                  font=dict(color='#ecf0f1'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_scat, use_container_width=True)

elif pagina == "Statistiche":
    st.title("Analisi Dettagliata")
    
    if not st.session_state.connesso:
        st.warning("Connetti dispositivo per dati storici")
    else:
        df = st.session_state.dati.copy()
        
        tab1, tab2, tab3 = st.tabs(["Velocità vs FC", "Temperatura", "Tabella"])
        
        with tab1:
            X_plot = df['Velocità (km/h)'].values.reshape(-1, 1)
            y_plot = df['FC Media'].values
            lr = LinearRegression()
            lr.fit(X_plot, y_plot)
            y_pred = lr.predict(X_plot)
            
            fig_reg = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='RPE', 
                                color='Ore Sonno', color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                                title="Velocità vs FC", height=400, opacity=0.6)
            fig_reg.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines', 
                               name='Trend', line=dict(color='#ef4444', width=3))
            fig_reg.update_layout(plot_bgcolor='rgba(22,33,62,0.5)', paper_bgcolor='rgba(26,26,46,0.8)', 
                                 font=dict(color='#ecf0f1'))
            st.plotly_chart(fig_reg, use_container_width=True)
            
            st.markdown(f"**Coefficiente:** +{lr.coef_[0]:.2f} bpm per km/h | **R²:** {lr.score(X_plot, y_plot):.2%}")
        
        with tab2:
            fig_temp = px.scatter(df, x='Temp (°C)', y='FC Media', size='Distanza (km)', 
                                 color='Velocità (km/h)', color_continuous_scale='Viridis',
                                 title="Temperatura vs FC", height=400, opacity=0.7)
            fig_temp.update_layout(plot_bgcolor='rgba(22,33,62,0.5)', paper_bgcolor='rgba(26,26,46,0.8)', 
                                  font=dict(color='#ecf0f1'))
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab3:
            tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno']].tail(15).copy()
            tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m')
            st.dataframe(tab_data, use_container_width=True, hide_index=True)
