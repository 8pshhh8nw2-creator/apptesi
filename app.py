import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURAZIONE APP ---
st.set_page_config(page_title="RunAI Coach", layout="wide", page_icon="🏃")

# CSS Professionale
st.markdown("""
<style>
    /* Tema moderno e pulito */
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #3b82f6;
    }
    
    h1 { 
        color: #1e293b;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    
    h2 {
        color: #334155;
        font-size: 1.8em;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5em;
    }
    
    h3 {
        color: #475569;
        font-size: 1.3em;
        font-weight: 500;
    }
    
    .stAlert {
        border-radius: 12px;
        padding: 15px;
        font-size: 1.05em;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. GENERATORE DATI INTELLIGENTE ---
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
    
    df['SMA'] = (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno']
    df['ISLR'] = (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)']
    df['IITR'] = (df['Temp (°C)'] * df['Vento (km/h)']) / df['Distanza (km)']
    df['IDET'] = (df['FC Media'] * df['Temp (°C)']) / df['Velocità (km/h)']
    
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati_intelligenti()
    st.session_state.connesso = False

# --- 3. MENU NAVIGAZIONE ---
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: white; font-size: 2em; margin: 0;'>🏃</h1>
        <h2 style='color: white; font-size: 1.5em; margin: 10px 0;'>RunAI Coach</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9em;'>Il tuo allenatore virtuale intelligente</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
pagina = st.sidebar.radio("Seleziona Sezione", 
    ["Home & Check-in", "Le mie Statistiche", "Analisi AI & Previsioni"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔗 Connetti Dispositivo", use_container_width=True):
        st.session_state.connesso = True
        st.sidebar.success("✓ Dispositivo connesso!")

with col_btn2:
    if st.button("🔄 Aggiorna Dati", use_container_width=True):
        st.session_state.dati = genera_dati_intelligenti()
        st.sidebar.info("✓ Dati aggiornati!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='insight-box'>
    <strong>💡 Suggerimento:</strong><br>
    Connetti il tuo smartwatch Garmin o Strava per dati automatici e analisi più precise!
</div>
""", unsafe_allow_html=True)

# =====================================================================
# PAGINA 1: HOME
# =====================================================================
if pagina == "Home & Check-in":
    st.title("Benvenuto! Compila il tuo Check-in Giornaliero")
    
    st.markdown("""
    <div class='insight-box'>
        <strong>Come funziona?</strong> Inserisci i dati di oggi e la nostra intelligenza artificiale 
        analizzerà il tuo stato di forma, calcolerà il rischio infortunio e ti darà consigli personalizzati.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌙 Recupero e Lavoro")
        ore_sonno = st.slider("Ore di sonno la notte scorsa", 2.0, 12.0, 7.0, 0.5)
        ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0, 0.5)
        stress_lavoro = st.select_slider(
            "Stress mentale/lavorativo", 
            options=[1,2,3,4,5,6,7,8,9,10], 
            value=5
        )
    
    with col2:
        st.subheader("🏃 Dati dell'Allenamento")
        km_fatti = st.number_input("Chilometri percorsi", 1.0, 42.0, 10.0)
        velocita = st.number_input("Velocità media (km/h)", 5.0, 20.0, 11.0)
        fc_media = st.number_input("Battito Cardiaco Medio (bpm)", 100, 200, 145)
        rpe = st.slider("RPE - Sforzo Percepito (Borg Scale)", 1, 10, 6)
        temp = st.number_input("Temperatura esterna (°C)", -5, 40, 22)

    st.markdown("---")
    st.subheader("📊 I Tuoi KPI Odierni")
    
    # Calcoli
    sma = (stress_lavoro * rpe) / ore_sonno if ore_sonno > 0 else 0
    islr = (ore_lavoro * stress_lavoro) / km_fatti if km_fatti > 0 else 0
    idet = (fc_media * temp) / velocita if velocita > 0 else 0
    iitr = (temp * 5) / km_fatti if km_fatti > 0 else 0  # vento generico
    
    met1, met2, met3, met4 = st.columns(4)
    
    with met1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0; color: white;'>Equilibrio Mente</h3>
            <p style='font-size: 2.5em; margin: 10px 0; font-weight: bold;'>{sma:.1f}</p>
            <p style='font-size: 0.85em; margin: 0; opacity: 0.9;'>SMA Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0; color: white;'>Impatto Lavoro</h3>
            <p style='font-size: 2.5em; margin: 10px 0; font-weight: bold;'>{islr:.1f}</p>
            <p style='font-size: 0.85em; margin: 0; opacity: 0.9;'>ISLR Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met3:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0; color: white;'>Affaticamento Cuore</h3>
            <p style='font-size: 2.5em; margin: 10px 0; font-weight: bold;'>{idet:.0f}</p>
            <p style='font-size: 0.85em; margin: 0; opacity: 0.9;'>IDET Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met4:
        st.markdown(f"""
        <div class='metric-box'>
            <h3 style='margin: 0; color: white;'>Impatto Clima</h3>
            <p style='font-size: 2.5em; margin: 10px 0; font-weight: bold;'>{iitr:.1f}</p>
            <p style='font-size: 0.85em; margin: 0; opacity: 0.9;'>IITR Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📈 Analisi Dettagliata")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("""
        **SMA (Stress Mentale Aggregato)**
        - Misura l'equilibrio tra stress, sforzo e recupero
        - Valori alti = corpo stressato dal lavoro
        - Ideale: < 3.5
        """)
    
    with col_exp2:
        st.markdown("""
        **ISLR (Impatto Stress Lavoro su Risultati)**
        - Indica quanto il lavoro influisce sulla tua corsa
        - Valori alti = fatica accumulata
        - Ideale: < 2.0
        """)

# =====================================================================
# PAGINA 2: STATISTICHE
# =====================================================================
elif pagina == "Le mie Statistiche":
    st.title("Analisi Dettagliata dei tuoi Ultimi 90 Giorni")
    
    if not st.session_state.connesso:
        st.warning("⚠️ Connetti un dispositivo per visualizzare i tuoi dati storici.")
    else:
        df = st.session_state.dati
        
        # Statistiche riassuntive
        st.subheader("📊 Riassunto Generale")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total KM", f"{df['Distanza (km)'].sum():.0f}", "ultimi 90 giorni")
        col_stat2.metric("Velocità Media", f"{df['Velocità (km/h)'].mean():.1f}", "km/h")
        col_stat3.metric("FC Media", f"{df['FC Media'].mean():.0f}", "bpm")
        col_stat4.metric("Sonno Medio", f"{df['Ore Sonno'].mean():.1f}", "ore/notte")
        
        st.markdown("---")
        
        # GRAFICO 1: Timeline dei volumi
        st.subheader("📈 Volume di Allenamento nel Tempo")
        st.markdown("""
        Questo grafico mostra i chilometri percorsi ogni giorno. I colori rappresentano lo sforzo percepito (RPE):
        - **Blu chiaro** = Sforzo basso (recupero)
        - **Blu scuro** = Sforzo massimo (intenso)
        """)
        
        fig_volume = px.bar(
            df, 
            x='Giorno', 
            y='Distanza (km)',
            color='RPE',
            color_continuous_scale=['#e0f2fe', '#0284c7', '#0c4a6e'],
            title="Chilometri Percorsi - Ultimi 90 Giorni",
            labels={'Distanza (km)': 'Km', 'Giorno': 'Data'},
            height=450
        )
        fig_volume.update_layout(
            hovermode='x unified',
            plot_bgcolor='rgba(240, 248, 255, 0.5)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            font=dict(size=11)
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        st.markdown("---")
        
        # GRAFICO 2: Sonno vs Fatica
        col_g2_1, col_g2_2 = st.columns(2)
        
        with col_g2_1:
            st.subheader("😴 Correlazione: Sonno vs Sforzo Percepito")
            st.markdown("""
            Ogni punto rappresenta una corsa. Osserva la tendenza:
            - **Punti in basso a sinistra**: Poco sonno, sforzo basso (RISCHIO!)
            - **Punti in alto a destra**: Molto sonno, sforzo controllato (IDEALE)
            - **Colore rosso**: Giornate ad alto rischio infortunio
            """)
            
            fig_scatter = px.scatter(
                df,
                x="Ore Sonno",
                y="RPE",
                size="Distanza (km)",
                color="Rischio Infortunio",
                color_continuous_scale=['#06b6d4', '#f43f5e'],
                opacity=0.7,
                title="Sonno vs Sforzo Percepito",
                labels={'Ore Sonno': 'Ore di Sonno', 'RPE': 'Sforzo Percepito (RPE)'},
                height=400
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(240, 248, 255, 0.5)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
                font=dict(size=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_g2_2:
            st.subheader("❤️ Frequenza Cardiaca vs Temperatura")
            st.markdown("""
            Mostra come la temperatura influisce sul battito cardiaco:
            - A temperature più alte, il cuore batte più velocemente
            - Questo è normale ma indica sforzo maggiore
            - Importante rimanere idratati quando fa caldo
            """)
            
            fig_heat = px.scatter(
                df,
                x="Temp (°C)",
                y="FC Media",
                size="Distanza (km)",
                color="Velocità (km/h)",
                color_continuous_scale='Viridis',
                opacity=0.7,
                title="Temperatura vs Frequenza Cardiaca",
                labels={'Temp (°C)': 'Temperatura (°C)', 'FC Media': 'FC Media (bpm)'},
                height=400
            )
            fig_heat.update_layout(
                plot_bgcolor='rgba(240, 248, 255, 0.5)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
                font=dict(size=10)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("---")
        
        # GRAFICO 3: Radar Chart
        st.subheader("🕸️ Il Tuo Profilo di Allenamento")
        st.markdown("""
        Questo grafico a ragnatela mostra come i tuoi KPI si bilanciano:
        - Una ragnatela **regolare** = allenamento equilibrato
        - Una ragnatela **irregolare** = qualche area su cui lavorare
        """)
        
        categorie = ['Stress Mentale', 'Impatto Lavoro', 'Impatto Clima', 'Affaticamento Cuore']
        valori = [
            df['SMA'].mean(),
            df['ISLR'].mean(),
            df['IITR'].mean(),
            df['IDET'].mean()/30  # Normalizzazione
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=valori,
            theta=categorie,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            marker=dict(size=10, color='#667eea')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(valori)*1.2],
                    tickfont=dict(size=9)
                ),
                angularaxis=dict(tickfont=dict(size=11))
            ),
            height=450,
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            plot_bgcolor='rgba(240, 248, 255, 0.3)'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        # Tabella dati
        st.subheader("📋 Ultimi 10 Allenamenti")
        tabella_mostra = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(10).copy()
        tabella_mostra['Giorno'] = tabella_mostra['Giorno'].dt.strftime('%d/%m/%Y')
        st.dataframe(tabella_mostra, use_container_width=True, hide_index=True)

# =====================================================================
# PAGINA 3: AI & PREVISIONI
# =====================================================================
elif pagina == "Analisi AI & Previsioni":
    st.title("Analisi Intelligente - Previsioni dell'IA")
    
    st.markdown("""
    <div class='insight-box'>
        <strong>Come funziona?</strong> Utilizziamo algoritmi di Machine Learning avanzati 
        per analizzare i tuoi 90 giorni di allenamento e fare previsioni accurate sul tuo stato di forma.
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.dati
    
    # ========== SEZIONE 1: RANDOM FOREST ==========
    st.subheader("⚠️ Previsione Rischio Infortunio (Random Forest)")
    
    st.markdown("""
    **Cosa significa?**
    
    La nostra IA analizza come i tuoi allenamenti passati si sono correlati a momenti di sovrallenamento.
    Utilizzando il modello Random Forest, prevediamo oggi quanto sei a rischio infortunio basandoci su:
    - Distanza e velocità della corsa
    - Qualità del sonno
    - Stress lavorativo e allenamento recente
    - Frequenza cardiaca e sforzo percepito
    """)
    
    X = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']]
    y = df['Rischio Infortunio']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    # Scenario medio
    prob_infortunio = rf.predict_proba([[10, 6.5, 6, 150, 6, 10]])[0][1] * 100
    
    col_risk, col_advice = st.columns([1.2, 1.8])
    
    with col_risk:
        colore_gauge = "#10b981" if prob_infortunio < 25 else "#f59e0b" if prob_infortunio < 60 else "#ef4444"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_infortunio,
            title={'text': "Rischio Infortunio (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colore_gauge},
                'steps': [
                    {'range': [0, 25], 'color': "#d1fae5"},
                    {'range': [25, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(height=400, paper_bgcolor='rgba(255, 255, 255, 0.95)')
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_advice:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if prob_infortunio < 25:
            st.success("""
            ✅ **TUTTO PERFETTO!**
            
            Il tuo corpo ha recuperato eccellentemente. Sei in perfette condizioni fisiche e mentali.
            
            **Cosa fare oggi:**
            - Puoi fare un allenamento intenso
            - Ottimo momento per lavoro intervallato
            - Prova a stabilire un nuovo PB!
            """)
        
        elif prob_infortunio < 60:
            st.warning(f"""
            ⚠️ **ATTENZIONE - Rischio Moderato ({prob_infortunio:.0f}%)**
            
            Stai accumulando fatica. Non è ancora critico, ma devi essere prudente.
            
            **Cosa fare oggi:**
            - Fai una corsa easy recovery
            - Niente velocità alta
            - Priorità al riposo e recupero
            - Migliora il sonno per le prossime notti
            """)
        
        else:
            st.error(f"""
            🛑 **RISCHIO CRITICO ({prob_infortunio:.0f}%)**
            
            I dati indicano altissimo rischio di infortunio o sovrallenamento!
            
            **COSA FARE:**
            - ❌ NON fare allenamenti intensi
            - ✅ Riposo totale o camminata leggera
            - ✅ Dormi di più (8+ ore)
            - ✅ Riduci lo stress lavorativo
            - ⚠️ Se il disagio persiste, consulta un medico
            """)
    
    st.markdown("---")
    
    # ========== SEZIONE 2: FEATURE IMPORTANCE ==========
    st.subheader("🔍 Cosa Influenza Maggiormente il Tuo Rischio Infortunio?")
    
    st.markdown("""
    Questo grafico mostra quali fattori hanno il maggiore impatto sul rischio infortunio:
    - **Barre più lunghe** = Fattore molto importante
    - **Barre più corte** = Fattore meno influente
    """)
    
    feature_importance = rf.feature_importances_
    feature_names = X.columns.tolist()
    
    fi_df = pd.DataFrame({
        'Fattore': feature_names,
        'Importanza': feature_importance
    }).sort_values('Importanza', ascending=True)
    
    fig_fi = px.barh(
        fi_df,
        x='Importanza',
        y='Fattore',
        color='Importanza',
        color_continuous_scale='Blues',
        title="Feature Importance - Fattori di Rischio",
        labels={'Importanza': 'Livello di Influenza', 'Fattore': ''},
        height=350
    )
    
    fig_fi.update_layout(
        plot_bgcolor='rgba(240, 248, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        showlegend=False,
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.markdown("---")
    
    # ========== SEZIONE 3: K-MEANS CLUSTERING ==========
    st.subheader("🎯 Tipologie di Allenamento (K-Means Clustering)")
    
    st.markdown("""
    **Cosa significa?**
    
    L'IA ha raggruppato i tuoi 90 allenamenti in 3 categorie distinte in base a come il tuo corpo ha risposto:
    """)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Tipo Allenamento'] = kmeans.fit_predict(df[['ISLR', 'FC Media']])
    
    tipo_nomi = {
        0: 'Corsa Rigenerante',
        1: 'Allenamento Intenso',
        2: 'Allenamento Stressato'
    }
    
    colori_tipo = ['#10b981', '#ef4444', '#f59e0b']
    
    df['Tipo Allenamento Nome'] = df['Tipo Allenamento'].map(tipo_nomi)
    
    fig_cluster = px.scatter(
        df,
        x="ISLR",
        y="FC Media",
        size="Distanza (km)",
        color="Tipo Allenamento Nome",
        color_discrete_sequence=colori_tipo,
        title="Classificazione dei Tuoi Allenamenti",
        labels={'ISLR': 'Impatto Stress Lavoro', 'FC Media': 'Frequenza Cardiaca Media (bpm)'},
        height=450,
        opacity=0.7,
        hover_data=['Distanza (km)', 'RPE', 'Ore Sonno']
    )
    
    fig_cluster.update_layout(
        plot_bgcolor='rgba(240, 248, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Statistiche per tipo
    col_tipo1, col_tipo2, col_tipo3 = st.columns(3)
    
    for idx, (tipo, nome) in enumerate(tipo_nomi.items()):
        df_tipo = df[df['Tipo Allenamento'] == tipo]
        conta = len(df_tipo)
        media_km = df_tipo['Distanza (km)'].mean()
        media_fc = df_tipo['FC Media'].mean()
        media_sonno = df_tipo['Ore Sonno'].mean()
        
        col = [col_tipo1, col_tipo2, col_tipo3][idx]
        with col:
            st.markdown(f"""
            <div class='metric-box' style='background: linear-gradient(135deg, {colori_tipo[idx]} 0%, {colori_tipo[idx]}dd 100%);'>
                <h3 style='margin: 0; color: white;'>{nome}</h3>
                <p style='font-size: 0.95em; margin: 8px 0;'>
                    <strong>{conta}</strong> allenamenti
                </p>
                <p style='font-size: 0.85em; margin: 4px 0; opacity: 0.95;'>
                    📏 {media_km:.1f} km | ❤️ {media_fc:.0f} bpm | 😴 {media_sonno:.1f}h
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== SEZIONE 4: REGRESSIONE LINEARE ==========
    st.subheader("📉 Previsione FC: Velocità vs Battito Cardiaco")
    
    st.markdown("""
    **Cosa mostra questo grafico?**
    
    La linea rossa è la previsione dell'IA: mostra come il tuo battito cardiaco aumenta proporzionalmente 
    alla velocità di corsa. Utilizza questa informazione per:
    - Stimare il carico cardiaco prima di una corsa
    - Tarare meglio le zone di allenamento
    - Evitare sovraccarichi indesiderati
    """)
    
    fig_reg = px.scatter(
        df,
        x="Velocità (km/h)",
        y="FC Media",
        trendline="ols",
        trendline_color_override="#ef4444",
        size="RPE",
        color="Ore Sonno",
        color_continuous_scale='RdYlGn_r',
        title="Velocità vs Frequenza Cardiaca - Trendline",
        labels={'Velocità (km/h)': 'Velocità Media (km/h)', 'FC Media': 'FC Media (bpm)'},
        height=450,
        opacity=0.7
    )
    
    fig_reg.update_layout(
        plot_bgcolor='rgba(240, 248, 255, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0.95)',
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_reg, use_container_width=True)
    
    # Calcolo coefficiente
    X_reg = df[['Velocità (km/h)']].values
    y_reg = df['FC Media'].values
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)
    
    st.markdown(f"""
    **Interpretazione della Regressione:**
    
    - **Formula:** FC = {lr.intercept_:.0f} + ({lr.coef_[0]:.2f} × Velocità)
    - **Significato:** Per ogni +1 km/h di velocità, il tuo battito aumenta di {lr.coef_[0]:.1f} bpm circa
    - **R² (Bontà del modello):** {lr.score(X_reg, y_reg):.2%}
    """)
    
    st.markdown("---")
    
    # ========== SEZIONE 5: CONSIGLI FINALI ==========
    st.subheader("💡 Consigli Personalizzati dell'IA")
    
    col_consiglio1, col_consiglio2 = st.columns(2)
    
    with col_consiglio1:
        st.markdown("""
        **Migliorare la Resistenza:**
        - Aumenta gradualmente i lunghi (max +10% a settimana)
        - Fai almeno 2-3 corse facili tra i lavori intensi
        - Priorità assoluta al sonno (7-9 ore)
        
        **Ridurre l'Infortunio:**
        - Corse easy recovery nei giorni con poco sonno
        - Warm-up adeguato (10 min) prima di allenamenti
        - Stretching dopo ogni corsa (10-15 min)
        """)
    
    with col_consiglio2:
        st.markdown("""
        **Bilanciare Lavoro e Allenamento:**
        - Giorni ad alto stress lavorativo = corse facili
        - Pianifica allenamenti intensi dopo giorni relax
        - Monitora il rapporto sforzo/recupero
        
        **Sfruttare le Condizioni Meteo:**
        - Giorni caldi = riduci pace, bevi più acqua
        - Giorni freddi = perfetti per il lavoro intenso
        - Vento forte = non per i lunghi
        """)

