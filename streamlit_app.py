import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sayfa Ayarları
st.set_page_config(page_title="Colliers AI | Mass Appraisal & Advisory", layout="wide")

# 1. Gerçek Veri Setini Yükleme
@st.cache_data
def load_and_prep_data():
    df_model = pd.read_excel("exported_modified.xlsx")
    df_map = pd.read_csv("yenisehir_numuneler.csv")
    
    df_map = df_map.dropna(subset=['latitude', 'longitude'])
    df_map['Unit_Price'] = (df_map['fiyat'] / df_map['brut']).astype(int)
    
    return df_model, df_map

# 2. Modeli Eğitme
@st.cache_resource
def train_model(df):
    features = [
        "Gross Area", "Distance to Bazaars", "Building Floors", 
        "Number of Bathrooms", "Net Area", "Building Age", 
        "Number of Rooms", "Elevation", "Distance to Coach Station", 
        "Heating System", "Floor Number"
    ]
    X = df[features]
    y = df["Price (TRY)"]
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model, X

# Veri ve Modeli Yükle
df_model, df_map = load_and_prep_data()
model, X = train_model(df_model)
explainer = shap.TreeExplainer(model)

# --- DASHBOARD ARAYÜZÜ ---
st.title("📊 Colliers Italia | Valutazione Immobiliare Massiva & Advisory")
st.markdown("Integrazione di **Machine Learning (XGBoost/SHAP)** e **Business Intelligence Demografica** per consulenze immobiliari strategiche.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Panoramica & Mappa", 
    "2. Prestazioni Macro & XAI", 
    "3. Simulatore (What-If)",
    "4. Dati Socio-Economici (BI)",
    "5. Consulenza Strategica"
])

# TAB 1: PANORAMA VE ESRI HARİTA (GÜNCELLENDİ)
with tab1:
    st.header("Contesto di Mercato")
    
    # Mersin ve Yenişehir Pazarı Hakkında Bilgi
    st.markdown("""
    **📍 Analisi del Territorio: Mersin e Yenişehir**  
    Mersin è uno dei poli logistici, commerciali e portuali più dinamici del Mediterraneo orientale. All'interno di questa metropoli, il distretto di **Yenişehir** rappresenta il cuore moderno e in rapida espansione urbana. 
    Caratterizzato da un'alta concentrazione di campus universitari, centri commerciali (Malls), ampi viali e una lunga costa sul Mediterraneo, Yenişehir attira principalmente investitori e residenti con un profilo socio-economico medio-alto (White-collar, accademici e professionisti). 
    *Questo rende il mercato immobiliare locale altamente reattivo ai parametri fisici e spaziali (vista, spazi ampi, vicinanza ai servizi).*
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.subheader("Statistiche del Dataset")
        st.markdown(f"""
        - **Campioni Totali:** {len(df_model)}
        - **Prezzo (Min-Max):** ₺ {df_model['Price (TRY)'].min():,.0f} — ₺ {df_model['Price (TRY)'].max():,.0f}
        - **Superficie:** {df_model['Gross Area'].min()} mq — {df_model['Gross Area'].max()} mq
        - **Età Edifici:** {df_model['Building Age'].min()} — {df_model['Building Age'].max()} anni
        - **Variabili utilizzate:** 11 determinanti critiche del valore (Superficie lorda, Superficie netta, Distanza dai mercati, Distanza dalla stazione degli autobus, Piani dell'edificio, Numero del piano, Numero di bagni, Numero di stanze, Età dell'edificio, Altitudine, Sistema di riscaldamento).
        """)
        st.info("💡 **Legenda Mappa:** La mappa interattiva mostra il prezzo al metro quadrato. I colori freddi (Blu) indicano prezzi bassi, mentre i colori caldi (Rossi) indicano valori premium. Puoi usare lo scroll per lo zoom.")
        st.info("🤖 **Modelli di Machine Learning:** I modelli utilizzati includono XGBoost, Random Forest e LightGBM. XGBoost è stato selezionato come modello finale (Champion Model).")

    with col2:
        # İtalyanca Tooltip ve Lejant ayarlamaları
        labels_dict = {
            "Unit_Price": "Prezzo/mq (₺)", 
            "fiyat": "Prezzo Totale (₺)", 
            "brut": "Superficie Lorda (mq)", 
            "mahalle": "Quartiere"
        }
        
        try:
            fig_map = px.scatter_map(
                df_map, lat="latitude", lon="longitude",
                color="Unit_Price", color_continuous_scale="RdYlBu_r",
                size="brut", size_max=15,
                hover_name="mahalle", 
                hover_data={"latitude": False, "longitude": False, "Unit_Price": True, "fiyat": True, "brut": True},
                labels=labels_dict,
                title="Distribuzione Spaziale (Dimensione punto = Superficie Lorda)"
            )
            fig_map.update_layout(
                map_style="white-bg",
                map_layers=[{"below": 'traces', "sourcetype": "raster", "sourceattribution": "Esri", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
                margin={"r":0,"t":40,"l":0,"b":0}, height=500, map_zoom=12, map_center={"lat": df_map['latitude'].mean(), "lon": df_map['longitude'].mean()}
            )
        except AttributeError:
            fig_map = px.scatter_mapbox(
                df_map, lat="latitude", lon="longitude",
                color="Unit_Price", color_continuous_scale="RdYlBu_r",
                size="brut", size_max=15,
                hover_name="mahalle", 
                hover_data={"latitude": False, "longitude": False, "Unit_Price": True, "fiyat": True, "brut": True},
                labels=labels_dict,
                title="Distribuzione Spaziale (Dimensione punto = Superficie Lorda)"
            )
            fig_map.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[{"below": 'traces', "sourcetype": "raster", "sourceattribution": "Esri", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}],
                margin={"r":0,"t":40,"l":0,"b":0}, height=500, mapbox_zoom=12, mapbox_center={"lat": df_map['latitude'].mean(), "lon": df_map['longitude'].mean()}
            )
            
        st.plotly_chart(fig_map, use_container_width=True)

# TAB 2: MAKRO & XAI AÇIKLAMASI
with tab2:
    st.header("L'Evoluzione dell'AI: Da Black Box a Glass Box")
    
    col_xai1, col_xai2 = st.columns([1, 1.2])
    with col_xai1:
        try:
            st.image("ai_spiegabile.png", caption="Confronto Architetturale: Tradizionale vs Spiegabile", use_container_width=True)
        except:
            st.warning("Immagine 'ai_spiegabile.png' non trovata.")
            
    with col_xai2:
        st.markdown("### Il passaggio da 'Fidati di me' a 'Verificami'")
        st.markdown("""
        | Parametro | AI Tradizionale (Black Box) | Explainable AI (XAI con SHAP) |
        | :--- | :--- | :--- |
        | **Approccio** | 'Fidati di me' | 'Verificami' |
        | **Output** | Singolo valore predittivo senza contesto | Scomposizione granulare di ogni variabile |
        | **Fiducia dell'Investitore** | Bassa per operazioni ad alto capitale | Elevata: il modello giustifica la scelta |
        | **Ruolo dell'Esperto** | Sostituito dalla macchina | Amplificato (Augmented Intelligence) |
        """)
        
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (XGBoost)", "0.87", "+15% vs OLS")
    col2.metric("MAPE (XGBoost)", "9.04%", "-12% vs OLS")
    with col3:
        st.metric("RMSE", "₺ 146.837", "Elevata Precisione")
        st.markdown(
            '<p style="display: inline-block; color: #a84400; background-color: #ffe0b2; '
            'font-size: 0.85rem; font-weight: 600; padding: 0.35rem 0.75rem; '
            'border-radius: 999px; margin-top: 0.25rem;">'
            'Margine di errore sui valori totali di vendita: circa 7.000 €.'
            '</p>',
            unsafe_allow_html=True
        )
    
    st.subheader("Impatto Globale delle Caratteristiche (SHAP Bar Plot)")
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_names = X.columns
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = mean_abs_shap[sorted_idx]
    
    colors = ['red' if feat in ["Building Age", "Distance to Coach Station"] else 'green' for feat in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(sorted_features, sorted_importance, color=colors, height=0.6)
    
    max_val = max(sorted_importance)
    for i, feat in enumerate(sorted_features):
        if feat == "Building Age":
            ax.annotate("Effetto Negativo (Invecchiamento)", xy=(sorted_importance[i], i), xytext=(sorted_importance[i]+(max_val*0.03), i), color='darkred', va='center')
        elif feat == "Distance to Coach Station":
            ax.annotate("Effetto Negativo (Rumore)", xy=(sorted_importance[i], i), xytext=(sorted_importance[i]+(max_val*0.03), i), color='darkred', va='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max_val * 1.5)
    st.pyplot(fig)
    plt.clf()

# TAB 3: SİMÜLATÖR
with tab3:
    st.header("Simulatore 'What-If'")
    col_sim, col_chart = st.columns([1, 2])
    with col_sim:
        gross_area = st.slider("Superficie Lorda (mq)", int(df_model["Gross Area"].min()), int(df_model["Gross Area"].max()), int(df_model["Gross Area"].median()))
        net_area = st.slider("Superficie Netta (mq)", int(df_model["Net Area"].min()), int(df_model["Net Area"].max()), int(df_model["Net Area"].median()))
        dist_bazaar = st.slider("Distanza dai Mercati (km)", float(df_model["Distance to Bazaars"].min()), float(df_model["Distance to Bazaars"].max()), float(df_model["Distance to Bazaars"].median()))
        dist_coach = st.slider("Distanza Staz. Autobus (km)", float(df_model["Distance to Coach Station"].min()), float(df_model["Distance to Coach Station"].max()), float(df_model["Distance to Coach Station"].median()))
        bld_floors = st.slider("Piani Totali", int(df_model["Building Floors"].min()), int(df_model["Building Floors"].max()), int(df_model["Building Floors"].median()))
        floor_num = st.slider("Piano Appartamento", int(df_model["Floor Number"].min()), int(df_model["Floor Number"].max()), int(df_model["Floor Number"].median()))
        baths = st.slider("Numero Bagni", int(df_model["Number of Bathrooms"].min()), int(df_model["Number of Bathrooms"].max()), int(df_model["Number of Bathrooms"].median()))
        rooms = st.slider("Numero Stanze", int(df_model["Number of Rooms"].min()), int(df_model["Number of Rooms"].max()), int(df_model["Number of Rooms"].median()))
        age = st.slider("Età Edificio (Anni)", int(df_model["Building Age"].min()), int(df_model["Building Age"].max()), int(df_model["Building Age"].median()))
        elev = st.slider("Altitudine (m)", int(df_model["Elevation"].min()), int(df_model["Elevation"].max()), int(df_model["Elevation"].median()))
        
        heating_options = {0: "0 - Nessun sistema", 1: "1 - Aria condizionata", 2: "2 - Caldaia a gas"}
        selected_heating_label = st.selectbox("Riscaldamento", list(heating_options.values()), index=1)
        heating = [k for k, v in heating_options.items() if v == selected_heating_label][0]
        
    with col_chart:
        input_df = pd.DataFrame([[gross_area, dist_bazaar, bld_floors, baths, net_area, age, rooms, elev, dist_coach, heating, floor_num]], columns=X.columns)
        pred = model.predict(input_df)[0]
        st.success(f"### Valore Stimato: ₺ {pred:,.0f} TRY")
        
        shap_values_single = explainer(input_df)
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(shap_values_single[0], max_display=15, show=False)
        st.pyplot(fig2)
        plt.clf()

# TAB 4: BI (GENİŞLETİLMİŞ GRAFİKLER)
with tab4:
    st.header("Business Intelligence: Profilazione Socio-Economica Avanzata")
    st.markdown("Analisi incrociata dei dati censuari (istruzione, demografia, transazioni immobiliari) per i tre quartieri target.")
    
    # 1. SATIR: LOLLIPOP GRAFİĞİ (ORTALAMA METREKARE FİYATI)
    target_hoods = {
        "fuatmorel": "Fuatmorel",
        "batikent": "Batıkent",
        "ciftlikkoy": "Çiftlikköy",
    }
    df_map['mahalle_key'] = df_map['mahalle'].astype(str).str.strip().str.lower()
    hood_stats = (
        df_map[df_map['mahalle_key'].isin(target_hoods)]
        .groupby('mahalle_key', as_index=False)['Unit_Price']
        .agg(min_price='min', mean_price='mean', max_price='max')
        .assign(mahalle=lambda data: data['mahalle_key'].map(target_hoods))
        .set_index('mahalle_key')
        .reindex(target_hoods)
        .reset_index()
    )
    
    fig_lollipop = go.Figure()
    fig_lollipop.add_trace(go.Scatter(
        x=hood_stats['mean_price'], y=hood_stats['mahalle'],
        mode='markers+text', marker=dict(color='#ff7f0e', size=20),
        text=hood_stats['mean_price'].round(0).astype(int), textposition="top center",
        name='Prezzo Medio/mq',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Min: ₺ %{customdata[0]:,.0f}/mq<br>'
            'Medio: ₺ %{x:,.0f}/mq<br>'
            'Max: ₺ %{customdata[1]:,.0f}/mq<extra></extra>'
        ),
        customdata=hood_stats[['min_price', 'max_price']]
    ))
    for i, row in hood_stats.iterrows():
        fig_lollipop.add_shape(
            type='line', x0=row['min_price'], y0=row['mahalle'],
            x1=row['max_price'], y1=row['mahalle'],
            line=dict(color='gray', width=4)
        )
        
    fig_lollipop.update_layout(
        title="Minimo, Medio e Massimo al mq (₺)",
        height=360,
        margin=dict(t=80, r=30, b=40, l=30),
        yaxis_title=""
    )
    st.plotly_chart(fig_lollipop, use_container_width=True)
    
    st.divider()
    
    # 2. SATIR: YAŞ DAĞILIMI (3 AYRI DONUT GRAFİĞİ)
    st.subheader("Distribuzione per Fasce d'Età")
    age_data = pd.DataFrame({
        "Fascia d'Età": ["0-14", "15-29", "30-44", "45-59", "60+"],
        "Fuatmorel": [3056, 2321, 2642, 2537, 1295],
        "Batıkent": [3208, 2652, 3524, 2500, 1592],
        "Çiftlikköy": [4305, 15292, 6977, 4527, 2472]
    })
    
    col_age1, col_age2, col_age3 = st.columns(3)
    colors_age = px.colors.qualitative.Pastel
    
    with col_age1:
        fig_age1 = px.pie(age_data, values="Fuatmorel", names="Fascia d'Età", hole=0.5, title="Fuatmorel", color_discrete_sequence=colors_age)
        st.plotly_chart(fig_age1, use_container_width=True)
    with col_age2:
        fig_age2 = px.pie(age_data, values="Batıkent", names="Fascia d'Età", hole=0.5, title="Batıkent", color_discrete_sequence=colors_age)
        st.plotly_chart(fig_age2, use_container_width=True)
    with col_age3:
        fig_age3 = px.pie(age_data, values="Çiftlikköy", names="Fascia d'Età", hole=0.5, title="Çiftlikköy", color_discrete_sequence=colors_age)
        st.plotly_chart(fig_age3, use_container_width=True)
        
    st.info("💡 **Insight:** Notare l'esplosione della fascia 15-29 anni a Çiftlikköy, indicativa di una forte concentrazione studentesca (Campus Universitario).")

    st.divider()

    # 3. SATIR: EĞİTİM, MEDENİ DURUM VE GAYRİMENKUL AKTİVİTESİ
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.subheader("Livello di Istruzione (Superiore)")
        edu_data = pd.DataFrame({
            "Livello": ["Superiore", "Laurea", "Master", "Dottorato"],
            "Fuatmorel": [2377, 2627, 572, 150],
            "Batıkent": [2851, 3154, 485, 61],
            "Çiftlikköy": [14145, 7433, 1472, 300]
        }).melt(id_vars="Livello", var_name="Quartiere", value_name="Persone")
        
        fig_edu = px.bar(edu_data, x="Livello", y="Persone", color="Quartiere", barmode="group",
                         color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"])
        st.plotly_chart(fig_edu, use_container_width=True)
        
        st.subheader("Stato Civile")
        marital_data = pd.DataFrame({
            "Stato Civile": ["Sposato", "Single", "Divorziato", "Vedovo"],
            "Fuatmorel": [5614, 2349, 565, 267],
            "Batıkent": [6388, 2808, 653, 419],
            "Çiftlikköy": [10077, 16351, 2227, 613]
        }).melt(id_vars="Stato Civile", var_name="Quartiere", value_name="Totale")
        
        fig_mar = px.bar(marital_data, x="Quartiere", y="Totale", color="Stato Civile", barmode="stack",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_mar, use_container_width=True)

    with col_b2:
        st.subheader("Dinamiche Immobiliari (Turnover & Vendite)")
        re_data = pd.DataFrame({
            "Quartiere": ["Fuatmorel", "Batıkent", "Çiftlikköy"],
            "Turnover (%)": [5.72, 5.48, 7.21],
            "Vendite Annuali": [244, 270, 1252]
        })
        
        # Çift Eksenli Grafik (Vendite & Turnover)
        fig_re = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_re.add_trace(go.Bar(x=re_data["Quartiere"], y=re_data["Vendite Annuali"], name="Vendite Annuali (Volumi)", marker_color='#9467bd'), secondary_y=False)
        fig_re.add_trace(go.Scatter(x=re_data["Quartiere"], y=re_data["Turnover (%)"], name="Turnover - Tasso di rotazione (%)", mode="lines+markers", marker_color='red', marker_size=10, line_width=3), secondary_y=True)
        
        fig_re.update_yaxes(title_text="Volume Vendite", secondary_y=False)
        fig_re.update_yaxes(title_text="Turnover (%)", secondary_y=True)
        fig_re.update_layout(title_text="Liquidità del Mercato")
        st.plotly_chart(fig_re, use_container_width=True)

        st.subheader("Profilo Socioeconomico dei Quartieri")
        socioeconomico_data = pd.DataFrame({
            "Gruppo": ["A+", "A", "B", "C", "D"],
            "Fuatmorel": [1318, 2275, 2536, 4158, 1564],
            "Batıkent": [1258, 2745, 3089, 4643, 1741],
            "Çiftlikköy": [3499, 6623, 15593, 5919, 1939]
        }).melt(id_vars="Gruppo", var_name="Quartiere", value_name="Persone")
        socioeconomico_data["Percentuale"] = (
            socioeconomico_data["Persone"]
            / socioeconomico_data.groupby("Quartiere")["Persone"].transform("sum")
            * 100
        )

        fig_socioeconomico = px.bar(
            socioeconomico_data,
            x="Percentuale",
            y="Quartiere",
            color="Gruppo",
            orientation="h",
            barmode="stack",
            text=socioeconomico_data["Percentuale"].round(1).astype(str) + "%",
            custom_data=["Persone"],
            color_discrete_map={
                "A+": "#173f5f",
                "A": "#20639b",
                "B": "#3caea3",
                "C": "#f6d55c",
                "D": "#ed553b"
            },
            labels={"Percentuale": "Composizione (%)", "Gruppo": "Status socioeconomico"},
            title="Composizione socioeconomica (quota percentuale)"
        )
        fig_socioeconomico.update_traces(
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Gruppo: %{fullData.name}<br>"
                "Quota: %{x:.1f}%<br>"
                "Persone: %{customdata[0]:,}<extra></extra>"
            )
        )
        fig_socioeconomico.update_layout(
            height=400,
            xaxis=dict(range=[0, 100], ticksuffix="%"),
            legend_title_text="Gruppo"
        )
        st.plotly_chart(fig_socioeconomico, use_container_width=True)

# TAB 5: YATIRIM DANIŞMANLIĞI (İLERİ ANALİZ & GÖRSELLEŞTİRME)
with tab5:
    st.header("Consulenza Strategica: Profilo Cliente & Ottimizzazione")
    
    # Detaylandırılmış Müşteri Profili (Brief)
    st.info("""
    📋 **Dossier Cliente (Client Brief):**
    *   **Composizione:** Famiglia di 3 persone (Coppia di professionisti con un bambino in età scolare).
    *   **Istruzione & Lavoro:** Entrambi laureati, impiegati di fascia alta (Categoria Socio-Economica "Gruppo A").
    *   **Esigenze di Spazio:** Ricerca di un immobile ampio (minimo 150 mq, preferibilmente 180-250 mq) per necessità di smart-working e comfort familiare.
    *   **Vincolo Economico:** Pur avendo un buon reddito, cercano il miglior rapporto qualità-prezzo (Budget target: Max 1.5M - 2M ₺).
    *   **Lifestyle & Vicinato:** Vogliono vivere in un quartiere tranquillo, lontano dal caos studentesco o industriale, circondati da un *peer group* (persone con simile background socio-culturale) e vicini a buone scuole.
    """)
    
    col_a1, col_a2 = st.columns([1.3, 1])
    
    with col_a1:
        st.markdown("### 🎯 Ricerca dello 'Sweet Spot' (Area Ideale)")
        # Dağılım Grafiği - Eksenler sınırlandırıldı ve hedef kutu (Target) netleştirildi
        fig_budget = px.scatter(
            df_model, x="Gross Area", y="Price (TRY)", color="Building Age",
            color_continuous_scale="Viridis", opacity=0.8,
            labels={"Gross Area": "Superficie Lorda (mq)", "Price (TRY)": "Prezzo (₺)", "Building Age": "Età Edificio (Anni)"}
        )
        # Eksenleri zoom'lama (Verinin yoğun olduğu ve müşterinin aradığı aralık)
        fig_budget.update_xaxes(range=[0, 350])
        fig_budget.update_yaxes(range=[0, 3500000])
        
        # Müşterinin bütçe ve alan hedefini çizen dikdörtgen
        fig_budget.add_shape(type="rect",
            x0=150, y0=df_model["Price (TRY)"].min(), x1=250, y1=1800000,
            line=dict(color="Red", width=3, dash="solid"),
            fillcolor="rgba(255, 0, 0, 0.15)",
        )
        fig_budget.add_annotation(x=200, y=1900000, text="🎯 Target Famiglia", showarrow=False, font=dict(color="red", size=14, weight="bold"))
        
        st.plotly_chart(fig_budget, use_container_width=True)
        st.markdown("""
        **Analisi XAI (Trade-off):** Il rettangolo rosso mostra lo spazio di ricerca ideale (150-250 mq sotto gli 1.8M ₺). I punti all'interno del target tendono ad avere colori chiari (Giallo/Verde), il che significa che per soddisfare questa equazione il cliente deve accettare un **Edificio più vecchio (10-20 anni)**. L'intelligenza artificiale dimostra che l'età è la leva negoziale perfetta per abbassare il prezzo senza sacrificare i metri quadrati.
        """)

    with col_a2:
        st.markdown("### 🏆 Matrice di Decisione (Quartieri)")
        st.markdown("Valutazione dei 3 quartieri rispetto ai rigidi criteri del cliente (Punteggio 1-10).")
        
        # Anlaşılmayan Funnel yerine çok net bir Heatmap (Karar Matrisi)
        decision_data = pd.DataFrame({
            "Criteri": ["Ambiente Familiare (Sposati)", "Profilo Elite (Gruppo A/Reddito)", "Tranquillità (Bassa Densità)", "Budget-Friendly (Costo/mq)"],
            "Fuatmorel": [9, 10, 10, 6],
            "Batıkent": [9, 7, 5, 8],
            "Çiftlikköy": [3, 9, 8, 8] # Çiftlikköy evli puanı düşük (bekar çok)
        })
        decision_data = decision_data.set_index("Criteri")
        
        fig_heatmap = px.imshow(
            decision_data, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="Greens",
            labels=dict(x="Quartiere", y="Criterio Cliente", color="Punteggio")
        )
        fig_heatmap.update_xaxes(side="top")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.success("""
        **Verdetto Finale (Takeaway):**
        Il quartiere vincitore è **Fuatmorel**. Sebbene abbia un costo al mq leggermente superiore, offre il punteggio perfetto (10/10) per densità, reddito e ambiente familiare. 
        **Il consiglio d'oro per il broker Colliers:** Proporre a questa famiglia un appartamento spazioso (>180mq) a Fuatmorel, costruito tra i 12 e i 18 anni fa. In questo modo si rispetta il budget sfruttando il deprezzamento fisiologico dell'immobile, garantendo al contempo la massima qualità di vita socio-culturale.
        """)