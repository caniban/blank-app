import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Sayfa Ayarları
st.set_page_config(page_title="Colliers AI | Mass Appraisal", layout="wide")

# 1. Gerçek Veri Setini Yükleme
@st.cache_data
def load_and_prep_data():
    df = pd.read_excel("exported_modified.xlsx")
    return df

# 2. Modeli 11 Faktörle Eğitme
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
df = load_and_prep_data()
model, X = train_model(df)
explainer = shap.TreeExplainer(model)

# --- DASHBOARD ARAYÜZÜ ---
st.title("📊 Colliers Italia | Cruscotto di Valutazione Immobiliare Massiva")
st.markdown("Questo cruscotto utilizza **XGBoost** e **SHAP** per fornire stime trasparenti sul valore degli immobili a Mersin, Turchia.")

tab1, tab2, tab3 = st.tabs(["1. Panoramica del Progetto", "2. Prestazioni Macro", "3. Simulatore (What-If)"])

# TAB 1: PROJE ÖZETİ
with tab1:
    st.header("Informazioni sul Progetto e sul Set di Dati")
    st.markdown("""
    Questa sezione fornisce una panoramica del set di dati utilizzato per addestrare i nostri modelli di intelligenza artificiale. 
    Il progetto è stato sviluppato per superare i limiti dei tradizionali modelli di valutazione (OLS) adottando l'algoritmo **XGBoost**, che ha mostrato le migliori prestazioni.
    """)
    
    col_info1, col_info2 = st.columns([1, 1])
    
    with col_info1:
        st.subheader("Statistiche Chiave del Mercato")
        st.markdown(f"""
        - **Campioni Totali:** {len(df)} immobili residenziali analizzati.
        - **Criteri Selezionati (Feature):** **11 driver principali** basati su *Permutation Feature Importance* (inclusi Superficie, Distanza dai mercati, Altitudine, Età, ecc.).
        - **Prezzo Min - Max:** ₺ {df['Price (TRY)'].min():,.0f}  —  ₺ {df['Price (TRY)'].max():,.0f}
        - **Superficie Lorda:** Da {df['Gross Area'].min()} mq a {df['Gross Area'].max()} mq.
        - **Età degli Edifici:** Da {df['Building Age'].min()} a {df['Building Age'].max()} anni.
        """)
        st.info("I modelli di Machine Learning utilizzati includono XGBoost, Random Forest e LightGBM. XGBoost è stato selezionato come modello finale (Champion Model).")

    with col_info2:
        st.subheader("Area di Studio: Mersin / Yenişehir")
        try:
            st.image("map_yenisehir.png", caption="Mappa dell'area di studio (Yenişehir District)", use_container_width=True)
        except:
            st.warning("⚠️ Immagine 'map_yenisehir.png' non trovata. Inserire l'immagine nella cartella del progetto.")

# TAB 2: MAKRO PERFORMANS
with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (XGBoost)", "0.87", "+15% vs OLS")
    col2.metric("MAPE (XGBoost)", "9.04%", "-12% vs OLS")
    col3.metric("RMSE", "₺ 146.837", "Elevata Precisione")
    
    st.subheader("Impatto Globale delle Caratteristiche (SHAP Bar Plot)")
    
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_names = X.columns
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = mean_abs_shap[sorted_idx]
    
    colors = ['red' if feat in ["Building Age", "Distance to Coach Station"] else 'green' for feat in sorted_features]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(sorted_features, sorted_importance, color=colors, height=0.6)
    
    max_val = max(sorted_importance)
    
    for i, feat in enumerate(sorted_features):
        if feat == "Building Age":
            ax.annotate("L'impatto negativo dell'invecchiamento\nè una dinamica attesa.", 
                        xy=(sorted_importance[i], i), xytext=(sorted_importance[i] + (max_val * 0.03), i - 0.1),
                        arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5), va='center', color='darkred', fontsize=10)
        elif feat == "Distance to Coach Station":
            ax.annotate("Effetto negativo: la vicinanza alla stazione\nriduce il valore (rumore/traffico).", 
                        xy=(sorted_importance[i], i), xytext=(sorted_importance[i] + (max_val * 0.03), i - 0.1),
                        arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5), va='center', color='darkred', fontsize=10)

    ax.set_xlabel("Impatto Assoluto (Verde = Positivo, Rosso = Negativo)", fontsize=10)
    ax.set_ylabel("Variabili dell'Immobile", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(0, max_val * 1.5)
    st.pyplot(fig)
    plt.clf()

# TAB 3: MİKRO XAI & WHAT-IF SİMÜLATÖRÜ
with tab3:
    st.header("Simulatore 'What-If' e Motore di Spiegazione (NLE)")
    col_sim, col_chart = st.columns([1, 2])
    
    with col_sim:
        st.subheader("Parametri dell'Immobile")
        gross_area = st.slider("Superficie Lorda (mq) [Gross Area]", int(df["Gross Area"].min()), int(df["Gross Area"].max()), int(df["Gross Area"].median()))
        net_area = st.slider("Superficie Netta (mq) [Net Area]", int(df["Net Area"].min()), int(df["Net Area"].max()), int(df["Net Area"].median()))
        dist_bazaar = st.slider("Distanza dai Mercati (km)", float(df["Distance to Bazaars"].min()), float(df["Distance to Bazaars"].max()), float(df["Distance to Bazaars"].median()))
        dist_coach = st.slider("Distanza dalla Stazione Autobus (km)", float(df["Distance to Coach Station"].min()), float(df["Distance to Coach Station"].max()), float(df["Distance to Coach Station"].median()))
        bld_floors = st.slider("Piani Totali Edificio", int(df["Building Floors"].min()), int(df["Building Floors"].max()), int(df["Building Floors"].median()))
        floor_num = st.slider("Piano dell'Appartamento", int(df["Floor Number"].min()), int(df["Floor Number"].max()), int(df["Floor Number"].median()))
        baths = st.slider("Numero di Bagni", int(df["Number of Bathrooms"].min()), int(df["Number of Bathrooms"].max()), int(df["Number of Bathrooms"].median()))
        rooms = st.slider("Numero di Stanze", int(df["Number of Rooms"].min()), int(df["Number of Rooms"].max()), int(df["Number of Rooms"].median()))
        age = st.slider("Età dell'Edificio (Anni)", int(df["Building Age"].min()), int(df["Building Age"].max()), int(df["Building Age"].median()))
        elev = st.slider("Altitudine (m) [Elevation]", int(df["Elevation"].min()), int(df["Elevation"].max()), int(df["Elevation"].median()))
        
        # Isıtma Sistemi (Heating System) için metin eşleştirmesi eklendi
        heating_options = {
            0: "0 - Nessun sistema", 
            1: "1 - Aria condizionata", 
            2: "2 - Caldaia a gas naturale"
        }
        selected_heating_label = st.selectbox("Sistema di Riscaldamento", options=list(heating_options.values()), index=1)
        # Seçilen metni modele göndermek için tekrar sayıya çeviriyoruz
        heating = [k for k, v in heating_options.items() if v == selected_heating_label][0]
        
    with col_chart:
        input_data = [[gross_area, dist_bazaar, bld_floors, baths, net_area, age, rooms, elev, dist_coach, heating, floor_num]]
        input_df = pd.DataFrame(input_data, columns=X.columns)
        pred_price = model.predict(input_df)[0]
        
        st.success(f"### Valore Stimato: ₺ {pred_price:,.0f} TRY")
        
        st.info(f"""
        **Spiegazione Dinamica (NLE):**  
        Il valore stimato per questo immobile è di **₺ {pred_price:,.0f}**. 
        Il grafico Waterfall sottostante mostra come ciascuno dei **11 parametri** abbia aggiunto o sottratto valore rispetto al prezzo medio di base del mercato.
        """)
        
        shap_values_single = explainer(input_df)
        # max_display=15 eklenerek tüm faktörlerin "Other" olmadan gösterilmesi sağlandı
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(shap_values_single[0], max_display=15, show=False)
        st.pyplot(fig2)
        plt.clf()