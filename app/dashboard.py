import os
from glob import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="FairCut — Control Room",
    page_icon="🎬",
    layout="wide",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: bold; color: #E50914;}
    .sub-title {font-size: 1rem; color: #888888; margin-bottom: 2rem;}
    .metric-card {background-color: #1E1E1E; padding: 1rem; border-radius: 8px; border: 1px solid #333;}
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">🎬 FairCut — Bias Monitoring & MLOps</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Because not every bias deserves to make the cut.</div>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_historical_data():
    path = "data/processed/processed_movies.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_latest_predictions():
    files = sorted(glob("data/processed/predictions_*.parquet"))
    if files:
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        return df, os.path.basename(latest_file)
    return None, None


df_hist = load_historical_data()
df_pred, pred_filename = load_latest_predictions()

# Barre latérale
st.sidebar.image("https://img.icons8.com/color/96/movie-projector.png", width=80)
st.sidebar.title("Configuration ML")
st.sidebar.info("**Modèle actif :** XGBoost Baseline v1.0\n\n**Statut :** Production")

# Navigation Onglets
tab1, tab2, tab3 = st.tabs(["🚀 Live Predictions", "⚖️ Equity Radar (Fairness)", "📈 Model Performance"])

# ==================== TAB 1 : LIVE PREDICTIONS ====================
with tab1:
    if df_pred is not None:
        st.subheader(f"📊 Prédictions du lot quotidien (`{pred_filename}`)")

        col1, col2, col3 = st.columns(3)
        total_movies = len(df_pred)
        popular_count = int(df_pred["predicted_is_popular"].sum())
        ratio = (popular_count / total_movies) * 100 if total_movies > 0 else 0

        col1.metric("Films analysés", total_movies)
        col2.metric("Prédits 'Populaires'", popular_count)
        col3.metric("Taux de sélection", f"{ratio:.1f}%")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### Liste des films et prédictions")
            display_cols = [
                "title",
                "release_date",
                "original_language",
                "certification",
                "popularity_probability",
                "predicted_is_popular",
            ]
            st.dataframe(
                df_pred[display_cols].sort_values("popularity_probability", ascending=False),
                column_config={
                    "popularity_probability": st.column_config.ProgressColumn(
                        "Score de Popularité", format="%.2f", min_value=0, max_value=1
                    ),
                    "predicted_is_popular": st.column_config.CheckboxColumn("Popularity Hit ?"),
                },
                use_container_width=True,
            )

        with col_right:
            st.markdown("### Distribution des scores")
            fig = px.histogram(
                df_pred,
                x="popularity_probability",
                nbins=15,
                title="Répartition des probabilités attribuées",
                labels={"popularity_probability": "Probabilité de succès"},
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucun fichier de prédiction quotidien trouvé dans `data/processed/`.")

# ==================== TAB 2 : EQUITY RADAR ====================
with tab2:
    st.subheader("⚖️ Analyse de l'Équité Algorithmique (Fairlearn)")

    if df_hist is not None:
        col_f1, col_f2 = st.columns(2)

        # Disparate Impact Langues
        with col_f1:
            st.markdown("#### Biais de Langue (Anglophone vs Non-Anglophone)")
            lang_counts = (
                df_hist.groupby("protected_is_english")["is_popular"]
                .mean()
                .reset_index()
            )
            lang_counts["Language"] = lang_counts["protected_is_english"].map(
                {1: "Anglophone (EN)", 0: "Non-Anglophone"}
            )

            fig_lang = px.bar(
                lang_counts,
                x="Language",
                y="is_popular",
                title="Taux de sélection réel par langue",
                color="Language",
                color_discrete_map={"Anglophone (EN)": "#2E7D32", "Non-Anglophone": "#D32F2F"},
            )
            st.plotly_chart(fig_lang, use_container_width=True)

            # Calcul Disparate Impact
            sr_en = lang_counts.loc[lang_counts["protected_is_english"] == 1, "is_popular"].values[0]
            sr_non_en = lang_counts.loc[lang_counts["protected_is_english"] == 0, "is_popular"].values[0]
            di_lang = sr_non_en / sr_en if sr_en > 0 else 0

            if di_lang < 0.8:
                st.error(f"⚠️ **Alerte Disparate Impact : {di_lang:.2f}** (Inférieur au seuil légal de 0.80)")
            else:
                st.success(f"✅ **Disparate Impact Conforme : {di_lang:.2f}**")

        # Disparate Impact Studios
        with col_f2:
            st.markdown("#### Biais de Studio (Majors vs Indépendants)")
            major_counts = (
                df_hist.groupby("protected_is_major_studio")["is_popular"]
                .mean()
                .reset_index()
            )
            major_counts["Studio"] = major_counts["protected_is_major_studio"].map(
                {1: "Majors Hollywoodiennes", 0: "Indépendants"}
            )

            fig_major = px.bar(
                major_counts,
                x="Studio",
                y="is_popular",
                title="Taux de sélection réel par type de studio",
                color="Studio",
                color_discrete_map={"Majors Hollywoodiennes": "#2E7D32", "Indépendants": "#D32F2F"},
            )
            st.plotly_chart(fig_major, use_container_width=True)

            sr_maj = major_counts.loc[major_counts["protected_is_major_studio"] == 1, "is_popular"].values[0]
            sr_ind = major_counts.loc[major_counts["protected_is_major_studio"] == 0, "is_popular"].values[0]
            di_major = sr_ind / sr_maj if sr_maj > 0 else 0

            if di_major < 0.8:
                st.error(f"⚠️ **Alerte Disparate Impact : {di_major:.2f}** (Inférieur au seuil de 0.80)")
            else:
                st.success(f"✅ **Disparate Impact Conforme : {di_major:.2f}**")
    else:
        st.warning("Données historiques indisponibles pour afficher l'Equity Radar.")

# ==================== TAB 3 : MLOPS & HEALTH ====================
with tab3:
    st.subheader("📈 Suivi du Modèle & Intégrité")
    st.info("Le tracking complet des expériences est enregistré sous **MLflow**.")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Modèle Champion", "XGBoost v1.0")
    col_m2.metric("F1-Score Baseline", "0.5023")
    col_m3.metric("ROC-AUC Baseline", "0.6862")