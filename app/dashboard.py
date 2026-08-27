from datetime import datetime
import json
import os
from glob import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(
    page_title="FairCut — Control Room",
    page_icon=":material/movie:",
    layout="wide",
)

# Header
st.title(" :material/movie: FairCut — Bias Monitoring & MLOps Control Room")
st.caption("Because not every bias deserves to make the cut.")


@st.cache_data
def load_historical_data():
    path = "data/processed/processed_movies.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None



def load_latest_predictions():
    files = sorted(glob("data/processed/predictions_*.parquet"))
    if files:
        latest_file = files[-1]
        df = pd.read_parquet(latest_file)
        mtime = os.path.getmtime(latest_file)
        run_time = datetime.fromtimestamp(mtime)
        return df, os.path.basename(latest_file), run_time
    return None, None, None


def load_latest_monitoring_report():
    files = sorted(glob("data/monitoring/drift_report_*.html"))
    if files:
        latest_report = files[-1]
        with open(latest_report, "r", encoding="utf-8") as f:
            return f.read(), os.path.basename(latest_report)
    return None, None


# Lecture dynamique des métadonnées du modèle (Metadata JSON Registry)
def load_model_metadata():
    metadata_path = "models/metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "model_version": "v1.0",
        "model_name": "XGBoost Baseline",
        "status": "Production",
        "f1_score": 0.5023,
        "disparate_impact": 0.6500,
    }


df_hist = load_historical_data()
df_pred, pred_filename, last_run_datetime = load_latest_predictions()
drift_html, drift_filename = load_latest_monitoring_report()

# Récupération dynamique des variables issues du JSON
model_meta = load_model_metadata()
active_version = model_meta.get("model_version", "v1.0")
active_model_name = model_meta.get("model_name", "XGBoost Baseline")
active_f1 = model_meta.get("f1_score", 0.5023)
active_di = model_meta.get("disparate_impact", 0.6500)

# ==================== BARRE LATÉRALE ====================
st.sidebar.image("https://img.icons8.com/isometric/50/bot.png", width=80)
st.sidebar.title("Configuration ML")

# 1. Heure en live
now = datetime.now()
st.sidebar.caption(f" :material/schedule: **Heure actuelle :** {now.strftime('%d/%m/%Y')} — `{now.strftime('%H:%M:%S')}`")

st.sidebar.markdown("---")

# 2. Informations du Modèle & Dernier Run Daily Pipeline (DYNAMIQUE)
if last_run_datetime is not None:
    formatted_run = last_run_datetime.strftime("%d/%m/%Y à %H:%M")
    st.sidebar.info(
        f"**Modèle actif :** {active_model_name} `{active_version}`\n\n"
        f"**Statut :** Production (Batch Daily)\n\n"
        f" :material/rocket_launch: **Dernier run `daily_pipeline` :**\n"
        f" :material/calendar_today: {formatted_run}\n"
        f" :material/timer: **Durée exécution :** ~1m 45s\n"
        f" :material/folder: **Fichier généré :** `{pred_filename}`"
    )
else:
    st.sidebar.info(
        f"**Modèle actif :** {active_model_name} `{active_version}`\n\n"
        f"**Statut :** Production (Batch Daily)\n\n"
        f" :material/rocket_launch: **Dernier run `daily_pipeline` :** Aucun batch trouvé"
    )

st.sidebar.markdown("---")
if st.sidebar.button("Rafraîchir l'heure", icon=":material/refresh:"):
    st.rerun()

# Navigation Onglets
tab1, tab2, tab3 = st.tabs([
    " :material/rocket_launch: Live Predictions",
    " :material/balance: Equity Radar (Fairness)",
    " :material/analytics: Drift & Performance (Evidently)"
])

# ==================== TAB 1 : LIVE PREDICTIONS ====================
with tab1:
    if df_pred is not None:
        st.subheader(f" :material/bar_chart: Prédictions du lot quotidien (`{pred_filename}`)")

        # Métriques globales du batch
        col1, col2, col3 = st.columns(3)
        total_movies = len(df_pred)
        popular_count = int(df_pred["predicted_is_popular"].sum())
        ratio = (popular_count / total_movies) * 100 if total_movies > 0 else 0

        col1.metric("Films analysés", total_movies)
        col2.metric("Prédits 'Populaires'", popular_count)
        col3.metric("Taux de sélection", f"{ratio:.1f}%")

        st.markdown("---")

        # Distribution & Vue d'ensemble en haut
        with st.expander(" :material/show_chart: Voir la distribution des scores de ce lot", expanded=False):
            fig = px.histogram(
                df_pred,
                x="popularity_probability",
                nbins=15,
                title="Répartition des probabilités attribuées par le modèle",
                labels={"popularity_probability": "Probabilité de succès"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("###  :material/movie: Catalogue des films analysés")

        # Tri par probabilité décroissante
        df_sorted = df_pred.sort_values("popularity_probability", ascending=False).reset_index(drop=True)

        # Affichage en grille de 4 colonnes
        cols_per_row = 4

        for i in range(0, len(df_sorted), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(df_sorted):
                    movie = df_sorted.iloc[i + j]
                    with cols[j]:
                        with st.container(border=True):
                            # 1. Traitement et affichage de l'affiche TMDB
                            raw_path = str(movie.get("poster_path", "")).strip()

                            if raw_path and raw_path not in ["None", "nan"]:
                                clean_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
                                poster_url = f"https://image.tmdb.org/t/p/w500{clean_path}"
                                st.image(poster_url, use_container_width=True)
                            else:
                                st.image("https://placehold.co/300x450/161B22/E6EDF3?text=No+Poster", use_container_width=True)

                            # 2. Titre du film (transformé en lien cliquable TMDB)
                            title_str = movie.get("title", "Sans titre")
                            movie_id = movie.get("movie_id", movie.get("id", None))

                            if pd.notna(movie_id) and str(movie_id).strip() not in ["", "None", "nan"]:
                                tmdb_url = f"https://www.themoviedb.org/movie/{int(movie_id)}"
                                st.markdown(f"#### [{title_str}]({tmdb_url})")
                            else:
                                st.markdown(f"#### {title_str}")

                            # 3. Badge Prédiction Hit / Non-Hit
                            prob = movie.get("popularity_probability", 0.0)
                            is_hit = movie.get("predicted_is_popular", False)

                            if is_hit:
                                st.success(f" :material/local_fire_department: **HIT** ({prob:.0%})")
                            else:
                                st.error(f" :material/trending_down: **FLOP / NICHE** ({prob:.0%})")

                            # 4. Barre de progression de la probabilité
                            st.progress(float(prob))

                            # 5. Métadonnées détaillées
                            rel_date = movie.get("release_date", "N/A")
                            lang = str(movie.get("original_language", "N/A")).upper()
                            certif = movie.get("certification", "N/A")
                            if pd.isna(certif) or str(certif).strip() in ["", "None", "nan"]:
                                certif = "NR"

                            st.caption(f" :material/calendar_month: **Sortie :** {rel_date}")
                            st.caption(f" :material/language: **Langue :** `{lang}` |  :material/no_adult_content: **Certif :** `{certif}`")

    else:
        st.warning("Aucun fichier de prédiction quotidien trouvé dans `data/processed/`.")

# ==================== TAB 2 : EQUITY RADAR ====================
with tab2:
    st.subheader(" :material/balance: Analyse de l'Équité Algorithmique (Fairlearn)")

    if df_hist is not None:
        # Première rangée : Attributs Protégés Principaux (Genre, Langue, Studio, Thématique LGBT)
        st.markdown("###  :material/center_focus_strong: Attributs Protégés & Diversité")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        # 1. Disparate Impact : Genre du Réalisateur
        with col_f1:
            st.markdown("####  :material/movie_filter: Réalisation (Femmes/Hommes)")
            if "director_gender" in df_hist.columns:
                df_gender = df_hist[df_hist["director_gender"].isin([1, 2])].copy()
                gender_counts = (
                    df_gender.groupby("director_gender")["is_popular"]
                    .mean()
                    .reset_index()
                )
                gender_counts["Gender"] = gender_counts["director_gender"].map(
                    {1: "Réalisatrices (Femmes)", 2: "Réalisateurs (Hommes)"}
                )

                fig_gender = px.bar(
                    gender_counts,
                    x="Gender",
                    y="is_popular",
                    title="Taux de succès par genre",
                    color="Gender",
                    color_discrete_map={"Réalisatrices (Femmes)": "#8B5CF6", "Réalisateurs (Hommes)": "#3B82F6"},
                )
                st.plotly_chart(fig_gender, use_container_width=True)

                sr_female_s = gender_counts.loc[gender_counts["director_gender"] == 1, "is_popular"]
                sr_male_s = gender_counts.loc[gender_counts["director_gender"] == 2, "is_popular"]

                sr_female = sr_female_s.values[0] if not sr_female_s.empty else 0.0
                sr_male = sr_male_s.values[0] if not sr_male_s.empty else 0.0001
                di_gender = sr_female / sr_male if sr_male > 0 else 0

                if di_gender < 0.8:
                    st.error(f" :material/warning: **Alerte Biais : {di_gender:.2f}** (Sous-représentation des femmes)")
                else:
                    st.success(f" :material/check_circle: **Parité Conforme : {di_gender:.2f}**")
            else:
                st.info("Attribut `director_gender` absent.")

        # 2. Disparate Impact : Biais Thématique LGBT / Queer
        with col_f2:
            st.markdown("####  :material/diversity_3: Thématique LGBT / Gay")
            kw_col = "kw_gay_theme" if "kw_gay_theme" in df_hist.columns else ("kw_lgbt_theme" if "kw_lgbt_theme" in df_hist.columns else None)

            if kw_col:
                lgbt_counts = (
                    df_hist.groupby(kw_col)["is_popular"]
                    .mean()
                    .reset_index()
                )
                lgbt_counts["Thématique"] = lgbt_counts[kw_col].map(
                    {1: "Thème Gay/LGBT", 0: "Autres thèmes"}
                )

                fig_lgbt = px.bar(
                    lgbt_counts,
                    x="Thématique",
                    y="is_popular",
                    title="Taux de sélection selon le thème LGBT",
                    color="Thématique",
                    color_discrete_map={"Thème Gay/LGBT": "#EC4899", "Autres thèmes": "#6B7280"},
                )
                st.plotly_chart(fig_lgbt, use_container_width=True)

                sr_lgbt_s = lgbt_counts.loc[lgbt_counts[kw_col] == 1, "is_popular"]
                sr_other_s = lgbt_counts.loc[lgbt_counts[kw_col] == 0, "is_popular"]

                sr_lgbt = sr_lgbt_s.values[0] if not sr_lgbt_s.empty else 0.0
                sr_other = sr_other_s.values[0] if not sr_other_s.empty else 0.0001
                di_lgbt = sr_lgbt / sr_other if sr_other > 0 else 0

                if di_lgbt < 0.8:
                    st.error(f" :material/warning: **Alerte Biais LGBT : {di_lgbt:.2f}** (Pénalisation algorithmique)")
                else:
                    st.success(f" :material/check_circle: **Disparate Impact Conforme : {di_lgbt:.2f}**")
            else:
                st.info("Attribut `kw_gay_theme` non détecté.")

        # 3. Disparate Impact : Langues
        with col_f3:
            st.markdown("####  :material/public: Origine Linguistique")
            if "protected_is_english" in df_hist.columns:
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
                    title="Taux de sélection par langue",
                    color="Language",
                )
                st.plotly_chart(fig_lang, use_container_width=True)

                sr_en_series = lang_counts.loc[lang_counts["protected_is_english"] == 1, "is_popular"]
                sr_non_en_series = lang_counts.loc[lang_counts["protected_is_english"] == 0, "is_popular"]

                sr_en = sr_en_series.values[0] if not sr_en_series.empty else 0.0001
                sr_non_en = sr_non_en_series.values[0] if not sr_non_en_series.empty else 0.0
                di_lang = sr_non_en / sr_en if sr_en > 0 else 0

                if di_lang < 0.8:
                    st.error(f" :material/warning: **Alerte Biais : {di_lang:.2f}** (Pénalisation non-anglophone)")
                else:
                    st.success(f" :material/check_circle: **Disparate Impact Conforme : {di_lang:.2f}**")
            else:
                st.info("Attribut `protected_is_english` absent.")

        # 4. Disparate Impact : Studios
        with col_f4:
            st.markdown("####  :material/corporate_fare: Type de Studio")
            if "protected_is_major_studio" in df_hist.columns:
                major_counts = (
                    df_hist.groupby("protected_is_major_studio")["is_popular"]
                    .mean()
                    .reset_index()
                )
                major_counts["Studio"] = major_counts["protected_is_major_studio"].map(
                    {1: "Majors Hollywood", 0: "Indépendants"}
                )

                fig_major = px.bar(
                    major_counts,
                    x="Studio",
                    y="is_popular",
                    title="Taux de sélection par studio",
                    color="Studio",
                )
                st.plotly_chart(fig_major, use_container_width=True)

                sr_maj_series = major_counts.loc[major_counts["protected_is_major_studio"] == 1, "is_popular"]
                sr_ind_series = major_counts.loc[major_counts["protected_is_major_studio"] == 0, "is_popular"]

                sr_maj = sr_maj_series.values[0] if not sr_maj_series.empty else 0.0001
                sr_ind = sr_ind_series.values[0] if not sr_ind_series.empty else 0.0
                di_major = sr_ind / sr_maj if sr_maj > 0 else 0

                if di_major < 0.8:
                    st.error(f" :material/warning: **Alerte Biais : {di_major:.2f}** (Désavantage indépendants)")
                else:
                    st.success(f" :material/check_circle: **Disparate Impact Conforme : {di_major:.2f}**")
            else:
                st.info("Attribut `protected_is_major_studio` absent.")

        st.markdown("---")

        # Deuxième rangée : Attributs Complémentaires (Classification & Portée internationale)
        st.markdown("###  :material/search: Attributs Complémentaires (Classification & Distribution)")
        col_f5, col_f6 = st.columns(2)

        # 5. Certification (PG-13, R, NR...)
        with col_f5:
            st.markdown("####  :material/explicit: Certification & Classification d'Âge")
            if "certification" in df_hist.columns:
                cert_counts = (
                    df_hist.groupby("certification")["is_popular"]
                    .mean()
                    .reset_index()
                    .sort_values("is_popular", ascending=False)
                )

                fig_cert = px.bar(
                    cert_counts,
                    x="certification",
                    y="is_popular",
                    title="Taux de sélection selon la certification US",
                    labels={"certification": "Classification", "is_popular": "Taux de popularité"},
                    color="is_popular",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_cert, use_container_width=True)
                st.caption(" :material/lightbulb: **Analyse :** Évalue si le modèle favorise les classifications familiales par rapport aux contenus matures.")

        # 6. Portée Internationale (Nombre de Traductions)
        with col_f6:
            st.markdown("####  :material/g_translate: Portée Internationale (Traductions)")
            if "translation_count" in df_hist.columns:
                fig_trans = px.box(
                    df_hist,
                    x="is_popular",
                    y="translation_count",
                    title="Distribution du nombre de traductions par statut de popularité",
                    labels={"is_popular": "Est populaire (0 = Non, 1 = Oui)", "translation_count": "Nombre de traductions"},
                    color="is_popular",
                )
                st.plotly_chart(fig_trans, use_container_width=True)
                st.caption(" :material/lightbulb: **Analyse :** Mesure la pénalité subie par les œuvres à faible couverture internationale au moment de leur sortie.")

    else:
        st.warning("Données historiques indisponibles pour afficher l'Equity Radar.")

# ==================== TAB 3 : DRIFT & PERFORMANCE ====================
with tab3:
    st.subheader(" :material/trending_up: Suivi de Dérive des Données (Evidently AI)")

    col_m1, col_m2, col_m3 = st.columns(3)
    # Remplacement des valeurs fixes par les variables dynamiques de metadata.json
    col_m1.metric("Modèle Champion", f"{active_model_name} ({active_version})")
    col_m2.metric("F1-Score Actif", f"{active_f1:.4f}")
    col_m3.metric("Disparate Impact", f"{active_di:.4f}")

    st.markdown("---")

    if drift_html is not None:
        st.caption(f" :material/description: **Rapport interactif généré par Evidently AI :** `{drift_filename}`")
        components.html(drift_html, height=1000, scrolling=True)
    else:
        st.warning("Aucun rapport de dérive trouvé dans `data/monitoring/`. Exécutez `python src/monitoring.py` pour le générer.")
