import glob
import logging
import os
from datetime import datetime
import pandas as pd

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_drift_monitoring(
    reference_path: str = "data/processed/processed_movies.parquet",
    predictions_path: str = None,
    output_dir: str = "data/monitoring",
):
    """Calcule le rapport de dérive entre le dataset de référence et les dernières prédictions."""
    logger.info("Début de l'analyse de dérive avec Evidently AI...")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Dataset de référence introuvable : {reference_path}")

    # Récupération du dernier fichier de prédictions si non spécifié
    if predictions_path is None:
        pred_files = sorted(glob.glob("data/processed/predictions_*.parquet"))
        if not pred_files:
            logger.warning("Aucun fichier de prédictions trouvé dans data/processed/.")
            return
        predictions_path = pred_files[-1]

    logger.info(f"Dataset de référence : {reference_path}")
    logger.info(f"Dataset courant (prédictions) : {predictions_path}")

    df_ref = pd.read_parquet(reference_path)
    df_curr = pd.read_parquet(predictions_path)

    # Liste exhaustive des features numériques, d'équité et de prédictions
    feature_cols = [
        "budget",
        "runtime",
        "vote_average",
        "vote_count",
        "translation_count",
        "belongs_to_collection",
        "kw_gay_theme",
        "kw_lgbt_theme",
        "protected_is_english",
        "protected_is_major_studio",
        "director_gender",
        "popularity_probability",
        "predicted_is_popular",
    ]

    # Conservation des colonnes réellement existantes dans les deux datasets
    selected_cols = [c for c in feature_cols if c in df_ref.columns and c in df_curr.columns]

    # Cast explicite des données en float pour uniformiser
    df_ref_clean = df_ref[selected_cols].astype(float)
    df_curr_clean = df_curr[selected_cols].astype(float)

    # Génération du rapport via l'API legacy
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=df_ref_clean,
        current_data=df_curr_clean,
    )

    # Export du rapport HTML
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    html_output_path = os.path.join(output_dir, f"drift_report_{today_str}.html")
    report.save_html(html_output_path)

    logger.info(f"Rapport de dérive généré avec succès ! Disponible sous : {html_output_path}")


if __name__ == "__main__":
    run_drift_monitoring()