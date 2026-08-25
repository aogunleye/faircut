import argparse
import logging
import os
from datetime import datetime, timedelta
import joblib
import pandas as pd

from ingestion import TMDBIngestor
from features import FeatureEngineeringPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_daily_prediction(model_path: str = "models/xgboost_baseline.joblib", output_dir: str = "data/processed"):
    """Exécute l'inférence quotidienne sur les films sortis hier (J-1)."""
    logger.info("Début du batch de prédiction quotidien...")

    # 1. Vérification du modèle
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} est introuvable. Entraînez d'abord le modèle avec train.py.")

    model = joblib.load(model_path)
    logger.info(f"Modèle chargé depuis {model_path}")

    # 2. Ingestion des films sortis hier (J-1)
    ingestor = TMDBIngestor()
    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday_dt.strftime("%Y-%m-%d")
    
    movie_ids = ingestor.discover_movies(start_date=yesterday_str, end_date=yesterday_str, max_pages=5)
    if not movie_ids:
        logger.warning(f"Aucun nouveau film trouvé pour la journée du {yesterday_str}.")
        return

    records = []
    for m_id in movie_ids:
        raw_details = ingestor.get_movie_details(m_id)
        if raw_details:
            parsed = ingestor.parse_movie_data(raw_details)
            # Sécurité pour s'assurer de capturer poster_path
            if "poster_path" not in parsed or not parsed["poster_path"]:
                parsed["poster_path"] = raw_details.get("poster_path")
            records.append(parsed)

    if not records:
        logger.warning("Aucun détail de film n'a pu être extrait.")
        return

    df_raw = pd.DataFrame(records)

    # 3. Preprocessing via Feature Engineering
    pipeline = FeatureEngineeringPipeline(top_k_keywords=50)
    df_processed = pipeline.fit_transform(df_raw)

    # Alignement strict des colonnes avec le modèle entraîné
    feature_cols = getattr(model, "feature_names_in_", None)
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        X_predict = df_processed[feature_cols]
    else:
        X_predict = df_processed

    # 4. Inférence
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)[:, 1]

    # 5. Assemblage du rapport final de prédiction
    df_raw["predicted_is_popular"] = predictions
    df_raw["popularity_probability"] = probabilities

    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"predictions_{today_str}.parquet"
    output_path = os.path.join(output_dir, output_filename)

    df_raw.to_parquet(output_path, index=False)
    logger.info(f"Prédictions générées avec succès pour les films du {yesterday_str} ! Fichier sauvegardé : {output_path} ({len(df_raw)} films)")


if __name__ == "__main__":
    run_daily_prediction()