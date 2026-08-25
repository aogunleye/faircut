import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Studios majeurs pour catégoriser l'attribut protégé production_companies
MAJORS_KEYWORDS = [
    "Warner Bros", "Universal Pictures", "Walt Disney", "Paramount",
    "Columbia Pictures", "20th Century Fox", "Sony Pictures", "Lionsgate", "Metro-Goldwyn-Mayer"
]


class FeatureEngineeringPipeline:
    """Pipeline de transformation des données brutes TMDB en features ML et attributs protégés."""

    def __init__(self, top_k_keywords: int = 50):
        self.top_k_keywords = top_k_keywords
        self.mlb_genres = MultiLabelBinarizer()
        self.top_keywords_list = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """S'entraîne sur le dataframe brut et renvoie le dataframe transformé."""
        logger.info(f"Début du feature engineering sur {len(df)} lignes...")
        df_out = df.copy()

        # 1. Cible (Target) : Popularité supérieure à la médiane
        median_pop = df_out["popularity"].median()
        df_out["is_popular"] = (df_out["popularity"] > median_pop).astype(int)
        logger.info(f"Seuil de popularité (médiane) : {median_pop:.2f}")

        # 2. Features Numériques & Booléennes simples
        df_out["belongs_to_collection"] = df_out["belongs_to_collection"].astype(int)

        # 3. Traitement des Genres
        genres_series = df_out["genres"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        genres_encoded = self.mlb_genres.fit_transform(genres_series)
        genre_cols = [f"genre_{g.lower().replace(' ', '_')}" for g in self.mlb_genres.classes_]
        df_genres = pd.DataFrame(genres_encoded, columns=genre_cols, index=df_out.index)

        # 4. Traitement des Keywords (Top K les plus fréquents)
        all_keywords = [
            kw for sublist in df_out["keywords"].dropna() if isinstance(sublist, (list, np.ndarray)) for kw in sublist
        ]
        kw_counts = pd.Series(all_keywords).value_counts()
        self.top_keywords_list = kw_counts.head(self.top_k_keywords).index.tolist()

        for kw in self.top_keywords_list:
            col_name = f"kw_{kw.lower().replace(' ', '_')}"
            df_out[col_name] = df_out["keywords"].apply(
                lambda x: 1 if isinstance(x, (list, np.ndarray)) and kw in x else 0
            )

        # 5. Attributs Protégés (Fairness Axes)
        
        # Axe A: original_language (Anglophone vs Non-Anglophone)
        df_out["protected_is_english"] = (df_out["original_language"] == "en").astype(int)

        # Axe B: certification (US Classification)
        def clean_certification(cert):
            if not cert or cert in ["NR", "Unrated", ""]:
                return "NR"
            cert_str = str(cert).upper()
            if cert_str in ["G", "PG", "PG-13", "R", "NC-17"]:
                return cert_str
            return "OTHER"

        df_out["protected_certification"] = df_out["certification"].apply(clean_certification)

        # Axe C: production_companies (Majors vs Studio Indépendant)
        def is_major_studio(companies):
            if not isinstance(companies, (list, np.ndarray)):
                return 0
            for company in companies:
                for major in MAJORS_KEYWORDS:
                    if major.lower() in str(company).lower():
                        return 1
            return 0

        df_out["protected_is_major_studio"] = df_out["production_companies"].apply(is_major_studio)

        # Axe D: director_gender (0: Non spécifié, 1: Femme, 2: Homme)
        df_out["protected_director_gender"] = df_out["director_gender"].fillna(0).astype(int)

        # Assemblage final
        df_final = pd.concat([df_out, df_genres], axis=1)

        # Nettoyage des colonnes brutes complexes
        columns_to_drop = ["genres", "keywords", "production_companies"]
        df_final = df_final.drop(columns=[col for col in columns_to_drop if col in df_final.columns])

        logger.info(f"Feature engineering terminé ! Forme finale : {df_final.shape}")
        return df_final


def run_feature_engineering(input_file: str = "historical_movies.parquet", output_file: str = "processed_movies.parquet"):
    """Lit le Parquet raw, applique le pipeline et enregistre le résultat dans data/processed/."""
    raw_path = os.path.join("data/raw", input_file)
    processed_dir = "data/processed"
    processed_path = os.path.join(processed_dir, output_file)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Le fichier brut {raw_path} est introuvable. Exécutez d'abord ingestion.py.")

    logger.info(f"Chargement des données depuis {raw_path}...")
    df_raw = pd.read_parquet(raw_path)

    pipeline = FeatureEngineeringPipeline(top_k_keywords=50)
    df_processed = pipeline.fit_transform(df_raw)

    os.makedirs(processed_dir, exist_ok=True)
    df_processed.to_parquet(processed_path, index=False)
    logger.info(f"Jeu de données traité enregistré dans : {processed_path}")


if __name__ == "__main__":
    run_feature_engineering()