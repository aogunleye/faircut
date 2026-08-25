import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement (.env)
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
DATA_RAW_DIR = "data/raw"


class TMDBIngestor:
    """Ingestor pour récupérer les données de films depuis l'API TMDB v3."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or TMDB_API_KEY
        if not self.api_key:
            raise ValueError("La clé d'API TMDB_API_KEY n'est pas définie dans les variables d'environnement.")

        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _fetch(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Effectue une requête GET avec gestion automatique des retries (backoff exponentiel)."""
        url = f"{TMDB_BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=10)
        
        # Gestion du Rate Limiting TMDB (HTTP 429)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2))
            logger.warning(f"Rate-limit atteint (429). Pause de {retry_after}s...")
            raise requests.exceptions.RequestException("Rate limited")
            
        response.raise_for_status()
        return response.json()

    def discover_movies(self, start_date: str, end_date: str, max_pages: int = 500) -> List[int]:
        """Récupère les identifiants des films sortis entre deux dates via /discover/movie."""
        movie_ids = []
        page = 1

        logger.info(f"Début du scanning /discover/movie du {start_date} au {end_date}...")

        while page <= max_pages:
            params = {
                "primary_release_date.gte": start_date,
                "primary_release_date.lte": end_date,
                "sort_by": "popularity.desc",
                "page": page,
            }
            try:
                data = self._fetch("discover/movie", params=params)
                results = data.get("results", [])
                if not results:
                    break

                ids = [movie["id"] for movie in results]
                movie_ids.extend(ids)

                total_pages = min(data.get("total_pages", 1), max_pages)
                if page % 10 == 0 or page == total_pages:
                    logger.info(f"Progression : Page {page}/{total_pages} ({len(movie_ids)} films trouvés)")

                if page >= total_pages:
                    break

                page += 1
            except Exception as e:
                logger.error(f"Erreur lors du scan de la page {page}: {e}")
                break

        return movie_ids

    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Récupère tous les détails d'un film avec append_to_response + paramètre de langue d'image."""
        params = {
            "append_to_response": "credits,release_dates,keywords,translations,videos,images",
            "include_image_language": "en,null",  # <--- CORRECTION DOC TMDB : Récupère les posters même sans filtre de langue restrictif
        }
        try:
            return self._fetch(f"movie/{movie_id}", params=params)
        except Exception as e:
            logger.error(f"Impossible de récupérer le film ID {movie_id}: {e}")
            return None

    @staticmethod
    def parse_movie_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait et aplatit les variables cibles (XGBoost + Fairlearn + Metadonnées Dashboard)."""
        # 1. Parsing des crédits (réalisateur/directrice + genre)
        crew = data.get("credits", {}).get("crew", [])
        directors = [member for member in crew if member.get("job") == "Director"]
        director_gender = directors[0].get("gender") if directors else 0  # 0: non spécifié, 1: Femme, 2: Homme

        # 2. Certification (US classification d'âge)
        certification = "NR"  # Not Rated par défaut
        releases = data.get("release_dates", {}).get("results", [])
        for rel in releases:
            if rel.get("iso_3166_1") == "US":
                for date_info in rel.get("release_dates", []):
                    if date_info.get("certification"):
                        certification = date_info.get("certification")
                        break
                break

        # 3. Compteurs & Listes (Images, Traductions, Vidéos)
        images = data.get("images", {})
        posters = images.get("posters", []) if isinstance(images, dict) else []
        backdrops = images.get("backdrops", []) if isinstance(images, dict) else []

        translations = data.get("translations", {}).get("translations", [])
        videos = data.get("videos", {}).get("results", [])

        genres = [g.get("name") for g in data.get("genres", [])]
        keywords = [k.get("name") for k in data.get("keywords", {}).get("keywords", [])]
        production_companies = [c.get("name") for c in data.get("production_companies", [])]

        return {
            "movie_id": data.get("id"),
            "title": data.get("title"),
            "release_date": data.get("release_date"),
            "poster_path": data.get("poster_path"),      # <--- AJOUT : chemin relatif de l'affiche
            "backdrop_path": data.get("backdrop_path"),  # <--- AJOUT : chemin relatif du fond d'écran
            # Features numériques
            "budget": data.get("budget", 0),
            "runtime": data.get("runtime", 0),
            "vote_count": data.get("vote_count", 0),
            "vote_average": data.get("vote_average", 0.0),
            "popularity": data.get("popularity", 0.0),
            "translation_count": len(translations),
            "video_count": len(videos),
            "poster_count": len(posters),
            "backdrop_count": len(backdrops),
            # Features catégorielles & booléennes
            "genres": genres,
            "keywords": keywords,
            "belongs_to_collection": data.get("belongs_to_collection") is not None,
            # Attributs Protégés (Fairness)
            "original_language": data.get("original_language"),
            "certification": certification,
            "production_companies": production_companies,
            "director_gender": director_gender,
        }

    def run_pipeline(self, start_date: str, end_date: str, output_filename: str) -> None:
        """Exécute le pipeline d'ingestion et écrit le fichier Parquet final."""
        movie_ids = self.discover_movies(start_date, end_date)
        logger.info(f"Début de l'extraction détaillée pour {len(movie_ids)} films...")

        records = []
        for idx, movie_id in enumerate(movie_ids, start=1):
            raw_data = self.get_movie_details(movie_id)
            if raw_data:
                parsed_data = self.parse_movie_data(raw_data)
                records.append(parsed_data)

            if idx % 500 == 0 or idx == len(movie_ids):
                logger.info(f"Extraction détaillée : {idx}/{len(movie_ids)} traités.")

        df = pd.DataFrame(records)
        
        # Sécurisation du dossier cible
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        output_path = os.path.join(DATA_RAW_DIR, output_filename)

        # Sauvegarde Parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Ingestion terminée avec succès ! File sauvegardé dans : {output_path} ({len(df)} lignes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'ingestion de données TMDB pour FairCut.")
    parser.add_argument("--historical", action="store_true", help="Extrait l'historique complet des 10 dernières années.")
    parser.add_argument("--start-date", type=str, help="Date de début (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Date de fin (YYYY-MM-DD).")
    args = parser.parse_args()

    ingestor = TMDBIngestor()

    if args.historical:
        # Période de 10 ans glissants
        end = datetime.now()
        start = end - timedelta(days=365 * 10)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        filename = "historical_movies.parquet"
        logger.info(f"Mode Historique activé : Extraction de {start_str} à {end_str}")
    else:
        # Période par défaut pour la routine quotidienne : 2 derniers jours (hier + aujourd'hui)
        end_str = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start_str = args.start_date or (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        filename = f"daily_movies_{end_str}.parquet"
        logger.info(f"Mode Quotidien activé : Extraction de {start_str} à {end_str}")

    ingestor.run_pipeline(start_date=start_str, end_date=end_str, output_filename=filename)