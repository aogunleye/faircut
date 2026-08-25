import json
import logging
import os
import warnings
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """Classe de gestion de l'entraînement XGBoost, de l'optimisation Optuna, de l'audit Fairlearn et de la promotion V2."""

    def __init__(self, data_path: str = "data/processed/processed_movies.parquet"):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.protected_df = None
        self.best_params = None
        self.best_model = None

    def load_and_prepare_data(self):
        """Charge le dataset traité et sépare les features, la cible et les attributs protégés."""
        logger.info(f"Chargement des données depuis {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)

        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df = self.df.sort_values("release_date").reset_index(drop=True)

        self.y = self.df["is_popular"]

        protected_cols = [col for col in self.df.columns if col.startswith("protected_")]
        self.protected_df = self.df[protected_cols]

        non_feature_cols = [
            "movie_id", "title", "release_date", "popularity", "vote_average", "is_popular",
            "original_language", "certification", "director_gender"
        ] + protected_cols

        feature_cols = [col for col in self.df.columns if col not in non_feature_cols]
        self.X = self.df[feature_cols]

        logger.info(f"Données préparées : {self.X.shape[0]} lignes, {self.X.shape[1]} features.")

    def optimize_hyperparameters(self, n_trials: int = 20) -> dict:
        """Optimise les hyperparamètres XGBoost via Optuna."""
        logger.info(f"Lancement de l'optimisation Optuna ({n_trials} trials)...")
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }

            f1_scores = []
            for train_idx, val_idx in tscv.split(self.X):
                X_tr, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_tr, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                f1_scores.append(f1_score(y_val, preds))

            return np.mean(f1_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        logger.info(f"Meilleurs hyperparamètres trouvés (F1-score moyen : {study.best_value:.4f})")
        return self.best_params

    def train_and_evaluate(self):
        """Entraîne le modèle Challenger (V2), compare au Champion (V1) et promeut si validé."""
        mlflow.set_experiment("faircut_xgboost_training")

        with mlflow.start_run():
            logger.info("Entraînement du modèle Challenger (V2)...")
            
            split_idx = int(len(self.X) * 0.8)
            X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
            y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]
            protected_test = self.protected_df.iloc[split_idx:]

            self.best_model = XGBClassifier(**self.best_params, random_state=42, n_jobs=-1)
            self.best_model.fit(X_train, y_train)

            y_pred = self.best_model.predict(X_test)
            y_proba = self.best_model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            v2_f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Audit Fairness
            mf_lang = MetricFrame(
                metrics=selection_rate,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=protected_test["protected_is_english"]
            )
            sr_non_english = mf_lang.by_group.get(0, 0.0001)
            sr_english = mf_lang.by_group.get(1, 0.0001)
            v2_di = sr_non_english / sr_english if sr_english > 0 else 0

            # Récupération des performances Champion (V1) depuis metadata ou baseline
            v1_f1 = 0.5023
            v1_di = 0.6500
            metadata_path = "models/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    v1_f1 = meta.get("f1_score", v1_f1)
                    v1_di = meta.get("disparate_impact", v1_di)

            logger.info(f"V1 Champion   -> F1: {v1_f1:.4f} | Disparate Impact: {v1_di:.4f}")
            logger.info(f"V2 Challenger -> F1: {v2_f1:.4f} | Disparate Impact: {v2_di:.4f}")

            # RÈGLES DE PASSAGE EN PRODUCTION (CHAMPION vs CHALLENGER)
            cond_f1 = v2_f1 >= (v1_f1 - 0.02)
            cond_di = v2_di > v1_di

            if cond_f1 and cond_di:
                logger.info("✅ CHALLENGER VALIDÉ : Promotion du modèle en Production !")
                
                # 1. Sauvegarde du modèle sous la baseline active
                os.makedirs("models", exist_ok=True)
                joblib.dump(self.best_model, "models/xgboost_baseline.joblib")

                # 2. Calcul dynamique du numéro de version (v1.0 -> v2.0 -> v3.0...)
                metadata_path = "models/metadata.json"
                current_version = 1
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            old_meta = json.load(f)
                            old_v_str = old_meta.get("model_version", "v1.0")
                            # Extraction du numéro majeur (ex: "v2.0" -> 2)
                            current_version = int(old_v_str.lower().replace("v", "").split(".")[0])
                    except Exception as e:
                        logger.warning(f"Impossible de lire l'ancienne version, réinitialisation à v1 ({e})")

                next_version_str = f"v{current_version + 1}.0"

                # 3. Mise à jour du registre local metadata.json
                metadata = {
                    "model_version": next_version_str,
                    "model_name": f"XGBoost Mitigated ({next_version_str})",
                    "status": "Production",
                    "f1_score": float(v2_f1),
                    "disparate_impact": float(v2_di),
                    "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)

                logger.info(f"🚀 Nouveau modèle promu avec succès : {next_version_str} !")

                # 4. Logs MLflow
                mlflow.log_params(self.best_params)
                mlflow.log_metric("f1_score", v2_f1)
                mlflow.log_metric("disparate_impact", v2_di)
                mlflow.xgboost.log_model(self.best_model, artifact_path="model")
            else:
                logger.warning("❌ CHALLENGER REJETÉ : Les critères de performance / équité ne sont pas remplis.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_and_prepare_data()
    trainer.optimize_hyperparameters(n_trials=20)
    trainer.train_and_evaluate()