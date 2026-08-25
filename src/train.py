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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Désactiver les logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """Classe de gestion de l'entraînement XGBoost, de l'optimisation Optuna et de l'audit Fairlearn."""

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

        # Tri chronologique obligatoire pour TimeSeriesSplit
        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df = self.df.sort_values("release_date").reset_index(drop=True)

        # Isolement de la cible
        self.y = self.df["is_popular"]

        # Isolement des attributs protégés pour Fairlearn
        protected_cols = [col for col in self.df.columns if col.startswith("protected_")]
        self.protected_df = self.df[protected_cols]

        # Exclusion des colonnes non prédictives ou protégées
        non_feature_cols = [
            "movie_id", "title", "release_date", "popularity", "vote_average", "is_popular",
            "original_language", "certification", "director_gender"
        ] + protected_cols

        feature_cols = [col for col in self.df.columns if col not in non_feature_cols]
        self.X = self.df[feature_cols]

        logger.info(f"Données préparées : {self.X.shape[0]} lignes, {self.X.shape[1]} features de prédiction.")

    def optimize_hyperparameters(self, n_trials: int = 30) -> dict:
        """Optimise les hyperparamètres XGBoost via Optuna et TimeSeriesSplit."""
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
        """Entraîne le modèle final, logue les métriques dans MLflow et réalise l'audit Fairlearn."""
        mlflow.set_experiment("faircut_xgboost_training")

        with mlflow.start_run():
            logger.info("Entraînement du modèle XGBoost final...")
            
            # Split train/test temporel (80% train / 20% test chronologique)
            split_idx = int(len(self.X) * 0.8)
            X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
            y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]
            protected_test = self.protected_df.iloc[split_idx:]

            self.best_model = XGBClassifier(**self.best_params, random_state=42, n_jobs=-1)
            self.best_model.fit(X_train, y_train)

            # Prédictions
            y_pred = self.best_model.predict(X_test)
            y_proba = self.best_model.predict_proba(X_test)[:, 1]

            # 1. Métriques ML standard
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            logger.info(f"Résultats Test -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            # Tracking MLflow - Hyperparamètres & Métriques ML
            mlflow.log_params(self.best_params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # 2. Audit de Fairness (Fairlearn)
            logger.info("Exécution de l'audit de fairness avec Fairlearn...")

            # Axe A : Langue Anglophone vs Non-Anglophone
            mf_lang = MetricFrame(
                metrics=selection_rate,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=protected_test["protected_is_english"]
            )
            
            # Calcul du Disparate Impact (Selection Rate Non-Anglophone / Selection Rate Anglophone)
            sr_non_english = mf_lang.by_group.get(0, 0.0001)
            sr_english = mf_lang.by_group.get(1, 0.0001)
            disparate_impact_lang = sr_non_english / sr_english if sr_english > 0 else 0

            # Axe B : Majors vs Studios Indépendants
            mf_major = MetricFrame(
                metrics=selection_rate,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=protected_test["protected_is_major_studio"]
            )
            sr_indie = mf_major.by_group.get(0, 0.0001)
            sr_major = mf_major.by_group.get(1, 0.0001)
            disparate_impact_major = sr_indie / sr_major if sr_major > 0 else 0

            logger.info(f"Disparate Impact (Langue Non-EN / EN) : {disparate_impact_lang:.4f}")
            logger.info(f"Disparate Impact (Studios Indés / Majors) : {disparate_impact_major:.4f}")

            # Tracking MLflow - Métriques d'Équité
            mlflow.log_metric("fairness_disparate_impact_language", disparate_impact_lang)
            mlflow.log_metric("fairness_disparate_impact_majors", disparate_impact_major)

            # Enregistrement du modèle dans MLflow
            mlflow.xgboost.log_model(self.best_model, artifact_path="model")
            
            # Sauvegarde locale du modèle et du préprocesseur
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.best_model, "models/xgboost_baseline.joblib")
            logger.info("Modèle enregistré localement dans models/xgboost_baseline.joblib")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_and_prepare_data()
    trainer.optimize_hyperparameters(n_trials=20)
    trainer.train_and_evaluate()