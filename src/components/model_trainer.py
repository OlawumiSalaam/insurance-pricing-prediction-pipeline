import os
import sys
import joblib
from dataclasses import dataclass
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
#from sklearn.base import RegressorMixin

from src.logger import logging
from src.exceptions import CustomException
from src.utils import (
    evaluate_models,
    save_object,
    load_object,
    print_evaluated_results,
)

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join("artifacts/models")
    voting_model_path: str = os.path.join(model_dir, "voting_pipeline.pkl")
    stacking_model_path: str = os.path.join(model_dir, "stacking_pipeline.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_base_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting base model training and evaluation...")

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            logging.info("Base model training completed.")
            logging.info(f"Model performance summary: {model_report}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)

    def tune_model(
        self, model_name: str, pipeline: Pipeline, param_grid: Dict, X_train, y_train, n_iter=10, cv=5
    ) -> Pipeline:
        try:
            logging.info(f"[TUNING] Starting hyperparameter tuning for {model_name}...")

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            search.fit(X_train, y_train)

            logging.info(f"[TUNING] Best parameters for {model_name}: {search.best_params_}")
            logging.info(f"[TUNING] Best R2 score for {model_name}: {search.best_score_:.4f}")

            return search.best_estimator_

        except Exception as e:
            raise CustomException(e, sys)

    def build_ensembles(
        self,
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
        best_cb: Pipeline,
        best_xgb: Pipeline,
        best_ada: Pipeline
    ):
        try:
            logging.info("Building ensemble models...")

            cb_model = best_cb.named_steps['model']
            xgb_model = best_xgb.named_steps['model']
            ada_model = best_ada.named_steps['model']

            # Voting Regressor
            voting_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", VotingRegressor([
                    ("catboost", cb_model),
                    ("xgboost", xgb_model),
                    ("adaboost", ada_model)
                ]))
            ])
            voting_pipe.fit(X_train, y_train)
            print_evaluated_results(X_train, y_train, X_test, y_test, voting_pipe)
            save_object(self.config.voting_model_path, voting_pipe)
            logging.info("Voting Regressor model saved successfully.")

            # Stacking Regressor
            stacking_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", StackingRegressor(
                    estimators=[
                        ("catboost", cb_model),
                        ("xgboost", xgb_model),
                        ("adaboost", ada_model)
                    ],
                    final_estimator=LinearRegression(),
                    n_jobs=-1,
                    cv=5
                ))
            ])
            stacking_pipe.fit(X_train, y_train)
            print_evaluated_results(X_train, y_train, X_test, y_test, stacking_pipe)
            save_object(self.config.stacking_model_path, stacking_pipe)
            logging.info("Stacking Regressor model saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, preprocessor):
        try:
            logging.info("Initiating model trainer...")

            cb_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", CatBoostRegressor(verbose=False))
            ])
            xgb_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", XGBRegressor(random_state=42))
            ])
            ada_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", AdaBoostRegressor(random_state=42))
            ])

            cb_params = {
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.05],
                "model__iterations": [300, 400]
            }
            xgb_params = {
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
                "model__n_estimators": [300, 400],
                "model__colsample_bytree": [0.4, 0.6]
            }
            ada_params = {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__loss": ["linear", "square"]
            }

            best_cb = self.tune_model("CatBoost", cb_pipe, cb_params, X_train, y_train, n_iter=10)
            best_xgb = self.tune_model("XGBoost", xgb_pipe, xgb_params, X_train, y_train, n_iter=15)
            best_ada = self.tune_model("AdaBoost", ada_pipe, ada_params, X_train, y_train, n_iter=15)

            self.build_ensembles(preprocessor, X_train, y_train, X_test, y_test, best_cb, best_xgb, best_ada)

            logging.info("Model trainer process completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Run data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Run data transformation
    transformer = DataTransformation()
    X_train_arr, X_test_arr, X_train_df, X_test_df, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)


    print("Data Transformation Completed")
    print(f"Transformed X_train shape: {X_train_arr.shape}")
    print(f"Transformed X_test shape: {X_test_arr.shape}")
    print(f"y_train length: {len(y_train)}")
    print(f"Preprocessor saved at: {preprocessor_path}")

    # Load preprocessor object from file
    preprocessor = load_object(preprocessor_path)

    # Initialize and run model trainer
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(X_train_df, y_train, X_test_df, y_test, preprocessor)