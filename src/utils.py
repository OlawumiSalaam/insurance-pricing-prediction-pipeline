import os
import sys
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exceptions import CustomException
from src.logger import logging

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: str):
    """
    Load and return a Python object from a file using joblib.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return joblib.load(file_path)
    
    except Exception as e:
        logging.info(f"Exception occurred while loading object from {file_path}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)
            report[name] = test_score
        return report
    except Exception as e:
        logging.error("Exception occurred during model evaluation.")
        raise CustomException(e, sys)

def model_metrics(true, predicted):
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, predicted)
        return mae, rmse, r2
    except Exception as e:
        logging.error("Error calculating model metrics.")
        raise CustomException(e, sys)

def print_evaluated_results(X_train, y_train, X_test, y_test, pipeline):
    try:
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        metrics = {
            "Train MAE": mean_absolute_error(y_train, y_pred_train),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "Train R2": r2_score(y_train, y_pred_train),
            "Test MAE": mean_absolute_error(y_test, y_pred_test),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "Test R2": r2_score(y_test, y_pred_test)
        }

        logging.info("Model Evaluation Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")
    except Exception as e:
        logging.error("Error printing model performance.")
        raise CustomException(e, sys)