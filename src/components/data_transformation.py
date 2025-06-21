import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_object  

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a column transformer for preprocessing numerical and categorical features.

        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            numerical_features = ["age", "bmi", "children"]
            categorical_features = ["sex", "smoker", "region"]

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing pipeline on train and test datasets.

        Args:
            train_path (str): Path to training CSV
            test_path (str): Path to testing CSV

        Returns:
            Tuple: Transformed train features, test features, target labels, and preprocessor path
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()
            logging.info("Duplicates removed from train and test datasets")

            X_train = train_df.drop(columns=["charges"])
            y_train = train_df["charges"]

            X_test = test_df.drop(columns=["charges"])
            y_test = test_df["charges"]

            preprocessor = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training data and transforming datasets")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor saved to {self.transformation_config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                X_train,  
                X_test,
                y_train,
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Run data ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Run data transformation
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

    print("✅ Data Transformation Completed")
    print(f"Transformed X_train shape: {X_train.shape}")
    print(f"Transformed X_test shape: {X_test.shape}")
    print(f"y_train length: {len(y_train)}")
    print(f"Preprocessor saved at: {preprocessor_path}")

     #Initialize and run model trainer
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(X_train, y_train, X_test, y_test, preprocessor)
