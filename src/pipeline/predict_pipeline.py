import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Handles loading of preprocessor and model, and performs predictions.
    """
    def __init__(self):
        #self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        self.model_path = os.path.join('artifacts', 'models', 'stacking_pipeline.pkl')

    def predict(self, features):
        """
        Transforms input features and returns the prediction result.
        Args:
            features (pd.DataFrame): Input features as a single-row DataFrame.
        Returns:
            float: Predicted insurance charge.
        """
        try:
            logging.info("Loading model pipeline")
            model = load_object(file_path=self.model_path)  
            logging.info("Making prediction using full pipeline") # has the prepocessor in model pipeline
            pred = model.predict(features)  
            return pred[0] # Since we are predicting for a single instance, return the first element

        except Exception as e:
            logging.error(f"Exception occurred in prediction pipeline: {e}")
            raise CustomException(e, sys)

        

       
                

class CustomData:
    """
    Collects and validates user input, and converts it into a DataFrame suitable for model prediction.
    """

    def __init__(self, age:int, sex:str, bmi:float, children:int, smoker:str, region:str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def validate_input(self):

        """
        Validates the input data.
        Raises:
        ValueError: If any input is invalid.
        """

        # Validate sex
        if self.sex not in ['male', 'female']:
            raise ValueError(f"Invalid value for sex: {self.sex}. Must be 'male' or 'female'.")

        # Validate smoker
        if self.smoker not in ['yes', 'no']:
            raise ValueError(f"Invalid value for smoker: {self.smoker}. Must be 'yes' or 'no'.")

        # Validate region
        valid_regions = ['southwest', 'southeast', 'northwest', 'northeast']
        if self.region not in valid_regions:
            raise ValueError(f"Invalid region: {self.region}. Must be one of {valid_regions}.")

        # Validate age (positive)
        if self.age <= 0:
            raise ValueError(f"Age must be positive, got {self.age}.")

        # Validate bmi (positive)
        if self.bmi <= 0:
            raise ValueError(f"BMI must be positive, got {self.bmi}.")

        # Validate children (non-negative integer)
        if self.children < 0:
            raise ValueError(f"Children must be non-negative integer, got {self.children}.")
                
                

    def get_data_as_dataframe(self):
        """
        Converts the input data into a DataFrame and validates it.
        Returns:
        pd.DataFrame: A DataFrame containing the input data.
        Raises:
        CustomException: If validation fails or an error occurs.
        """
        try:
            # Validate input
            self.validate_input()
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created from input data')
            return df

        except ValueError as ve:
            logging.error(f"Input validation error: {ve}")
            raise CustomException(ve, sys)

        except Exception as e:
            logging.error(f"Exception occurred in creating dataframe: {e}")
            raise CustomException(e, sys)
            