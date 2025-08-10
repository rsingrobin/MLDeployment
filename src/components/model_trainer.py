import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from src.components.data_transformation import DataTransformationConfig


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training initiated.")
            logging.info("Splitting train and test arrays into features and target variables.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Models to be trained
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }

            #Hyperparameter tuning for models
            logging.info("Starting model training with hyperparameter tuning.")
            params = {
                'Decision Tree' : {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error','poisson']
                },
                'Random Forest': {
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.01, 0.1, 0.5,0.001],
                    'subsample': [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression': {},
                'XGBoost': {
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'CatBoost': {
                    'learning_rate': [0.01, 0.1, 0.5, 0.001],
                    'depth': [4, 6, 8, 10],
                    'iterations': [100, 200, 300]
                },
                'AdaBoost': {
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.01, 0.1, 0.5, 0.001]
                }   
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.")
            logging.info("Saving the best model.")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info("Model training completed successfully.")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model on test data: {r2_square}")
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
