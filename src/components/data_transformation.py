import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features =['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            # Create numerical and categorical feature pipelines
            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler(with_mean=False))])
            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('one_hot_encoder', OneHotEncoder()),('scaler', StandardScaler(with_mean=False))])
            logging.info("Numerical and categorical feature pipelines created successfully.")
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")
            
            # Combine both pipelines into a ColumnTransformer
            logging.info("Combining numerical and categorical pipelines into a ColumnTransformer.")
            preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numerical_features),('cat', cat_pipeline, categorical_features)])

            # Log the preprocessor object
            logging.info("ColumnTransformer created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")
            
            preprocessor_obj = self.get_data_transformer_object()
            target_column = 'math_score'
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]
            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column]

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test)]
            logging.info("Data transformation completed successfully.")

            save_object (
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info(f"Saving preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


