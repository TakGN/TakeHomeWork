import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import logging
import parameters
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        logger.info('starting the prediction')

    @staticmethod
    def load_dataset(path, name):
        file_path = f"{path}/{name}"
        return pd.read_csv(file_path)

    @staticmethod
    def define_model(**model_params):
        return GradientBoostingClassifier(**model_params)

    @staticmethod
    def define_preprocessing(feature_name, tf_idf_params):
        tf_idf_transformer = Pipeline(
            [
                ("tf_idf_vectorizer", TfidfVectorizer(**tf_idf_params)),
            ]
        )
        return ColumnTransformer([("email_transformer", tf_idf_transformer, feature_name)])

    @staticmethod
    def define_pipeline(preprocess_pipeline, model):
        return Pipeline([("preprocessing", preprocess_pipeline), ("model", model)])

    @staticmethod
    def save_fitted_pipeline(pipeline, path, name):
        file_path = f"{path}/{name}"
        joblib.dump(pipeline, file_path)

    @staticmethod
    def load_pipeline(path, name):
        file_path = f"{path}/{name}"
        return joblib.load(file_path)

    @staticmethod
    def train_pipeline(train_df, params, email_col, target):
        model = Model.define_model(**params["model_params"])
        preprocessing = Model.define_preprocessing(email_col, params["tf_idf_params"])
        pipeline = Model.define_pipeline(preprocessing, model)
        pipeline.fit(train_df[[email_col]], train_df[target])
        return pipeline

    @staticmethod
    def get_results(fitted_pipeline, test_df, target_df):
        predictions = fitted_pipeline.predict(test_df)
        score = accuracy_score(predictions, target_df)
        return predictions, score

    @staticmethod
    def get_model(params, pipeline_name):
        df = Model.load_dataset(parameters.data_path, parameters.df_name)
        train_df, test_df = train_test_split(df, test_size=0.2)
        fitted_pipeline = Model.train_pipeline(train_df, params,
                                               parameters.email_col, parameters.target)
        Model.save_fitted_pipeline(fitted_pipeline, parameters.model_path,
                                   pipeline_name)
        results = Model.get_results(fitted_pipeline,
                                    test_df[[parameters.email_col]],
                                    test_df[parameters.target])
        return results



