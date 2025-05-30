import logging
import joblib
import pandas as pd
import os 
from snowflake.snowpark import types as T 

DB_STAGE='FORMULASTAGE'
version ='1.0'

model_file_path = 'driver_position_' + version
model_file_packaged = 'driver_position_' + version + '.joblib'

LOCAL_TEMP_DIR = f'tmp/driver_position'
DOWNLOAD_DIR = os.path.join(LOCAL_TEMP_DIR,'download')
TARGET_MODEL_DIR_PATH = os.path.join(LOCAL_TEMP_DIR,'ml_mode')
TARGET_LIB_PATH = os.path.join(LOCAL_TEMP_DIR,'lib')

FEATURE_COLS = [
        "RACE_YEAR"
        ,"CIRCUIT_NAME"
        ,"GRID"
        ,"CONSTRUCTOR_NAME"
        ,"DRIVER"
        ,"DRIVERS_AGE_YEARS"
        ,"DRIVER_CONFIDENCE"
        ,"CONSTRUCTOR_RELIABILITY"
        ,"TOTAL_PIT_STOPS_PER_RACE"]

def register_udf_for_prediction(p_predictor, p_session, p_dbt):
    def predict_position(p_df: T.PandasDataFrame[int, int,int,int,int , int,int,int,int]) -> T.PandasSeries[int]:

        p_df.columns = [*FEATURE_COLS]

        pred_array = p_predictor.predict(p_df)

        df_predicted = pd.Series(pred_array)
        return df_predicted 

    udf_packages = p_dbt.config.get('packages')

    predict_position_udf = p_session.udf.register(
        predict_position,
        name = f'predict_position',
        packages = udf_packages 
    )

    return predict_position_udf 

def download_models_and_libs_from_stage(p_session):
    p_session.file.get(f'@{DB_STAGE}/{model_file_path}/{model_file_packaged}',DOWNLOAD_DIR)

def load_model(p_session):
    model_fl_path = os.path.join(DOWNLOAD_DIR, model_file_packaged)
    predictor = joblib.load(model_fl_path)
    return predictor 

def model(dbt,session):
    dbt.config(
        materialized="table",
        tags = "predict", 
        packages=['snowflake-snowpark-python','scipy','scikit-learn','pandas','numpy']
    )

    session._use_scoped_temp_objects = False 
    download_models_and_libs_from_stage(session)
    predictor = load_model(session)
    predict_position_udf = register_udf_for_prediction(predictor, session, dbt)

    hold_out_df = (dbt.ref("hold_out_dataset_for_prediction").select(*FEATURE_COLS))

    new_predictions_df = hold_out_df.withColumn("position_predicted", predict_position_udf(*FEATURE_COLS))

    return new_predictions_df
