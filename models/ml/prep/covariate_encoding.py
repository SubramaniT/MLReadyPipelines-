import pandas as pandas
import numpy as numpy
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def model(dbt,session):
    dbt.config(packages = ["pandas","numpy","scikit-learn"])

    data = dbt.ref("ml_data_prep").to_pandas()

    covariates = data[['RACE_YEAR','CIRCUIT_NAME','GRID','CONSTRUCTOR_NAME','DRIVER','DRIVERS_AGE_YEARS','DRIVER_CONFIDENCE','CONSTRUCTOR_RELIABILITY','TOTAL_PIT_STOPS_PER_RACE','ACTIVE_DRIVER','ACTIVE_CONSTRUCTOR', 'POSITION']]

    fil_cov = covariates[(covariates['ACTIVE_DRIVER']==1) & (covariates['ACTIVE_CONSTRUCTOR']==1)]

    le = LabelEncoder()
    fil_cov['CIRCUIT_NAME'] = le.fit_transform(fil_cov['CIRCUIT_NAME'])
    fil_cov['CONSTRUCTOR_NAME'] = le.fit_transform(fil_cov['CONSTRUCTOR_NAME'])
    fil_cov['DRIVER'] = le.fit_transform(fil_cov['DRIVER'])
    fil_cov['TOTAL_PIT_STOPS_PER_RACE'] = le.fit_transform(fil_cov['TOTAL_PIT_STOPS_PER_RACE'])

    def position_index(x):
        if x<4:
            return 1
        if x>10:
            return 3
        else :
            return 2

    encoded_data = fil_cov.drop(['ACTIVE_DRIVER','ACTIVE_CONSTRUCTOR'],axis=1)
    encoded_data['POSITION_LABEL'] = encoded_data['POSITION'].apply(lambda x: position_index(x))
    encoded_data_grouped_target = encoded_data.drop(['POSITION'],axis=1)

    return encoded_data_grouped_target 
