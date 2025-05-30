import pandas as pd 

def model(dbt, session):
    dbt.config(packages=['pandas'],tags="predict")
    encoding = dbt.ref("covariate_encoding").to_pandas()

    year = 2020
    hold_out_dataset = encoding.loc[encoding['RACE_YEAR'] == year]

    return hold_out_dataset
    