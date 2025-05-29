import pandas as pandas

def model(dbt, session):
    dbt.config(package=['pandas'], tags='train')
    encoding = dbt.ref("covariate_encoding").to_pandas()

    start_year=2010
    end_year=2020

    train_test_dataset = encoding.loc[encoding['RACE_YEAR'].between(start_year,end_year)]
    return train_test_dataset