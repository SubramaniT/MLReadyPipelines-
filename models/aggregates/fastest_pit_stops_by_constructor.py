import numpy as numpy
import pandas as pandas

def model(dbt, session):
    dbt.config(packages=["pandas","numpy"])

    pit_stops_joined = dbt.ref("pit_stops_joined").to_pandas()
    year=2021

    pit_stops_joined["PIT_STOP_SECONDS"] = pit_stops_joined["PIT_STOP_MILLISECONDS"]/1000
    fastest_pit_stops = pit_stops_joined[(pit_stops_joined["RACE_YEAR"]==year)].groupby(by="CONSTRUCTOR_NAME")["PIT_STOP_SECONDS"].describe().sort_values(by='mean')
    fastest_pit_stops.reset_index(inplace=True)
    fastest_pit_stops.columns = fastest_pit_stops.columns.str.upper()
    
    return fastest_pit_stops.round(2)