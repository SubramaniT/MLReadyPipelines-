import pandas as pandas
def model(dbt, session):
    dbt.config(packages=["pandas"])

    fct_results = dbt.ref("fct_results").to_pandas()

    start_year = 2010
    end_year = 2020

    data = fct_results.loc[fct_results['RACE_YEAR'].between(start_year,end_year)]

    data['POSITION'] = data['POSITION'].astype(float)

    data['TOTAL_PITS_STOPS_PER_RACE'] = data['TOTAL_PIT_STOPS_PER_RACE'].fillna(0)

    mapping = {'Force India': 'Racing Point', 'Sauber': 'Alfa Romeo', 'Lotus F1': 'Renault', 'Toro Rosso': 'AlphaTauri'}
    data['CONSTRUCTOR_NAME'].replace(mapping,inplace=True)

    dnf_by_driver = data.groupby('DRIVER').sum(numeric_only=True)['DNF_FLAG']
    driver_race_entered = data.groupby('DRIVER').count()['DNF_FLAG']
    driver_dnf_ratio = (dnf_by_driver) / (driver_race_entered)
    driver_confidence = 1 - driver_dnf_ratio
    driver_confidence_dict = dict(zip(driver_confidence.index,driver_confidence))

    dnf_by_constructor = data.groupby('CONSTRUCTOR_NAME').sum(numeric_only=True)['DNF_FLAG']
    constructor_race_entered = data.groupby('CONSTRUCTOR_NAME').count()['DNF_FLAG']
    constructor_dnf_ratio = (dnf_by_constructor /constructor_race_entered)
    constructor_reliability = 1- constructor_dnf_ratio
    constructor_reliability_dict = dict(zip(constructor_reliability.index,constructor_reliability))

    data['DRIVER_CONFIDENCE'] = data['DRIVER'].apply(lambda x:driver_confidence_dict[x])
    data['CONSTRUCTOR_RELIABILITY'] = data['CONSTRUCTOR_NAME'].apply(lambda x:constructor_reliability_dict[x])

    active_constructors = ['Renault', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                        'AlphaTauri', 'Racing Point', 'Alfa Romeo', 'Red Bull',
                        'Haas F1 Team']
    active_drivers = ['Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                    'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                    'Lando Norris', 'Sebastian Vettel', 'Kimi Räikkönen',
                    'Charles Leclerc', 'Lewis Hamilton', 'Daniil Kvyat',
                    'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                    'Sergio Pérez', 'Esteban Ocon', 'Antonio Giovinazzi',
                    'Romain Grosjean','Nicholas Latifi']
    
    data['ACTIVE_DRIVER'] = data['DRIVER'].apply(lambda x: int(x in active_drivers))
    data['ACTIVE_CONSTRUCTOR'] = data['CONSTRUCTOR_NAME'].apply(lambda x: int(x in active_constructors))

    return data