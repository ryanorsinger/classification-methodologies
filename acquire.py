import env
import pandas as pd

def get_connection(db, user=env.user, host=env.host, password=env.password):
    """Template for generating a connection. Import env and then provide the db name as the argument to this function"""
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    """ Return raw Titanic dataset as a pandas dataframe """
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

# def get_iris_data():
#     """"" Return raw Titanic dataset as a pandas dataframe """
