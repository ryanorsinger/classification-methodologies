# Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from acquire import get_titanic_data

df = get_titanic_data()

def prepare_titanic_data(df):
    """
        Encodings for "Embarked" column
        2 == "S" == Southampton == 644 people
        0 == "C" == Cherbourg == 168 people
        1 == "Q" == Queenstown == 77 people
        3 == "Unknown" == 2 people

        177 records missing age values set to the average age
    
        Missing embark_towns are set to "Other"

        Encodings for "Class"
        First class == 0
        Second class == 1
        Third class == 2

        Encodings for "Sex"
        1 == male
        0 == female
    """

    df.embark_town.fillna('Other', inplace=True)

    # Drop deck and embarked_town
    df.drop(columns=['deck', 'embark_town'], inplace=True)

    # Encoding: Objects (Categorical Variables) to Numeric
    # Use sklearn's LabelEncoder
    encoder = LabelEncoder()

    # Set Unknown and encode Embarked column to numbers
    # 2 == "S" == Southampton == 644 people
    # 0 == "C" == Cherbourg == 168 people
    # 1 == "Q" == Queenstown == 77 people
    # 3 == "Unknown" == 2 people
    df.embarked.fillna('Unknown', inplace=True)
    encoder.fit(df.embarked)
    df.embarked = encoder.transform(df.embarked)

    # Encode the Class (first class, second, etc...)
    # First class == 0
    # Second class == 1
    # Third class == 2
    encoder.fit(df["class"])
    df["class"] = encoder.transform(df["class"])

    # Encode gender
    # male == 1 == 577 records
    # female == 0 == 314 records
    encoder.fit(df.sex)
    df.sex = encoder.transform(df.sex)

    # Handle the 177 records with missing age values
    average_age = df.age.mean()
    df.age.fillna(average_age, inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(df[['fare']])
    df.fare = scaler.transform(df[['fare']])

    scaler = MinMaxScaler()
    scaler.fit(df[['age']])
    df.age = scaler.transform(df[['age']])

    # Set the index to the passenger id
    df = df.set_index("passenger_id")
    return df
