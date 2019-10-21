import pandas as pd
import numpy as np
import acquire

import env

def clean_data(df):
    df = df.dropna()
    return df

#def wrangle_telco():
   # return clean_data(get_data_from_mysql())