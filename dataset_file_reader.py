import pandas as pd
import numpy as np

def csv_to_dataframe(file_name, col_names=None, header=None):
    # header=0
    df = pd.read_csv(file_name, names=col_names, header=header)
    return df

def dataframe_to_array(df):
    array = pd.DataFrame(df).to_numpy()
    return array

def array_to_dataframe(array, col_names=None):
    df = pd.DataFrame(array, columns = col_names)
    return df

file_name = "iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width",
             "scientific_name"]

# === Conversions
df = csv_to_dataframe(file_name, col_names=col_names, header=None)
print(df); print()
print(df.shape); print()
print(df.head()); print()
print(list(df.columns)); print()

array = dataframe_to_array(df)
print(array[:5]); print()

df = array_to_dataframe(array, col_names=col_names)
print(df); print()

# === Save
df.to_csv("file_name.csv", index=False, header=True) 
np.save("file_name.npy", array)

# === Load
df = csv_to_dataframe("file_name.csv", header=0) # file_name.csv has a header row.
print(df.iloc[50:60]); print()
array = np.load("file_name.npy", allow_pickle=True)
print(array[:5])
