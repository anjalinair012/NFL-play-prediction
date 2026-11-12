import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv")
print(df.head())
print("number of unique values in each column:")
print(df.nunique())
print("summary statistics for each column:")
print(df.describe())


#