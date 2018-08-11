import numpy as np
import pandas as pd


'''
script to flatten the columns of a .csv file into a single column.
Crossvalidation outputs were saved by month, so this script is useful 
if you want to flatten all the 70 years of crossvalidated results into 
a single column.
'''

path = 'input_file.csv'
df = pd.read_csv(path)
array = pd.DataFrame.as_matrix(df)
flat_array = df.values.flatten()
np.savetxt('output_file.txt', flat_array)
