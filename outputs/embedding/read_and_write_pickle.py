import pickle
import pandas as pd

# Load data from pickle file
with open(
    "./vector_embeddings_nibe_f2040_231844-5_and_Vaillant_flexoTHERM_400V.pkl", "rb"
) as pickle_file:
    data = pickle.load(pickle_file)
# Convert data to a string using pformat
# Check if the data is a pandas DataFrame
if isinstance(data, pd.DataFrame):
    data_str = data.to_string()
# Write data to text file
with open("output.txt", "w") as text_file:
    text_file.write(data_str)
