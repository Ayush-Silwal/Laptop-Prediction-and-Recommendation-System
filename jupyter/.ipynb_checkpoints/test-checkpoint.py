import pickle
from customModel import DecisionTree,RandomForest   
# Replace 'your_pickle_file.pkl' with the path to your pickle file
pickle_file_path = 'pipe.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Inspect the data
print(type(data))  # Check the type of the data
print(data)        # Print the contents of the data
