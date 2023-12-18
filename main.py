import pickle
file_path = 'data/data_ml_2/data_ml_2_combination/datasets_text_5000.kpl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)
data_text = data['data']
print(data)