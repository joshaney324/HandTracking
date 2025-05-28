from src.helper_functions import collect_data


labels = ['1', '2', '3', '4', '5']

one_hot_dict = collect_data('../../data/number_data.csv', labels)
