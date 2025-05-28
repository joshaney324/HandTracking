import string
from src.helper_functions import collect_data

labels = [letter for letter in string.ascii_uppercase if letter not in ('J', 'Z')] + ['space', 'delete']
# print(labels)


one_hot_dict = collect_data('../../data/alphabet_data.csv', labels)


