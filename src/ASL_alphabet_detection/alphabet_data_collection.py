import string
from src.helper_functions import collect_data
import cv2
import mediapipe as mp
import time
import numpy as np
import csv


labels = [letter for letter in string.ascii_uppercase if letter not in ('J', 'Z')] + ['space', 'delete', 'nothing']



one_hot_dict = collect_data('../../data/alphabet_data.csv', labels)


