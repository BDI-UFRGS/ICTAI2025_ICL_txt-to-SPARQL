from typing import List
import pickle

def load_object(obj_ref: str):
    with open(obj_ref, 'rb') as file:
        obj = pickle.load(file)
        return obj