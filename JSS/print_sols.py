import pickle
import sys

_, fname = sys.argv
with open(f'solutions/{fname}_sol.pkl', 'rb') as f:
    print(pickle.load(f))