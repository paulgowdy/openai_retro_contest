import pickle

with open('roc_results.p', 'rb') as f:

    results = pickle.load(f)

for z in results:

    print z
