import pickle

results = pickle.load(open('saved/config-results.p', 'rb'))

results.sort(key=lambda x: x[0])

best = results[:21]

pickle.dump(best, open('saved/best-results.p', 'wb'))

