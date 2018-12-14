import pickle

results = pickle.load(open('saved/config-results.p', 'rb'))

results.sort(key=lambda x: x[0])

best = results[:21]
for i in best:
    print(i)

pickle.dump(best, open('saved/best-results.p', 'wb'))

