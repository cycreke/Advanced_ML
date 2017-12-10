from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print clf.predict([[2, 2],[-1, -1]])

import scipy.stats
print scipy.stats.entropy([1, 1],base=2)