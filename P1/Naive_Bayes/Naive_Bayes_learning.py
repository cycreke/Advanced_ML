import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib

import matplotlib.pyplot as plt

x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB() #classfier
clf.fit(x, y) #training
print clf.predict([[-0.8, -1], [1, 1.5]])
