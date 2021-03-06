import sys
sys.path.append(r'C:\Users\xcm\Desktop\udacity\advanced_ML\P1\Naive_Bayes')
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(C=1000, kernel='rbf')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print accuracy_score(pred, labels_test)
prettyPicture(clf, features_test, labels_test)
