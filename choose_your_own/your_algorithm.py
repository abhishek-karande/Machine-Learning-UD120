import sys
sys.path.append("../choose_your_own/")
from prep_terrain_data import makeTerrainData
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.metrics import accuracy_score
X_train, y_train, X_test, y_test= makeTerrainData()
neigh=KNeighborsClassifier(n_neighbors=4)
t0=time()
neigh.fit(X_train,y_train)
print 'training time is:',round(time()-t0,3),'s'
t0=time()
pred=neigh.predict(X_test)
print 'prediction time is:',round(time()-t0,3),'s'
acc=accuracy_score(pred,y_test)
print acc