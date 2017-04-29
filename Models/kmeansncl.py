import numpy
from sklearn.cluster import KMeans
from sklearn import preprocessing

X_train = numpy.load('/home/rhett/Projects/GWData/saveddatanos.npy')

scaler = preprocessing.MinMaxScaler()

X_train_minmax = scaler.fit_transform(X_train)

kmeans = KMeans(n_clusters = 100, random_state = 0).fit(X_train_minmax)

print kmeans.labels_
