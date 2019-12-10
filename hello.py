from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris() 

x = iris.data
y = iris.target

k = KNeighborsClassifier(n_neighbors=2)
k.fit(x,y)
prediction = k.predict([[2,4,3,1],[1,2,3,4]])
print(iris.feature_names)
for i in range(len(prediction)):
        print(iris.target_names[i])


