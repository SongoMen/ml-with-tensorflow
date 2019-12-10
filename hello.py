from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import math

iris = load_iris() 

x = iris.data
y = iris.target

# classifier

k = KNeighborsClassifier(n_neighbors=2)

k.fit(x,y)

prediction = k.predict([[2,4,3,1],[1,2,3,4]])

result = ''

for i in range(len(prediction)):
        result += ' ' + iris.target_names[i]
print(result)

knn = KNeighborsClassifier(n_neighbors = 5)

# test accuracy

score = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print("accuracy 1: ",math.floor(score.mean()*100))

range1 = range(2,50)

for i in range1:
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    print("accuracy ", i, ": ",math.floor(score.mean()*100))

score = cross_val_score(knn,x,y,cv=10,scoring='accuracy')



