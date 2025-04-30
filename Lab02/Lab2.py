import numpy
import matplotlib.pyplot as mpl

L = numpy.zeros(150)
D = numpy.zeros((150, 4))
c = 0

with open("iris.csv", "r") as file:
    for line in file:
        x = line.strip().split(",")
        flower_type = x.pop(4)
        if flower_type == "Iris-setosa":
            L[c] = 0
        elif flower_type == "Iris-versicolor":
            L[c] = 1
        elif flower_type == "Iris-virginica":
           L[c] = 2
        D[c] = x
        c += 1

D = D.T

D0 = D[:, L == 0].astype(float) #Setosa                  Il comando D0 = D[:, L == 0] inserisce in D0 le colonne di D dove la consizione L == 0 è true
D1 = D[:, L == 1].astype(float) #Versicolor
D2 = D[:, L == 2].astype(float) #Virginica


mpl.figure("Sepal lengths")
mpl.hist(D0[0], bins = 10, density = True)
mpl.hist(D1[0], bins = 10, density = True)
mpl.hist(D2[0], bins = 10, density = True)

mpl.figure("Sepal width")
mpl.hist(D0[1], bins = 10, density = True)
mpl.hist(D1[1], bins = 10, density = True)
mpl.hist(D2[1], bins = 10, density = True)

mpl.figure("Petal lengths")
mpl.hist(D0[2], bins = 10, density = True)
mpl.hist(D1[2], bins = 10, density = True)
mpl.hist(D2[2], bins = 10, density = True)

mpl.figure("Petal widths")
mpl.hist(D0[3], bins = 10, density = True)
mpl.hist(D1[3], bins = 10, density = True)
mpl.hist(D2[3], bins = 10, density = True)

mpl.figure("Sepal width x sepal lengths")
mpl.scatter(D0[0], D0[1])
mpl.scatter(D1[0], D1[1])
mpl.scatter(D2[0], D2[1])
#e così via

mpl.show()

mu = D.mean(1).reshape((D.shape[0], 1))
print('Mean:')
print(mu)
print()

DC = D - mu

C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
print('Covariance:')
print(C)
print()

var = D.var(1)
std = D.std(1)
print('Variance:', var)
print('Std. dev.:', std)
print()

for cls in [0,1,2]:
    print('Class', cls)
    DCls = D[:, L==cls]
    mu = DCls.mean(1).reshape(DCls.shape[0], 1)
    print('Mean:')
    print(mu)
    C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
    print('Covariance:')
    print(C)
    var = DCls.var(1)
    std = DCls.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
