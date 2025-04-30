import numpy
import matplotlib.pyplot as mpl

L = numpy.zeros(6000)
D = numpy.zeros((6000, 6))
c = 0

with open("./trainData.txt", "r") as file:
    for line in file:
        x = line.strip().split(",")
        L[c] = x.pop(6)
        D[c] = x
        c += 1

D = D.T

D0 = D[:, L == 0].astype(float) #False                 Il comando D0 = D[:, L == 0] inserisce in D0 le colonne di D dove la consizione L == 0 è true
D1 = D[:, L == 1].astype(float) #True

mpl.figure("Feature 0")
mpl.hist(D0[0], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[0], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Feature 1")
mpl.hist(D0[1], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[1], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Feature 2")
mpl.hist(D0[2], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[2], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Feature 3")
mpl.hist(D0[3], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[3], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Feature 4")
mpl.hist(D0[4], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[4], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Feature 5")
mpl.hist(D0[5], color="orange", bins = 10, density = True, alpha = 0.8)
mpl.hist(D1[5], color="blue", bins = 10, density = True, alpha = 0.8)

mpl.figure("Features 0 x 1")
mpl.scatter(D0[0], D0[1], color="orange", label="Classe1", alpha = 0.4)
mpl.scatter(D1[0], D1[1], color="blue", label="Classe2", alpha = 0.4)

mpl.figure("Features 1 x 2")
mpl.scatter(D0[1], D0[2], color="orange", label="Classe1", alpha = 0.4)
mpl.scatter(D1[1], D1[2], color="blue", label="Classe2", alpha = 0.4)

mpl.figure("Features 2 x 3")
mpl.scatter(D0[2], D0[3], color="orange", label="Classe1", alpha = 0.4)
mpl.scatter(D1[2], D1[3], color="blue", label="Classe2", alpha = 0.4)

mpl.figure("Features 3 x 4")
mpl.scatter(D0[3], D0[4], color="orange", label="Classe1", alpha = 0.4)
mpl.scatter(D1[3], D1[4], color="blue", label="Classe2", alpha = 0.4)

mpl.figure("Features 4 x 5")
mpl.scatter(D0[4], D0[5], color="orange", label="Classe1", alpha = 0.8)
mpl.scatter(D1[4], D1[5], color="blue", label="Classe2", alpha = 0.8)

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

for cls in [0,1]:
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
