import pca
import lda
import numpy
import matplotlib.pyplot as plt

from lda import vcol, vrow, load_iris

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)


if __name__ == '__main__':

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

    # PCA
    print("\n\n ******* PCA ******** \n\n")
    UD_pca = pca.compute_pca(D, 1)
    D_pca = pca.apply_pca(UD_pca, D)

    label_mapping = {0: "False", 1: "True"}

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[0, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 0')
    plt.title('PCA 0')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[1, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 1')
    plt.title('PCA 1')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[2, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 2')
    plt.title('PCA 2')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[3, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 3')
    plt.title('PCA 3')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[4, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 4')
    plt.title('PCA 4')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[5, L == label], bins=10, alpha=0.6, label=label_mapping[label])

    plt.xlabel('Principal Component 5')
    plt.title('PCA 5')
    plt.legend()
    plt.grid()
    plt.show()


    # LDA
    print("\n\n ******* LDA ******** \n\n")
    UD_lda = lda.compute_lda_JointDiag(D, L, m=1)                # Recompute LDA on the raw data
    D_lda = lda.apply_lda(UD_lda, D)                             # Apply LDA to the raw data
    if D_lda[0, L == 0].mean() > D_lda[0, L == 1].mean():
        UD_lda = -UD_lda
        DTR_lda = lda.apply_lda(UD_lda, D)

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_lda[0, L == label], bins=30, alpha=0.6, label=label_mapping[label])
    
    plt.title('LDA')
    plt.legend()
    plt.grid()
    plt.show()

    # LDA as classifier with different threshold
    print("\n\n ******* LDA as classifier with different threshold ******** \n\n")
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    ULDA = lda.compute_lda_JointDiag(DTR, LTR, m=1)
    DTR_lda = lda.apply_lda(ULDA, DTR)

    if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)
    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    err = 100
    for i in numpy.arange(-0.1, 0, 0.0001):
        threshold = i

        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        print('Labels:     ', LVAL)
        print('Predictions:', PVAL)
        print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
        print(threshold)
        if ((PVAL!= LVAL).sum()/ float(LVAL.size) * 100) < err:
            err = (PVAL!= LVAL).sum()/ float(LVAL.size) * 100
            th = threshold
    print("Migliore th", th, "err", err)

    # PCA pre-processing and LDA classifier
    print("\n\n ******* PCA pre-processing and LDA classifier ******** \n\n")
    for m in numpy.arange(1, 6, 1):
        UPCA = pca.compute_pca(DTR, m)[:, 0:m]  # Estimated only on model training data
        DTR_pca = pca.apply_pca(UPCA, DTR)   # Applied to original model training data
        DVAL_pca = pca.apply_pca(UPCA, DVAL) # Applied to original validation data
        ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, 1)  # Estimated only on model training data, after PCA has been applied
        
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)  # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
        
        if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
            ULDA = -ULDA
            DTR_lda = lda.apply_lda(ULDA, DTR_pca)

        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)  # Applied to PCA-transformed validation data

        threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0  # Estimated only on model training data

        PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        print("m uguale a ", m)
        print('Labels:     ', LVAL)
        print('Predictions:', PVAL)
        print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))