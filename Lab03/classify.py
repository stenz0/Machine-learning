import pca
import lda
import numpy
import matplotlib.pyplot as plt

from lda import vcol, vrow, load_iris

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
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
    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    #D = DIris  # Utilizzare tutti i dati senza filtrare
    #L = LIris  # Utilizzare tutte le etichette senza filtrare
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = lda.compute_lda_JointDiag(DTR, LTR, m=1)

    DTR_lda = lda.apply_lda(ULDA, DTR)

    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0  # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))
          
    # Solution with PCA pre-processing with dimension m.
    m = 2
    UPCA = pca.compute_pca(DTR, m)  # Estimated only on model training data
    DTR_pca = pca.apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = pca.apply_pca(UPCA, DVAL) # Applied to original validation data

    ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m=1)  # Estimated only on model training data, after PCA has been applied

    DTR_lda = lda.apply_lda(ULDA, DTR_pca)  # Applied to PCA-transformed model training data, 
    # the projected training samples are required to check the orientation of the direction and to compute the threshold
    
    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation
    if DTR_lda[0, LTR == 1].mean() > DTR_lda[0, LTR == 2].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

    DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)  # Applied to PCA-transformed validation data

    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0  # Estimated only on model training data

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    label_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    # Plotting the PCA results
    D_pca = pca.apply_pca(UPCA, D)  # Apply PCA to the entire dataset
    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.scatter(D_pca[0, L == label], D_pca[1, L == label], label=label_mapping[label])
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of IRIS dataset')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting the LDA results on raw data
    ULDA_raw = lda.compute_lda_JointDiag(D, L, m=2)  # Recompute LDA on the raw data
    D_lda_raw = lda.apply_lda(ULDA_raw, D) * -1 # Apply LDA to the raw data

    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.scatter(D_lda_raw[0, L == label], D_lda_raw[1, L == label], label=label_mapping[label])
    
    plt.xlabel('LDA Component 1')
    plt.ylabel('Zero Axis')
    plt.title('LDA of IRIS dataset (Raw Data)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting the PCA on one direction
    D_pca = pca.apply_pca(UPCA, D)  # Apply PCA to the entire dataset
    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_pca[0, L == label], bins=10, alpha=0.6, label=label_mapping[label])
    
    plt.xlabel('Principal Component 1')
    plt.title('PCA of IRIS dataset')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting the LDA results on PCA results data
    D_lda = lda.apply_lda(ULDA, D_pca) *-1  # Apply LDA to the PCA-transformed data
    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(D_lda[0, L == label], bins=9, alpha=0.6, label=label_mapping[label])


    plt.xlabel('LDA Component 1')
    plt.ylabel('Zero Axis')
    plt.title('LDA of IRIS dataset')
    plt.legend()
    plt.grid()
    plt.show()

    # Using LDA as classifier, feature 0
    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(DTR_lda[0, LTR == label], bins=5, alpha=0.6, label=label_mapping[label])

    plt.title('DTR, LTR')
    plt.legend()
    plt.grid()
    plt.show()

    # Using LDA as classifier, feature 1
    plt.figure(figsize=(8, 6))
    for label in numpy.unique(L):
        plt.hist(DVAL_lda[0, LVAL == label], bins=5, alpha=0.6, label=label_mapping[label])

    plt.title('DVAL, LVAL')
    plt.legend()
    plt.grid()
    plt.show()