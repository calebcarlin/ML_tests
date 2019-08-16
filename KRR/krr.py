import numpy as np
from math import exp
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.fftpack import fft

def K(x1,x2,sigma):
    """
    kernel function
    """
    distance = cdist(np.atleast_2d(x1),np.atleast_2d(x2))

    return np.exp(-0.5*np.square(distance/sigma))

def Determine_alphas(X,lambdax,sigma,y_val):
    """
    Solve for the covariance matrix, solve the inverse
    with the regularizer, and then determine the alphas
    that minimize the original function.
    Also calculates the eigen vectors and values, as
    those are generally useful for analysis.
    """

    lambdaI = np.eye(len(X)) * lambdax
    K_matrix = K(X,X,sigma)
    
    K_lambdaI = K_matrix + lambdaI
    K_inv = np.linalg.inv(K_lambdaI)
    alpha = np.matmul(K_inv,y_val)

    eig_val,eig_fun = np.linalg.eig(K_matrix)

    return alpha, eig_val, eig_fun

def predict_y(alpha,X,X_i,y_avg,sigma):
    """
    given the set of coefficents and the set of features,
    predicts the value corresponding to a new input X_i
    y_avg is the averge of the known output values because
    the algorithm tends toward zero for poorly modeled inputs
    unless we add in the average
    """

    add_up = np.sum(np.matmul(K(X,X_i,sigma).T,alpha))+y_avg

    return add_up

def cross_validate(y_val,fingerprints,k,lambdax,sigmai,y_ave):
    """
    Split up the data set and use the mean square error to measure
    the precision of the combination of hyperparameters.
    """
    
    KF = KFold(k,True)
    avg_error = 0
    for train, test in KF.split(fingerprints):
        error = 0
        alpha,eig_val,eig_fun = Determine_alphas(fingerprints[train],lambdax,sigma,y_val[train])
        for i in test:
            y_prime = predict_y(alpha, fingerprints[train],fingerprints[i],y_ave,sigma)
            error += (y_prime - y_val[i])**2
        avg_error += error

    return avg_error/k

def pca(finger,eigen_fun,n_vec,eigen_val):
    """
    take the n_vec eigen functions corresponding to the largest
    eigen values and use them to project the known fingerprints
    in the feature space of the PCA.
    """
    feature_vec = []
    indices = np.argsort(eigen_val)
    coeff = 0
    for i in range(n_vec):
        feature_vec.append(eigen_fun[indices[-i-1]])
        coeff += (eigen_val[indices[-i-1]])

    prin_comp = np.matmul(np.asarray(feature_vec),finger)

    return

if __name__ == "__main__":

    lambdax = 1.0e-3
    sigma = 0.1
    sigma_f = 0.025
    
    #read in fingerprints  of known and unknown
    #data set.
    raw_fingerprints = np.loadtxt('raw_finger')
    unknowns = np.loadtxt('raw_unknown')

    #read in known values and shift to the average
    y_val = np.loadtxt('known_be')
    y_ave = np.mean(y_val)
    y_val -= y_ave
    
    #Apply gaussian blur to known fingerprints
    fingerprints = []
    for j in raw_fingerprints:
        fingerprints.append(gaussian_filter(j,sigma_f))
    fingerprints = np.asarray(fingerprints)
    
    #Apply gaussian blur to unknown fingerprints
    unknown_finger = []
    for j in unknowns:
        unknown_finger.append(gaussian_filter(j,sigma_f))
    unknown_finger = np.asarray(unknown_finger)

    #Fit the model by solving for the weights.
    alpha,eig_val,eig_fun = Determine_alphas(fingerprints,lambdax,sigma,y_val)


#    pca(fingerprints,eig_fun,4,eig_val)
    #lines for cross validating.  These should be included inside loops over the possible
    #values of the hyper-parameters but does little as a stand alone.
#    for sigma in [0.001,0.025,0.1,1.0,2.0,3.0]:
#        average_err = cross_validate(y_val,fingerprints,10,lambdax,sigma,y_ave)
#        print("lambda = "+str(lambdax)+" sigmaf = "+str(sigma_f)+" sigma = "+str(sigma)+" ave_err = "+str(average_err))
    
    #output the alpha weights in case we want to just read them in
#    np.savetxt('alphas',alpha)
   
    #using the KRR model, reads in the unknown fingerprints and returns the predicted
    #value after accounting for the previously applied shift to account for the bias.
    for i in range(len(unknown_finger)):
        y_prime = predict_y(alpha,fingerprints,unknown_finger[i],y_ave,sigma)
        print(y_prime)
