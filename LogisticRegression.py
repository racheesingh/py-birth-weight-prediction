#!/usr/bin/python
from numpy import loadtxt, zeros, ones, array, e, where
import numpy as np
from scipy import optimize

def sigmoid( X ):
    '''Compute the sigmoid function '''

    sigmoidValue = 1.0 / ( 1.0 + e ** (-1.0 * X) )

    return sigmoidValue

def compute_cost( X, y, theta, lam ):

    '''Compute cost for logistic regression.'''
    
    # Number of training examples
    m = y.shape[0]

    # Compute the prediction based on theta and X
    predictions = X.dot( theta )

    # Preprocessing values before sending to sigmoid function.
    # If the argument to sigmoid function >= 0, we know that the
    # sigmoid value is 1. Similarly for the negative values.
    predictions[ where( predictions >= 20 ) ] = 20
    predictions[ where( predictions <= -500 ) ] = -500
    hypothesis = sigmoid( predictions )

    hypothesis[ where( hypothesis == 1.0 ) ] = 0.99999

    # Part of the cost function without regularization
    J1 = ( -1.0 / m ) * sum( ( y * np.log( hypothesis ) ) + 
                            ( ( 1.0 - y ) * np.log( 1.0 - hypothesis ) ) ) 

    # Computing the regularization term
    J2 = lam / ( 2.0 * m ) * sum( theta[ 1:, ] * theta[ 1:, ] )
    error = hypothesis - y

    return J1 + J2

def predict(theta, X):
    
    ''' Predict the probability of y = 1 '''

    m, n = X.shape
    p = zeros(shape=(m, 1))
    
    h = sigmoid(X.dot(theta.T))
    
    for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0

    return p

def map_feature( X ):

    ''' Maps the input features to all quadratic and cubic features.
    Returns a new feature array with more features, comprising of
    x0, x1, x2 .. x6, x0^2 .. , x0^3 .. '''

    m = X.shape[0]
    out = ones(shape=( m, 36+84 ))

    # Setting the first 7 features as x1, x2 .. x6
    out[ :, 1:8 ] = X[:, ]

    # Maintaining count of the number of features added
    count = 7

    # Adding all quadratic features
    for i in range( 7 ):
        for j in range( i + 1 ):
            count += 1
            out[ :, count ] = X[ :, i ] * X[ :, j ]

    # Adding all cubic features
    for i in range( 7 ):
        for j in range( i + 1 ):
            for k in range( j + 1 ):
                count += 1
                out[ :, count ] = X[ :, i ] * X[ :, j ] * X[ :, k ]

    return out


def logisticRegression( X, y, m ):

    # Initial values of theta
    initial_theta = zeros( shape=( X.shape[1], 1 ) )

    # Set regularization parameter lambda to 1
    # Tweak this a little bit to see if accuracy improves.
    lam = 2;

    # Optimizing theta using built-in function
    # Increasing maximum iterations increases the accuracy since
    # J reduces with number of iterations but then the J curve flattens.
    options = {'full_output': True, 'maxiter': 1000000, 'maxfun': 10000000}

    theta, cost, _, _, _ = \
        optimize.fmin(lambda t: compute_cost( X, y , t, lam ), 
                      initial_theta, **options)

    print 'Cost at theta found by fminunc: %f' % cost
    print 'theta: %s' % theta
    
    # Compute accuracy on our training set
    p = predict(array(theta), X)
    count = 0
    for i in range( m ):
        if p[i] == y[i]:
            count += 1
    accuracy = ( count * 100.0 ) / m 
    print 'Accuracy of prediction is:', accuracy

def main():
    # Loading the dataset
    data = loadtxt( 'lowbwtm11.dat' )

    # Number of features in each training example (remove 1st 2 columns
    num_features = data.shape[1] - 2

    # Number of training examples
    m = data.shape[0]

    # Features
    X = data[ :, 2 : data.shape[1] ]

    # Map features x1, x2 ... x7 to all their quadratic terms
    # X would contain, x1 .. x7, x1^2, x1x2 and so on
    it = map_feature( X )
    print it.shape

    # Target variables
    y = data[ :, 1 ]

    # Lets plot all combinations of 2 features from the 7 we have
    #for i in range( 6 ):
    #    for j in range( 6 ):
    #        Xsmall[ :, 0 ] = X[ :, i ]
    #        Xsmall[ :, 1 ] = X[ :, j ]
    #        logisticRegression( Xsmall, y, m )

    logisticRegression( it, y, m )

            
if __name__ == "__main__":
    main()
