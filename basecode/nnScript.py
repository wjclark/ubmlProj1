
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
import math
import random as rand

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = math.sqrt(6) / math.sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    
    return  1/(1.0+math.e**-z) #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat')

    train_data = np.empty((50000,784))
    train_label = np.empty((50000,1))
    validation_data = np.empty((10000,784))
    validation_label = np.empty((10000, 1))
    test_data = np.empty((0,784))
    test_label = np.empty((0,1))


    #testing = empty((0,784))
    #testing_label = empty((0,1))
    training = np.empty((0,784))
    training_label = np.empty((0,1))
    test = "test"
    for i in xrange(10):
        testArr = mat[test + str(int(i))]
        length = len(testArr)
        labels = np.ones((length, 1)) * i
        test_label = np.concatenate((test_label, labels))
        test_data = np.concatenate((test_data, testArr))
    #print "\n", test_data.shape
    #print test_label.shape

    train = "train"
    for i in xrange(10):
        trainArr = mat[train + str(int(i))]
        length = len(trainArr)
        labels = np.ones((length, 1)) * i
        training = np.concatenate((training, trainArr))
        training_label = np.concatenate((training_label, labels))
    #print "\n", training.shape
    #print training_label.shape

    test_data = test_data/255
    training = training/255
    
    training_indecies = rand.sample(xrange(60000), 50000)
    training_indecies = sorted(training_indecies)
    validation_indecies = []
    offset = 0
    try:
        for i in range(60000):
            if training_indecies[i - offset] != i:
                offset += 1
                validation_indecies.append(i)
    except:
        #print i
        #print training_indecies
        #print validation_indecies
        for x in range(i, 60000):
            validation_indecies.append(x)
            

    for i in range(50000):
        new_data = training[training_indecies[i]].reshape(1,784)
        new_label = training_label[training_indecies[i]].reshape(1,1)
        train_data[i] = new_data
        train_label[i] = new_label

    for i in range(10000):
        new_data = training[validation_indecies[i]].reshape(1,784)
        new_label = training_label[validation_indecies[i]].reshape(1,1)
        validation_data[i] = new_data
        validation_label[i] = new_label
    print "done preprocessing"
    
    return train_data[0:100,:], train_label[0:100,:], validation_data[0:100,:], validation_label[0:100,:], test_data[0:100,:], test_label[0:100,:]
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    iteration = 0
    
    #Your code here
    def feedforward_prop(input_array, w1, w2, n_hidden, n_class): #for a single training/validation/test entry
        a_array = np.zeros((n_hidden))
        z_array = np.zeros((n_hidden))
        b_array = np.zeros((n_class))
        o_array = np.zeros((n_class))
        #calculate a_array values
        for j in xrange(n_hidden):
            for i in xrange(len(input_array + 1)):
                a_array[j] += w1[j,i]*input_array[i]
        #for j in range(n_hidden):
        z_array = sigmoid(a_array)

        for l in range(n_class):
            for j in range(n_hidden):
                b_array[l] += w2[l,j]*z_array[j]
        o_array = sigmoid(b_array)
        return o_array, z_array
    
    
    def error_func(output_array, y, n_class): #output_array from feedforward, y = true label from training_labels
        yp = np.array([y == i for i in range(n_class)], dtype = int)
        return .5*sum((yp - output_array)**2)

    def calc_delta(y, output_array, n_class):
        yp = np.array([y == i for i in range(n_class)], dtype = int)
        delta = np.array([(yp[l] - output_array[l])*(1-output_array[l])*output_array[l] for l in xrange(n_class)])
        #print yp.shape, output_array.shape, delta.shape
        return delta
    
    def calc_grad2(z_array, y, output_array, n_class):
        #J_grad = zeros((len(input_array), n_class))
        delta = calc_delta(y,output_array, n_class)
        #print z_array.shape
        #print delta.shape
        J_grad2 = np.array([(delta[l] * z_array[j]) for l in xrange(n_class) for j in xrange(len(z_array))])
        return J_grad2.reshape((n_class, len(z_array)))

    def calc_grad1(input_array, w2, delta, n_class):
        #sum_del_w = 0
        #for l in range(n_class):
            #sum_del_w += delta[l]*w[####################################
        J_grad1 = np.array([-1*(1-z_array[j])*z_array[j]*sum(delta*w2[:,j])*(input_array[i]) for j in xrange(w2.shape[1] - 1) for i in xrange(input_array.shape[0])]).reshape((w2.shape[1] - 1, input_array.shape[0]))
    #obj_grad return val is sum of J_grad1 over all training data *1/n (n=num training examples)
        return J_grad1
    
    def back_prop(grad1, grad2):
        learning_rate = .1
        print w1.shape, w2.shape
        for j in xrange(w1.shape[0]):
            for i in xrange(w1.shape[1] - 1):
                w1[j,i] -= learning_rate*grad1[j,i]
        for l in xrange(w2.shape[0]):
            for j in xrange(w2.shape[1] - 1):
                w2[l,j] -= learning_rate*grad2[l,j]
        #w1_new = w1 - learning_rate*grad1
        #w2_new = w2 - learning_rate*grad2
        return w1_new, w2_new
    
    
    obj_val = 0
    obj_grad1 = np.zeros((w1.shape[0],training_data.shape[1]))
    obj_grad2 = np.zeros((w2.shape[0], w2.shape[1] -1))
    for j in xrange(len(training_data)):
        o_array, z_array = feedforward_prop(training_data[j], w1, w2, n_hidden, n_class)
        delta = error_func(o_array, training_label[j], n_class)
        obj_val += delta
        obj_grad2 += calc_grad2(z_array, training_label[j], o_array, n_class)
        obj_grad1 += calc_grad1(training_data[j], w2, delta, n_class)
        #w1, w2 = back_prop(obj_grad1, obj_grad2)
        learning_rate = .1
        #print w1.shape, obj_grad1.shape, w2.shape, obj_grad2.shape
        for j in xrange(w1.shape[0]):
            for i in xrange(w1.shape[1] - 1):
                w1[j,i] -= learning_rate*obj_grad1[j,i]
        for l in xrange(w2.shape[0]):
            for j in xrange(w2.shape[1] - 1):
                w2[l,j] -= learning_rate*obj_grad2[l,j]
        
    obj_grad = np.concatenate((obj_grad1.flatten(), obj_grad2.flatten()),0)
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    
    print iteration 
    iteration += 1
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    labels = np.empty((len(data)))
    for x in xrange(len(data)):
        a_array = np.zeros((n_hidden))
        z_array = np.zeros((n_hidden))
        b_array = np.zeros((n_class))
        o_array = np.zeros((n_class))
        #calculate a_array values
        for j in xrange(n_hidden):
            for i in xrange(len(input_array + 1)):
                a_array[j] += w1[j,i]*input_array[i]
        #for j in range(n_hidden):
        z_array = sigmoid(a_array)
        for l in range(n_class):
            for j in range(n_hidden):
                b_array[l] += w2[l,j]*z_array[j]
        o_array = sigmoid(b_array)
        pred = 0
        for i in range(n_class):
            if o_array[i] > o_array[pred]:
                pred = i
        labels[x] = i
    
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
print "minimized"
#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Test Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
