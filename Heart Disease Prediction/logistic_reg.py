# Do with Logistic Regression
# Functions
import copy
import math
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    
    cost = cost/m
    return cost

def compute_gradient_descent(x,y,w,b):
    m,n = x.shape
    dj_w = np.zeros(n,)
    dj_b = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i],w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_w[j] = dj_w[j] + err_i* x[i,j]
        dj_b = dj_b + err_i
    dj_w /= m
    dj_b /= m
    
    return dj_w, dj_b

def gradient_descent(x,y,w_in,b_in,alpha,epochs):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    initial_cost = compute_cost(x, y, w, b)
    j_history.append(initial_cost)
    
    for i in range(epochs):
        dj_w, dj_b = compute_gradient_descent(x,y,w,b)
        
        w = w - alpha * dj_w
        b = b - alpha * dj_b
        
        if i<100000:
            j_history.append( compute_cost(x,y,w,b) )
        if i % math.ceil(epochs/10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]}")
    return w, b

def predict(x,w,b):
    m = x.shape[0]
    pred = []
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        # therehold
        if sigmoid(z_i) <= 0.5:
            pred.append(0)
        elif sigmoid(z_i) > 0.5:
            pred.append(1)
    return np.array(pred)

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_pred == y_true)
    accuracy = correct_predictions / len(y_true)
    return "{:.2f}%".format(accuracy * 100)