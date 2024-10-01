import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(PHI, y, w):
    return -np.dot((y - np.dot(PHI,w)),PHI)

def total_gradient(X, Y, W):
    total = 0
    for j in range(4):
        phi = np.array([1, X[j][0], X[j][1]])
        total += gradient(phi, Y, W)
    return total

W = np.array([0, 0 ,0])
eta = 0.01
X = [[1,1], [-1,1], [-2,-1], [-1,-1]]
Y = [1,0,1,0]

for i in range(2):
    W = W - eta*total_gradient(X,Y[i],W)

print(W)