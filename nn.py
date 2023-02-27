"""
A Neural Network from scratch, based on Karpathy's micrograd.
References:
    - https://github.com/karpathy/micrograd
"""

import math

class Value:
    def __init__(self, data, parents = set()):
        self.data = float(data)
            
        self.parents = parents
        
        self.grad = 0.0 # Global derivate, i.e. dL/dv
        
        self._backward = lambda: None  # Applies chain rule
        
    def __repr__(self):
        return f"Value = {self.data}; Gradient = {self.grad}"
    
    def __neg__(self):
        return self * -1
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data     = self.data + other.data,
            parents  = {self, other}
        )
        
        def _backward():
            self.grad  += out.grad # o = a + b; dL/da = dL/do . do/da = dL/do
            other.grad += out.grad # similarly to dL/db
        
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data     = self.data - other.data,
            parents  = {self, other}
        )
        
        def _backward():
            self.grad  += out.grad # o = a - b; dL/da = dL/do . do/da = dL/do
            other.grad -= out.grad # o = a - b; dL/db = dL/do . do/db = - dL/do
        
        self._backward = _backward
        
        return out
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data     = self.data * other.data,
            parents  = {self, other}
        )
        
        def _backward():
            self.grad  += out.grad * other.data # o = a * b; dL/da = dL/do . do/da = dL/do . b
            other.grad += self.data * out.grad  # o = a * b; dL/db = dL/do . do/db = a . dL/do
        
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        return other * self
    
    def __truediv__(self, other):
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data     = self.data * other.data,
            parents  = {self, other}
        )
        
        def _backward():
            self.grad  += out.grad / other.data # o = a / b; dL/da = dL/do . do/da = dL/do / b
            other.grad += self.data / out.grad  # o = a / b; dL/db = dL/do . do/db = a / dL/do
        
        out._backward = _backward
        
        return out
        """
        # a / b == a * (1/b) == a * (b**(-1))
        return self * (other ** -1)
    
    def __rtruediv__(self, other): 
        return other * (self ** -1)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(
            data     = self.data ** other,
            parents  = {self, other}
        )
        
        def _backward():
            # o = a^b; dL/da = dL/do . do/da = dL/do . (b . a ^ (b - 1))
            self.grad  += out.grad * (other * (self.data ** (other - 1)))
            # o = a^b; dL/db = dL/do . do/db = dL/do . (a**b ln a)
            other.grad += out.grad * ((self.data**other) * math.log(self.data)) 
        
        out._backward = _backward
        
        return out
    
    def exp(self):
        out = Value(
            data     = math.exp(self.data),
            parents  = {self}
        )
        
        def _backward():
            # o = e^a; dL/da = dL/do . do/da = dL/do . e^x
            self.grad  += out.grad * out.data # out.data == math.exp(self.data) == e^x
    
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(
            data     = max(0, self.data),
            parents  = {self}
        )
        
        def _backward():
            # o = a.relu(); dL/da = dL/do . do/da = dL/do . (1 if x > 0 else 0 if x < 0)
            self.grad  += out.grad * int(self.data > 0)
    
        out._backward = _backward
        
        return out
    
    def backward(self):
        topsort = []
        visited = set()
        
        def build_topsort(v): 
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topsort(parent)
                topsort.append(v)
        
        build_topsort(self)   
        self.grad = 1.0
        for v in reversed(topsort):
            v._backward()


if __name__ == "__main__":
    """
    NN to fit the Iris dataset.
    4 input layer, 7, 7, 3
    """
    import numpy as np
    import random
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True) # len(X) == 150
    y = y.reshape((150, 1))
    data = np.append(X, y, axis=1)
    np.random.shuffle(data)

    X_train, y_train = data[0:120,:4], data[0:120,4]
    X_test, y_test = data[120:,:4], data[120:,4]

    i1, i2, i3, i4 = Value(0), Value(0), Value(0), Value(0)

    w1s = [[Value(random.uniform(-1,1)) for _ in range(5)] for _ in range(5)] # np.random.uniform(-1, 1, size=(5, 5))
    h11 = (i1*w1s[0][0] + i2*w1s[0][1] + i3*w1s[0][2] + i4*w1s[0][3] + w1s[0][4]).relu()
    h12 = (i1*w1s[1][0] + i2*w1s[1][1] + i3*w1s[1][2] + i4*w1s[1][3] + w1s[1][4]).relu()
    h13 = (i1*w1s[2][0] + i2*w1s[2][1] + i3*w1s[2][2] + i4*w1s[2][3] + w1s[2][4]).relu()
    h14 = (i1*w1s[3][0] + i2*w1s[3][1] + i3*w1s[3][2] + i4*w1s[3][3] + w1s[3][4]).relu()
    h15 = (i1*w1s[4][0] + i2*w1s[4][1] + i3*w1s[4][2] + i4*w1s[4][3] + w1s[4][4]).relu()

    w2s = [[Value(random.uniform(-1,1)) for _ in range(6)] for _ in range(5)] # np.random.uniform(-1, 1, size=(5, 6))
    h21 = (h11*w2s[0][0] + h12*w2s[0][1] + h13*w2s[0][2] + h14*w2s[0][3] + h15*w2s[0][4] + w2s[0][5]).relu()
    h22 = (h11*w2s[1][0] + h12*w2s[1][1] + h13*w2s[1][2] + h14*w2s[1][3] + h15*w2s[1][4] + w2s[1][5]).relu()
    h23 = (h11*w2s[2][0] + h12*w2s[2][1] + h13*w2s[2][2] + h14*w2s[2][3] + h15*w2s[2][4] + w2s[2][5]).relu()
    h24 = (h11*w2s[3][0] + h12*w2s[3][1] + h13*w2s[3][2] + h14*w2s[3][3] + h15*w2s[3][4] + w2s[3][5]).relu()
    h25 = (h11*w2s[4][0] + h12*w2s[4][1] + h13*w2s[4][2] + h14*w2s[4][3] + h15*w2s[4][4] + w2s[4][5]).relu()

    w3s = [[Value(random.uniform(-1,1)) for _ in range(6)] for _ in range(3)] # np.random.uniform(-1, 1, size=(3, 6))
    o1aux = (h21*w3s[0][0] + h22*w3s[0][1] + h23*w3s[0][2] + h24*w3s[0][3] + h25*w3s[0][4] + w3s[0][5])
    o2aux = (h21*w3s[1][0] + h22*w3s[1][1] + h23*w3s[1][2] + h24*w3s[1][3] + h25*w3s[1][4] + w3s[1][5])
    o3aux = (h21*w3s[2][0] + h22*w3s[2][1] + h23*w3s[2][2] + h24*w3s[2][3] + h25*w3s[2][4] + w3s[2][5])

    den = o1aux.exp() + o2aux.exp() + o3aux.exp()

    o1 = o1aux.exp() / den 
    o2 = o2aux.exp() / den
    o3 = o3aux.exp() / den 

    # TODO: define loss

    # TODO: train
    
    # TODO: test