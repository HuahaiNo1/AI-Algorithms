import numpy as np

def relu(self, Z):
    """ReLU激活函数"""
    return np.maximum(0, Z)

def relu_derivative(self, Z):
    """ReLU激活函数导数"""
    return (Z > 0).astype(float)

def sigmoid(self, Z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-Z))

def forward(self, X):

    # 从参数字典中获取参数
    W1 = self.params['W1']  # 隐藏层权重
    b1 = self.params['b1']  # 隐藏层偏置
    W2 = self.params['W2']  # 输出层权重
    b2 = self.params['b2']  # 输出层偏置

    # 隐藏层前向传播
    Z1 = np.dot(W1, X) + b1  # 线性变换: W1 * X + b1
    A1 = self.relu(Z1)  # ReLU激活函数

    # 输出层前向传播
    Z2 = np.dot(W2, A1) + b2  # 线性变换: W2 * A1 + b2
    A2 = self.sigmoid(Z2)  # 使用sigmoid激活

    # 缓存中间结果，用于反向传播
    cache = {
        'Z1': Z1,  # 隐藏层线性输出
        'A1': A1,  # 隐藏层激活输出
        'Z2': Z2,  # 输出层线性输出
        'A2': A2,  # 输出层激活输出
        'X': X  # 输入数据
    }

    return A2, cache

def backward(self, cache, Y):

    # 获取样本数量
    m = Y.shape[1]
    # 从前向传播缓存中获取中间计算结果
    Z1, A1, Z2, A2, X = cache['Z1'], cache['A1'], cache['Z2'], cache['A2'], cache['X']

    # 1. 输出层梯度计算
    dZ2 = A2 - Y  # 输出层误差

    # 2. 输出层参数梯度
    dW2 = np.dot(dZ2, A1.T) / m  # W2的梯度 = 输出层误差 × 隐藏层输出转置，再求平均
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # b2的梯度 = 输出层误差求和，再求平均

    # 3. 隐藏层梯度反向传播
    dA1 = np.dot(self.params['W2'].T, dZ2)  # 隐藏层输出A1的梯度 = W2转置 × 输出层误差
    dZ1 = dA1 * self.relu_derivative(Z1)  # 隐藏层线性输出Z1的梯度，考虑ReLU激活函数的导数

    # 4. 隐藏层参数梯度
    dW1 = np.dot(dZ1, X.T) / m  # W1的梯度 = 隐藏层误差 × 输入数据转置，再求平均
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # b1的梯度 = 隐藏层误差求和，再求平均

    # 5. 返回所有参数的梯度字典
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

