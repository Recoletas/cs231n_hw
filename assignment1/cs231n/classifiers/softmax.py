from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
 
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # 计算梯度 - 正确的方法
        dscores = p.copy()
        dscores[y[i]] -= 1
        
        # 更新所有权重类别
        for j in range(num_classes):
            dW[:, j] += X[i] * dscores[j]



    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W 

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # 1. 计算所有样本的得分矩阵
    scores = X.dot(W)  # 形状: (N, C)
    N = scores.shape[0]

    # 2. 数值稳定性处理（减去最大值）
    max_scores = np.max(scores, axis=1, keepdims=True)
    scores -= max_scores

    # 3. 计算Softmax概率
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    # 避免除以0的情况
    sum_exp_scores = np.maximum(sum_exp_scores, 1e-16)  # 添加一个小值防止除零
    probabilities = exp_scores / sum_exp_scores

    # 4. 计算损失
    # 确保probabilities不会为0，避免log(0)
    probabilities = np.maximum(probabilities, 1e-16)
    correct_logprobs = -np.log(probabilities[np.arange(N), y])
    data_loss = np.mean(correct_logprobs)
    # 计算正则化损失时使用更稳定的方式
    reg_loss = 0.5 * reg * np.sum(W * W)  # 乘以0.5是常见做法，使得梯度更简洁
    loss = data_loss + reg_loss

    # 5. 计算梯度 - 确保与naive版本一致，且数值稳定
    dscores = probabilities.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    # 梯度计算使用与naive版本相同的系数（乘以2）
    dW = X.T.dot(dscores) + reg * W


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
