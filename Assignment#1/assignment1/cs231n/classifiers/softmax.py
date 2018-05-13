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
  data_loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass - You shall not pass:

  num_ex = X.shape[0]
  num_cl = W.shape[1]

  for i in range(num_ex):

      # Extract the ith training example:
      X_i = X[i,:]

      # Compute the linear scores for the classes:
      # (1 x C) -> (1 x D)*(D x C)
      scores = np.dot(X_i, W)

      # Transalte the scores so that the max value is zero:
      scores -= np.max(scores)

      # Compute the exponential of the ith example:
      exp_sc_i = np.exp(scores)

      # Compute the class probabilities using the softmax function:
      probs = exp_sc_i/(np.sum(exp_sc_i, axis=0, keepdims = True))

      # Compute the loss function for the ith example:
      for j in range(num_cl):

          dW[:,j] += probs[j]*X_i

          # Only consider the probability of the correct class:
          if y[i] ==j:

              data_loss += -np.log(probs[j])

              dW[:,j] -= X_i



   # Take the mean of the data loss and the gradient:
  data_loss /= num_ex
  dW /= num_ex

   # Now compute the reg loss:
  reg_loss = 0.5*reg*np.sum(W*W)

  # Add the regularization contribution to the gradient:
  dW += reg*W

    # Total loss:
  loss = data_loss + reg_loss


  #############################################################################
  #                          END OF YOUR CODE                                 #
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass

  num_ex = X.shape[0]

  # Dimension of scores is:
  # (N x C) -> (N x D)*(D x C)
  scores = np.dot(X,W)

  scores -= np.max(scores)

  # Compute the probabilities of each class using the softmax function:
  exp_sc = np.exp(scores)

  probs = exp_sc/(np.sum (exp_sc, axis =1 , keepdims = True))

  data_loss = np.sum(-np.log(probs[np.arange(num_ex), y]), axis =0)

  data_loss /= num_ex

  # Compute the gradient of the loss function w.r.t to the probabilities:
  dprobs = probs
  dprobs[np.arange(num_ex), y] -= 1

  # Compute the gradient of the loss function w.r.t weights:
  dW = np.dot(X.T,dprobs)/(num_ex)

  # Regularization loss:
  reg_loss = 0.5*reg*np.sum(W*W)

  # Add the regularization contribution to the gradient:
  dW += reg*W

  # Total loss:
  loss = data_loss + reg_loss


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
