import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero - num_classes x data_size

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1

            if margin > 0:
                loss += margin
                dW[:,j] += X[i,:].T
                dW[:,y[i]] -= X[i,:].T

 # Compute the gradient for the:


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)

  # Normalize the gradient:
  dW /= num_train

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # Set the parameter delta to 1:
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.
  # Compare the speed in performance by using vectorization
  #############################################################################
  # pass  -  You shall not pass :P


  # 1. Compute the scores as the linear mapping scores = W*X
  # W - num_classes x num_pixels
  # X - num_pixels x num_train
  # scores - num_classes x num_train
  scores = X.dot(W)

  # 2. Compute the scores of the correct classes:
  # correct_class_scores - num_train x 1
  correct_class_scores = scores[np.arange(num_train), y]

  # 3. Compute the margins for each class in each training example:
  # margins - num_classes x num_train
  # Broadcasting should work over here : Works like a charm!!!
  margins = np.maximum(0,scores - correct_class_scores[:, np.newaxis] + delta)

  # 4. Set the margins of the correct classes to zero:
  margins[np.arange(num_train), y] = 0

  # 5. Compute the loss as the mean of all the thresholded margins:
  loss = np.sum(margins)

  loss /= num_train

  # 6. Add the regularization loss:
  loss += 0.5*reg*np.sum(W*W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass - You shall not pass :P

    # Fully vectorized version. Roughly 10x faster.
  X_mask = np.zeros(margins.shape)
  # column maps to class, row maps to sample; a value v in X_mask[i, j]
  # adds a row sample i to column class j with multiple of v
  X_mask[margins > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train # average out weights
  dW += reg*W # regularize the weights
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
