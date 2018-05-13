from builtins import range
import numpy as np

# Utility function for performing convolutional filtering of a simple 3-D image:
def conv_simple(x,w,b,pad, stride):

    """
    Description:

    Inputs:
    1. x - Input image - (D,H,W)
    2. w - Convolutional filter to be applied - (D,HH,WW)
    3. b - Bias value - Scalar
    4. pad -
    5. stride -

    Outputs:
    1. imOut - Output image with the convolutional filter and bias applied

    """

    # Extract shapes:
    D, H, W = x.shape
    D, HH, WW = w.shape

    # Compute output dimensions:
    OH = int( (H - HH + 2*pad)/stride + 1 )
    OW = int( (W - WW + 2*pad)/stride + 1 )

    imOut = np.zeros((OH,OW))

    # Indices to keep track of output image
    ii = 0
    jj = 0

    # Zero the Pad the input image:
    xp = np.pad(x,((0,0), (pad,pad) , (pad, pad)) , 'constant')

    # Nested loops to perform convolution:
    for row in range(0,H, stride):

        jj=0

        for col in range(0, W, stride):

            imOut[ii, jj] = np.sum( xp[:, row: row + HH, col: col + WW]*w) + b

            jj +=1

        ii+=1


    return imOut



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k) - N is the number of training examples
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.
    # Why have multiple dimensions for each input example?? - I think this    #
    # is to illustrate the example of an RGB image which has dim (H,W,3)      #
    # Each image needs to be reshaped into a column vector.
    ###########################################################################
    #pass


    # Reshape the input x and perform matrix multiplication with the parameter matrix W:
    out = np.dot(x.reshape(x.shape[0], w.shape[0]),w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # pass:

    # 1. Gradient w.r.t weights in the current layer (dw)
    dw = np.dot(x.reshape(x.shape[0], w.shape[0]).T, dout)

    # 2. Gradient w.r.t bias in the current layer:
    db = np.sum(dout, axis=0)

    # 3. Gradient w.r.t the input values x:
    dx = np.dot(dout, w.T).reshape(x.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # pass

    # The NumPy broadcast operation prodcues output with the same dimension:
    out = np.maximum(0,x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # pass
    dx = dout

    # ReLU function only passes the gradient through the inputs that did activate it.
    dx[cache<=0] =0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D) - N is Number of data samples in minibatch and D is dimension of each sample.
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability - From the 2015 paper by Ioffe and Szegedy
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # pass

        # Step 1: Compute the mean and varaiance of each feature: x is an array of shape (N,D)- Thus compute along axis=0
        mu_b = np.mean(x, axis= 0)
        sigma_b  = np.var(x, axis=0)

        # Step 2: Normalize the incoming data :i.e x -> (N,D)
        xhat = (x - mu_b)/ (np.sqrt(sigma_b + eps))  # TO DO: Test and see if broadcasting will work here!

        # Step 3: Scale and shift the normalized data:
        out = xhat*gamma + beta

        # Step 4: Update the running mean and running variance values:
        running_mean = momentum * running_mean + (1 - momentum) * mu_b
        running_var = momentum * running_var + (1 - momentum) * sigma_b

        # Step 5: Store variables in cache:
        cache  = (x , xhat, mu_b , sigma_b , gamma, eps)



        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # pass:

        # In the test case use the running mean and variance to normalize the data:
        xhat = (x - running_mean)/np.sqrt(running_var + eps)


        # Scale and shift the data:
        out = xhat*gamma + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    x , xhat,  mu_b , sigma_b , gamma , eps = cache
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    # pass

    N,D = dout.shape

    # Step 1: Graident of loss w.r.t mean vector of minibatch: mu_b:
    dmu_b = np.sum(dout*(-1*gamma)*(1/np.sqrt(sigma_b + eps)) , axis =0)

    # print (" d mu_b shape is:")
    # print (dmu_b.shape)

    # Step 2: Gradient of loss w.r.t variance vector of minibatch : sigma_b:
    dsigma_b = np.sum(dout *(x - mu_b)*(-0.5*gamma)*np.power((sigma_b + eps), -1.5) , axis=0)

    # print (" d sigma_b shape is:")
    # print (dsigma_b.shape)

    # Step 3: Gradient of loss w.r.t inputs x: Use formula derived on paper:
    dx = dout*gamma*(1/np.sqrt(sigma_b + eps)) + (1/N)*dmu_b  + (2/N)*dsigma_b*(x - mu_b)

    # print (" d x shape is:")
    # print (dx.shape)

    # Step 4: Gradient of loss w.r.t gamma:
    dgamma = np.sum (dout*xhat , axis=0)

    # print (" d gamma shape is:")
    # print (dgamma.shape)

    # Step 5: Gradient of loss w.r.t beta:
    dbeta = np.sum (dout, axis = 0)

    # print (" d beta shape is:")
    # print (dbeta.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None

    x , xhat,  mu_b , sigma_b , gamma , eps = cache
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    #pass

    N,D = dout.shape

    # Gradient of loss w.r.t normalized input:
    dxhat = dout*gamma   #(N,D) x (D,)

    # Graident of loss w.r.t input vector x:
    dx = (1.0/N)*(np.power(sigma_b +eps , -0.5))* ( N*dxhat - np.sum(dxhat , axis=0) - xhat*np.sum( dxhat*xhat , axis=0) )


    # Step 4: Gradient of loss w.r.t gamma:
    dgamma = np.sum (dout*xhat , axis=0)

    # print (" d gamma shape is:")
    # print (dgamma.shape)

    # Step 5: Gradient of loss w.r.t beta:
    dbeta = np.sum (dout, axis = 0)

    # print (" d beta shape is:")
    # print (dbeta.shape)



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout. Inverted dropout is the trick where the weights are scaled by the inverse of p
    at training time itself so that no additional work is to be performed at test time

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # pass
        # Completed on 8th April 2018:

        # Generate the mask from a binomial distribution using the probability p and scale it with the inverse of p :
        mask = np.random.binomial(1,p, size = x.shape)/p

        # Multiply the input with the mask  to get the output:
        out = x*mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # pass
        # Completed on 8th April 2018:
        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    p = dropout_param['p']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # pass:
        # Completed on 8th April 2018:
        dx = dout/p

        dx[mask==0]=0

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Extract values from conv_param:
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Extract shapes:
    N, C, H, W = x.shape
    F, C , HH , WW = w.shape

    # Set the dimensions of output tensor:
    OH = int ( (H - HH + 2*pad)/stride + 1)
    OW = int ( (W - WW + 2*pad)/stride + 1)
    out = np.zeros((N,F,OH,OW))

    for n in range(0, N):

        for f in range(0,F):

            out[n,f, : , :] = conv_simple(x[n,:,:,:] , w[f,:,:] , b[f], pad , stride)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None


    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # pass

    # Extract cached variables:
    x,w,b,conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Extract shapes:
    N, C, H, W = x.shape
    F, C , HH , WW = w.shape
    N,F, OH, OW = dout.shape


    # Zero pad the input images x:
    xPad  = np.pad(x, ((0,0), (0,0), (pad,pad), (pad, pad) ), 'constant')

    # Initialize the gradient of loss function w.r.t x:
    dx = np.zeros((N,C,H + 2*pad , W + 2*pad))

    # Initialize the gradient w.r.t weights (filters):
    dw = np.zeros(w.shape)

    # Initialize the gradient w.r.t the bias vector:
    db = np.zeros(b.shape)

    # Gradient w.r.t x:
    for n in range(N):

        for f in range(F):

            # Gradient w.r.t to the biase vector
            db[f] = np.sum(dout[:,f, : , :])

            for row in range(0,H, stride):

                for col in range(0, W, stride):

                    # print (row/stride)

                    dx[n,:,row: row + HH,col: col + WW] += dout[n,f, int(row/stride), int(col/stride)]*w[f,:,:,:]  # (1, C, HH, WW) = (1,1).* (1, C, HH, WW)

                    dw[f, : , :, :] += dout[n,f, int(row/stride), int(col/stride)]*xPad[n,:,row: row + HH , col: col + WW]  # (1, C, HH, WW) = (1,1).* (1, C, HH, WW)

    # Delete the extra rows in dx:
    delRows = list(range(pad)) + list(range(H + pad, H + 2*pad,1))
    delCols = list(range(pad)) + list(range(W + pad, W + 2*pad,1))
    dx = np.delete(dx, delRows, axis=2)
    dx = np.delete(dx, delCols, axis =3)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """

    # Extract the shape of the input:
    N, C, H, W = x.shape

    # Extract the parameters:
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']

    # Compute the shape of the output:
    OH = int( (H - PH)/S) + 1
    OW = int( (W - PW)/S) + 1

    out = np.zeros((N,C, OH, OW))


    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # pass

    for n in range(N):

        for c in range(C):

            for row in range(0,H, S):

                for col in range(0, W, S):

                    out[n,c,int(row/S), int(col/S)] = np.max (x[n,c, row:row + PH , col:col + PW])



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    # Extract the cached variables:
    x, pool_param = cache

    # Extract the shape of the input images:
    N,C,H,W = x.shape

    # Extract the parameters:
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']

    # Initialize the gradient w.r.t x:
    dx = np.zeros(x.shape)

    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # pass

    for n in range(N):

        for c in range(C):

            for row in range(0,H, S):

                for col in range(0, W, S):

                    ind_max = np.unravel_index( np.argmax( x[n,c, row:row +PH , col: col + PW], axis=None), (1,1,PH, PW))

                    rowMax = ind_max[2]
                    colMax = ind_max[3]

                    dx[n,c, row + rowMax, col + colMax] += dout[n,c, int(row/S), int(col/S)]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass

    # Extract the dimensions of the input:
    N, C, H, W  = x.shape

    # Transpose the input to get the channels value first:
    x= x.transpose((0,2,3,1))

    # Reshape into (C, --)
    xVec = np.reshape(x, (-1,C))

    # Use the regular batch normalization layer now:
    out_batch_vec , cache = batchnorm_forward(xVec, gamma, beta, bn_param)

    # Reshape the output from regular batch normalization layer:
    out = np.reshape(out_batch_vec, (N,H,W,C))

    # Transpose the output tensor:
    out = out.transpose((0,3,1,2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # pass

    # Extract the dimensions of the input:
    N, C, H, W  = dout.shape

    # Transpose the input to get the channels value first:
    dout = dout.transpose((0,2,3,1))

    # Reshape into (C, --)
    doutVec = np.reshape(dout, (-1,C))

    # Implement the regular batch normalization back prop function:
    dxVec , dgamma , dbeta = batchnorm_backward_alt(doutVec, cache)

    # Reshape the gradient w.r.t x:
    dx = np.reshape(dxVec, (N,H,W,C))

    # Transpose:
    dx = dx.transpose((0,3,1,2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
