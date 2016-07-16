import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.

    w: Filter weights of shape (F, C, HH, WW)
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    #used for cnn_relu_maxpool
    self.params['W1']=np.random.normal(0,weight_scale,(num_filters,C,filter_size,filter_size))
    self.params['b1']=np.zeros(num_filters)
    #used for affine
    self.params['W2']=np.random.normal(0,weight_scale,(num_filters*H/2*W/2,hidden_dim))
    self.params['b2']=np.zeros(hidden_dim)
    #used for classfication
    self.params['W3']=np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    self.params['b3']=np.zeros(num_classes)
    #used for nomarlbatch
    self.params['beta1']=np.zeros(num_filters)
    self.params['gamma1']=np.ones(num_filters)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    self.bn_params = [{'mode': 'train'} for i in xrange(2)]


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    for bn_param in self.bn_params:
        bn_param[mode] = mode



    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    beta1, gamma1 = self.params['beta1'], self.params['gamma1']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_relu_bn_pool_out, conv_relu_bn_pool_cache =conv_relu_bn_pool_forward(X, W1, b1,
         conv_param, pool_param, gamma1, beta1, self.bn_params[0])
    #conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    aff_relu_out, aff_relu_cache = affine_relu_forward(conv_relu_bn_pool_out, W2, b2)
    aff2_out, aff2_cache = affine_forward(aff_relu_out, W3, b3)
    scores=aff2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, gradW=softmax_loss(scores,y)

    loss+=0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2)+ np.sum(W3*W3))

    affine_dx, affine_dw, affine_db = affine_backward(gradW, aff2_cache)
    grads['W3'] = affine_dw + self.reg * W3
    grads['b3'] = affine_db

    affine_relu_dx, affine_relu_dw, affine_relu_db = affine_relu_backward(affine_dx, aff_relu_cache)
    grads['W2'] = affine_relu_dw + self.reg * W2
    grads['b2'] = affine_relu_db


    #dx, dw, db = conv_relu_pool_backward(affine_relu_dx, conv_relu_pool_cache)
    dx, dw, db ,dgamma ,dbeta =conv_relu_bn_pool_backward(affine_relu_dx, conv_relu_bn_pool_cache)
    grads['W1'] = dw + self.reg * W1
    grads['b1'] = db
    grads['gamma1']=dgamma
    grads['beta1']=dbeta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

class ConvNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims,hidden_dim=[384,192], input_dim=(3, 32, 32), num_classes=10,
               filter_size=3,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None,pool_period=5):

    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.params={}
    self.dtype=dtype
    self.pool_period=pool_period
    self.hidden_dim=hidden_dim
    C, H, W = input_dim

    #used for cnn_relu_bn_maxpool
    for i in range(len(hidden_dims)):
        keyw='W'+str(i+1)
        keyb='b'+str(i+1)
        keybeta='beta'+str(i+1)
        keygamma='gamma'+str(i+1)
        num_filters=hidden_dims[i]
        self.params[ keyw ]=np.random.normal(0,weight_scale,(num_filters,C,filter_size,filter_size))
        self.params[ keyb ]=np.zeros(num_filters)
        if self.use_batchnorm:
            self.params[ keybeta ]=np.zeros(num_filters)
            self.params[ keygamma ]=np.ones(num_filters)
        C=num_filters

    for i in range(len(hidden_dim)):
      #used for affine
      if(i==0):
        self.params['W'+str(self.num_layers+i)]=np.random.normal(0,weight_scale,
            (num_filters*H/(2**(len(hidden_dims)/pool_period))*W/(2**(len(hidden_dims)/pool_period)),hidden_dim[i]))
      else:
        self.params['W'+str(self.num_layers+i)]=np.random.normal(0,weight_scale,
        (hidden_dim[i-1],hidden_dim[i]))
      self.params['b'+str(self.num_layers+i)]=np.zeros(hidden_dim[i])


    #used for classfication
    self.params['W'+str(self.num_layers+len(hidden_dim))]=np.random.normal(0,weight_scale,(hidden_dim[len(hidden_dim)-1],num_classes))
    self.params['b'+str(self.num_layers+len(hidden_dim))]=np.zeros(num_classes)


    ########use_batchnorm#################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    hidden_dim=self.hidden_dim

    ######## dropout ###########
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode  


    ######## batchnorm ###########
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None

    current_input=X
    #layer_cache saves neuron values during forward pass
    layer_cache={}
    dropout_cache={}


    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    for i in range(self.num_layers-1):
      keya='W'+str(i+1)
      keyb='b'+str(i+1)
      keygamma='gamma'+str(i+1)
      keybeta='beta'+str(i+1)
      if not self.use_batchnorm:
        #Conv_relu_pool
        filter_size = self.params[keya].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        if((i+1)%self.pool_period==0):
            current_input, layer_cache[keya] = conv_relu_pool_forward(current_input, 
              self.params[keya], 
              self.params[keyb],conv_param,pool_param)
        else:
            current_input, layer_cache[keya] = conv_relu_forward(current_input, 
              self.params[keya], 
              self.params[keyb], conv_param)

      else:
        filter_size = self.params[keya].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        if((i+1)%self.pool_period==0):
            current_input, layer_cache[keya] = conv_relu_bn_pool_forward(current_input, self.params[keya], 
            self.params[keyb],conv_param, pool_param, self.params[ keygamma ],
            self.params[ keybeta ], self.bn_params[i])
        else:
            #x, w, b, conv_param, gamma, beta, bn_param
            current_input, layer_cache[keya] = conv_relu_bn_forward(current_input, self.params[keya], 
            self.params[keyb],conv_param, self.params[ keygamma ],
            self.params[ keybeta ], self.bn_params[i])

      if self.use_dropout:
        current_input,dropout_cache[i]= dropout_forward(current_input,self.dropout_param)

    for i in range(len(hidden_dim)):
      keya='W'+str(self.num_layers+i)
      keyb='b'+str(self.num_layers+i)
      current_input, layer_cache[keya] = affine_relu_forward(current_input,
           self.params[keya],
           self.params[keyb])

    aff2_out, aff2_cache = affine_forward(current_input,
         self.params['W'+str(self.num_layers+len(hidden_dim))],
         self.params['b'+str(self.num_layers+len(hidden_dim))])

    scores=aff2_out




    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    loss, gradW=softmax_loss(scores,y)


    indot=0
    for i in range(self.num_layers+1):
      indot+=np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
    loss+=0.5*self.reg*indot
    #loss+=0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2)+ np.sum(W3*W3))

    dx, affine_dw, affine_db = affine_backward(gradW, aff2_cache)
    grads['W'+str(self.num_layers+len(hidden_dim))] = affine_dw + self.reg * self.params['W'+str(self.num_layers+len(hidden_dim))]
    grads['b'+str(self.num_layers+len(hidden_dim))] = affine_db

    for i in reversed(range(len(hidden_dim))):
      keya='W'+str(self.num_layers+i)
      keyb='b'+str(self.num_layers+i)
      dx, affine_relu_dw, affine_relu_db = affine_relu_backward(dx, layer_cache[keya])
      grads[keya] = affine_relu_dw + self.reg * self.params[keya]
      grads[keyb] = affine_relu_db

    for i in reversed(range(self.num_layers-1)):
      if self.use_dropout:
        dx = dropout_backward(dx,dropout_cache[i])

      keya = 'W'+str(i+1)
      keyb = 'b'+str(i+1)
      if not self.use_batchnorm: 
        if((i+1)%self.pool_period==0):
            dx, grads[ keya ], grads[ keyb ]=conv_relu_pool_backward(dx, layer_cache[ keya ])
        else:
            dx, grads[ keya ], grads[ keyb ]=conv_relu_backward(dx, layer_cache[ keya ])
        grads[ keya ]+=self.params[ keya ]*self.reg
      else:
        keygamma='gamma'+str(i+1)
        keybeta='beta'+str(i+1)
        if((i+1)%self.pool_period==0):
            dx, grads[ keya ], grads[ keyb ], dgamma, dbeta = conv_relu_bn_pool_backward(dx, layer_cache[ keya ]) 
        else:
            dx, grads[ keya ], grads[ keyb ], dgamma, dbeta = conv_relu_bn_backward(dx, layer_cache[ keya ])
        grads[ keya ]+=self.params[ keya ]*self.reg
        grads[ keybeta ]=dbeta
        grads[ keygamma ]=dgamma

    return loss, grads

