import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
print(tf.__version__)
import scipy.io as sio
import math

from gae.layers import GraphConvolution

def multihead_attention_gcn(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    

    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.contrib.layers.fully_connected(queries, num_units) # (N, T_q, C)
        K = tf.contrib.layers.fully_connected(queries, num_units ) # (N, T_k, C)
        V = tf.contrib.layers.fully_connected(keys, num_units) # (N, T_k, C)
        
        Q1 = tf.reshape(Q,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
        K1 = tf.reshape(K,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))        
        V1 = tf.reshape(V,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
        
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q1, num_heads, axis = 2),axis =0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K1, num_heads, axis = 2),axis =0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V1, num_heads, axis = 2),axis =0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
       # 
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
          
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
  # 
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
        matt    = outputs
        # Dropouts
        outputs = tf.contrib.layers.dropout(outputs, keep_prob=dropout_rate, is_training=tf.convert_to_tensor(is_training))
        
        ki = ['1','2', '3', '4','5','6','7','8']
        for k in range(0,num_heads):
            adj = matt[k,:,:]
            gcnin = V_[k,:,:]
            scopex1 = ki[k]
            scopex2 = ki[k] + 'x'

            print(scopex2)
            a_t = (adj)
            idg = tf.where(tf.not_equal(a_t, 0))
            sparse = tf.SparseTensor(idg, tf.gather_nd(a_t, idg), a_t.get_shape())


            gcnout1 = GraphConvolution(input_dim=num_units/num_heads,
                                                output_dim=num_units/num_heads,
                                                adj=sparse,
                                                act=tf.nn.relu,
                                                dropout=1-dropout_rate,
                                                logging=False,
                                                scope = scopex1)(gcnin)

            gcnout2 = GraphConvolution(input_dim=num_units/num_heads,
                                                  output_dim=num_units/num_heads,
                                                  adj=sparse,
                                                  act=tf.nn.relu,
                                                  dropout=1-dropout_rate,
                                                  logging=False,
                                                  scope = scopex2)(gcnout1)

            
     
            
            if k == 0:
                gcnout = tf.expand_dims(gcnout2, axis = 0)

            else:
                temp_gcn = tf.expand_dims(gcnout2, axis = 0)
                gcnout = tf.concat([gcnout, temp_gcn],axis = 0)

        
        # Restore shape
        outputs = tf.concat(tf.split(gcnout, num_heads,axis = 0),axis =2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries              
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs, matt