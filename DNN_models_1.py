#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# © 2022 Victor Molokwu <vcm1@hw.ac.uk>
# Distributed under terms of the MIT license.
# A module of functions for computing the losses during training

import os
import batch_loss
import tensorflow_probability as tfp
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU: -1

import tensorflow as tf
import numpy as np
import DNN_models as dnn


dt_type='float64'
#============================================================================================================================================
# A Custom Layer class for a fully connected deep neural network with residual connections. 
class RNN_Layer(tf.keras.layers.Layer):
    def __init__(self,depth=1,width=16,kernel_size=3,nstacks=None,network_type=None,res_par={},fluid_idx=None,layer_name='',gaussian_process=None,\
                 hlayer_act_func=tf.nn.swish,out_act_func=None,idx=1,batch_norm={'Use_Batch_Norm':False,'Before_Activation':True,'Momentum':0.99},dropout={'Add':False,'Dropout_Idx_Stack':[1,1,0],'Rate':0.05},attn={},**kwargs):
        super(RNN_Layer, self).__init__(name=layer_name,**kwargs)
        self.width=width
        self.res_par=res_par
        self.gpc=gaussian_process
        self.layer_name=layer_name
        self.nstacks=nstacks
        self.network_type=network_type
        self.hlayer_act_func=hlayer_act_func
        if self.nstacks is None:
            self.nstacks=width['No_Gens'][idx]
        if self.network_type is None:
            self.network_type=self.res_par['Network_Type']
        if self.hlayer_act_func is None:
            self.res_par['Activation_Func']=lambda x:x
            
        self.stack_filters=[i for i in dnn.sparse_pad_list(depth=depth,nstacks=self.nstacks) if i!=0]
        self.res_depth=list(range(tf.shape(self.stack_filters)[0]))
        self.ragged_depth=[list(range(self.stack_filters[i])) for i in self.res_depth]
        self.filters=[i for i in dnn.network_width_list(depth=depth,width=width['Bottom_Size'],ngens=self.nstacks,growth_rate=width['Growth_Rate'],growth_type=width['Growth_Type'],network_type='resn') if i!=0]
        self.res_filters=[[self.filters[i] for j in range(self.stack_filters[i])] for i in self.res_depth]
        self._kernel_initializer=tf.keras.initializers.get(self.res_par['Kernel_Init'])
        self._kernel_regularizer=tf.keras.regularizers.get(self.res_par['Kernel_Regu']['Hidden_Layer'])

        # RNN Architecture 
        self.ks=[[kernel_size for j in range(self.stack_filters[i])] for i in self.res_depth]
        self.strides=[[1 for j in range(self.stack_filters[i])] for i in self.res_depth]
        self.paddings=[['same' for j in range(self.stack_filters[i])] for i in self.res_depth]
        #self.zero_pad_layers=[tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1))) for i in range(depth)]
        self.dense_layers=[[self.make_dense_layer(filters=self.res_filters[i][j],padding=self.paddings[i][j],kernel_size=self.ks[i][j],\
                                                 strides=self.strides[i][j],encoder=True,network_type=self.network_type,\
                                                     gaussian_process=self.gpc,idx=i*self.stack_filters[i]+j,**self._get_common_kwargs_for_sublayer(),name=f'stack_{i}_{j}') for j in range(self.stack_filters[i])] for i in self.res_depth]
        self.activation_layers=[[tf.keras.layers.Activation(self.res_par['Activation_Func'],name='main_activation_layer'+str(i*self.stack_filters[i]+j+1)) for j in range(self.stack_filters[i])] for i in self.res_depth]
        self.output_layer=[self.make_output_layer(network_type=self.network_type,activation=out_act_func,gaussian_process=self.gpc,**self._get_common_kwargs_for_sublayer()) for i in range(1)]
        # Batch Normalization (if any)
        self.batch_norm=batch_norm
        if 'Use_Batch_Norm' in self.batch_norm:
            if self.batch_norm['Use_Batch_Norm']:
                self.batch_norm_layers=[[tf.keras.layers.BatchNormalization(axis=1, momentum=self.batch_norm['Momentum'], epsilon=1e-6, name='_batch_norm_'+str(i*self.stack_filters[i]+j+1)) for j in range(self.stack_filters[i])] for i in self.res_depth]

        # Linear projection layers for skip connections
        self.linproj_layers=[self.make_dense_layer(filters=self.res_filters[i][0],padding='valid',kernel_size=1,strides=1,\
                                                   encoder=True,network_type=self.network_type,gaussian_process=self.gpc,idx=i,\
                                                       **self._get_common_kwargs_for_sublayer(),name=f'skip_{i}') for i in self.res_depth]
        self.add_linproj_layers=[tf.keras.layers.Add() for i in self.res_depth]
        # Dropout
        self.dropout=dropout
        if self.dropout['Add']:
            if (tf.shape(self.dropout['Dropout_Idx_Stack'])[0]-tf.shape(self.stack_filters)[0])<1:
                add_idx=tf.math.abs(tf.shape(self.dropout['Dropout_Idx_Stack'])[0]-tf.shape(self.stack_filters)[0])
                self.dropout['Dropout_Idx_Stack']+=[1 for i in range(add_idx)]
        
            # Create the dropout layers
            self.dropout_layers=[tf.keras.layers.Dropout(self.res_par['Dropout']['Rate'], noise_shape=None, seed=None) for i in self.res_depth]
        
    def call(self,inputs,return_skip_conn=False,output_layer=False):
        skip_conn={}
        hlayer=inputs
        skip_conn[0]=hlayer
        for i in self.res_depth:
            skip_layer=skip_conn[i]
            for j in self.ragged_depth[i]:
                hlayer=self.dense_layers[i][j](hlayer)
                if j<self.ragged_depth[i][-1]:
                    if 'Use_Batch_Norm' in self.batch_norm and self.batch_norm['Use_Batch_Norm']:
                        if self.batch_norm['Before_Activation']:
                            hlayer=self.batch_norm_layers[i][j](hlayer)
                            hlayer=self.activation_layers[i][j](hlayer)
                        else:
                            hlayer=self.activation_layers[i][j](hlayer)                                 
                            hlayer=self.batch_norm_layers[i][j](hlayer)
                    else:
                        hlayer=self.activation_layers[i][j](hlayer)
                else:
                    if 'Use_Batch_Norm' in self.batch_norm and self.batch_norm['Use_Batch_Norm']:
                        if self.batch_norm['Before_Activation']:
                            hlayer=self.batch_norm_layers[i][j](hlayer)
                                
            # Add the skip connection to the output at end of each group
            # Pads only for CNN2D
            if self.network_type.upper() in ['CNN','CNN2D']:
                skip_layer=self.pad_skip_layer(skip_layer,hlayer)
            
            # Linear projection of the innermost dimension of skip connections if not same the final residual output
            if skip_layer.shape[-1]!=hlayer.shape[-1]:
                skip_layer=self.linproj_layers[i](skip_layer)
            
            # Add the Skip layer
            hlayer=self.add_linproj_layers[i]([hlayer,skip_layer])
            
            if i<self.res_depth[-1]:
                # Add activation layer
                hlayer=self.activation_layers[i][j](hlayer)
            
                if 'Use_Batch_Norm' in self.batch_norm and self.batch_norm['Use_Batch_Norm']:
                    if not self.batch_norm['Before_Activation']:
                        hlayer=self.batch_norm_layers[i][j](hlayer)
                                        
                # Apply dropout regularization -- this is usually after the activation
                if self.dropout['Add'] in [True,1] and self.dropout['Dropout_Idx_Stack']==1:
                    hlayer=self.dropout_layers[i](hlayer)   
            else:
            # Output layer or returns an unactivated last layer output
                if output_layer:
                    hlayer=self.activation_layers[i][j](hlayer)
                    hlayer=self.output_layer[0](hlayer)
            # Save skip layer values
            skip_conn[i+1]=hlayer
        
        outputs=hlayer
        if return_skip_conn:
            outputs=hlayer,skip_conn
        return outputs
        
    def make_dense_layer(self,padding='valid',kernel_size=3,filters=32,strides=1,activation=None,kernel_init=None,kernel_regu=None,idx='',gaussian_process=False,encoder=True,network_type='CNN2D',name=''):
        if network_type.upper() in ['CNN2D','CNN']:
            if encoder:
                dense_layer=tf.keras.layers.Conv2D(filters,kernel_size, strides=strides, padding=padding,data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=kernel_init,kernel_regularizer=kernel_regu,activation=activation, name=name+'_CNV2D_ENC_layer_'+str(idx+1))
            else:
                dense_layer=tf.keras.layers.Conv2DTranspose(filters,kernel_size, strides=strides, padding=padding,data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=kernel_init,kernel_regularizer=kernel_regu,activation=activation, name=name+'_CNV2D_DEC_layer_'+str(idx+1))
        else:
            dense_layer=tf.keras.layers.Dense(filters, activation=activation, kernel_initializer=kernel_init,kernel_regularizer=kernel_regu, name=name+'_DENSE_layer_'+str(idx+1))
        # Add a wrapper for gaussian process
        if gaussian_process:
            dense_layer=tfa.layers.SpectralNormalization(dense_layer,)
        return dense_layer 

    def make_output_layer(self,padding='valid',kernel_size=1,filters=1,strides=1,activation=None,kernel_init=None,kernel_regu=None,gaussian_process=False,no_classes=10,network_type='CNN2D',name=''):
        #if not gaussian_process:
        if network_type.upper() in ['CNN2D','CNN']:
            dense_layer=tf.keras.layers.Conv2D(filters,kernel_size, strides=strides, padding=padding,data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=kernel_init, kernel_regularizer=kernel_regu, activation=activation, name=name+'_CNV2D_output_layer_',dtype=self.dtype_policy.variable_dtype)
        else:
            dense_layer=tf.keras.layers.Dense(filters, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=kernel_regu, name=name+'_DENSE_output_layer_',dtype=self.dtype_policy.variable_dtype)
        # else:
        #     tfm.nlp.layers.RandomFeatureGaussianProcess(no_classes,gp_cov_momentum=1)
        return dense_layer
    
    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regu=self._kernel_regularizer)
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_init = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config())
        common_kwargs["kernel_init"] = kernel_init
        return common_kwargs
    def pad_skip_layer(self,skip_input,target):
        if np.prod(skip_input.shape[1:2])!=np.prod(target.shape[1:2]):
            # Pad the layer with offset starting left
            layer_diff=tf.math.abs(skip_input.shape[1]-target.shape[1])
            pad_no=int(layer_diff/2)
            if layer_diff%2==0:  # Even
                padding=tf.constant([[0, 0], [pad_no, pad_no], [pad_no, pad_no],[0, 0]])
            else:
                padding=tf.constant([[0, 0], [pad_no+1, pad_no], [pad_no+1, pad_no],[0, 0]])
            skip_pad=tf.pad(skip_input,padding,mode='CONSTANT', constant_values=0) 
        else:
            skip_pad=skip_input
        return skip_pad

