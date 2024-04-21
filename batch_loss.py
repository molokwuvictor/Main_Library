#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# © 2022 Victor Molokwu <vcm1@hw.ac.uk>
# Distributed under terms of the MIT license.
# A module of functions for computing the losses during training

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from matplotlib import pyplot
from pickle import load
import time
import find_root_chandrupatla as chp
dt_type='float32'

#================================================================================================================================
# A function that computes the regularized loss for the domain and boundary (PINN) using tensorflow second-order optimizers.
def loss_and_gradient_fun(model,BG):
    # x is a list of numpy arrays or list of tuples
    # y is a list of numpy arrays or list of tuples
    
    # Obbtain the shapes of all trainable parameters in the model
    # Also the stitch indices and partition indices to be used later for tf.dynamic_stitch and tf.dynamic_partition
    # converts the initial paramters to 1D 
    shapes=convert_1D(model).shapes
    n_tensors=convert_1D(model).n_tensors
    idx=convert_1D(model).idx
    part=convert_1D(model).part
    nT=3     # No of model output terms -- pressure, gas and condensate saturations
    if model.cfd_type['Type']=='PINN':
        batch_sse_grad_func=pinn_batch_sse_grad
    else:
        batch_sse_grad_func=nopinn_batch_sse_grad_1
    if model.cfd_type['Fluid_Type']=='dry-gas':
        output_keys=['p','sg']
    else:
        output_keys=['p','sg','so']
    
    # Update the parameters of the model
    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))
    
    no_bt=BG.__len__()
    def f(params_1d):
        # update the parameters in the model
        #assign_new_model_parameters(params_1d)  -- once the function is called by the lpfgs outer method
        assign_new_model_parameters(params_1d)
        
        # Create a zero scalar tensor for the initial losses and a 1D zeros_like tensor for the initial gradient--tf.stitch is to be used
        # The initial losses is the SSE and it includes a tuning regularization parameter
        sum_batch_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_dom_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_dbc_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_nbc_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_ibc_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_ic_sse=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_td_sse={key:tf.Variable(tf.zeros((),dtype=dt_type),trainable=False) for key in output_keys}
        
        train_vars_1d=tf.dynamic_stitch(idx,model.trainable_variables)
        sum_batch_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_dom_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_dbc_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_nbc_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_ibc_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_ic_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        sum_td_sse_grad=tf.zeros_like(train_vars_1d, dtype=dt_type)
        
        # Sum the error counts of each solution term in the batch
        sum_dom_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_dbc_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_nbc_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_ibc_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_ic_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        sum_td_error_count=tf.Variable(tf.zeros((),dtype=dt_type),trainable=False)
        
        # Resets the metrics
        for m in model.metrics:
            m.reset_states()
        
        for i in tf.range(0, no_bt):
            # Determine starting and ending slice indexes for the current batch
            # Get the input data for the batch
            # The return value using the nopinn_batch_sse_grad function is a list with indexing:
            # [0]: Weighted SSE loss list=[batch,DOM,DBC,NBC,IBC,IC,TD]
            # [1]: Weighted SSE gradient list=[batch,DOM,DBC,NBC,IBC,IC,TD]
            # [2]: Error count list=[batch,DOM,DBC,NBC,IBC,IC,TD]
            # [3]: Weighted MSE loss list=[batch,DOM,DBC,NBC,IBC,IC,TD]
            # [4]: Model's output
        
            data=BG.__getitem__(i)
            xi,yi=data
            batch_sse_grad=batch_sse_grad_func(model,xi,yi)
            sum_batch_sse.assign_add(batch_sse_grad[0][0])
            sum_dom_sse.assign_add(batch_sse_grad[0][1])
            sum_dbc_sse.assign_add(batch_sse_grad[0][2])
            sum_nbc_sse.assign_add(batch_sse_grad[0][3])
            sum_ibc_sse.assign_add(batch_sse_grad[0][4])
            sum_ic_sse.assign_add(batch_sse_grad[0][5])
            sum_td_sse['p'].assign_add(batch_sse_grad[0][6][0]);sum_td_sse['sg'].assign_add(batch_sse_grad[0][6][1]);sum_td_sse['so'].assign_add(batch_sse_grad[0][6][2]) 
            
            # Stitch (convert to 1D tensor) the gradients
            sum_batch_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][0])
            sum_dom_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][1])
            sum_dbc_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][2])
            sum_nbc_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][3])
            sum_ibc_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][4])
            sum_ic_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][5])
            sum_td_sse_grad+=tf.dynamic_stitch(idx,batch_sse_grad[1][6])
            
            sum_dom_error_count.assign_add(batch_sse_grad[2][1])
            sum_dbc_error_count.assign_add(batch_sse_grad[2][2])
            sum_nbc_error_count.assign_add(batch_sse_grad[2][3])
            sum_ibc_error_count.assign_add(batch_sse_grad[2][4])
            sum_ic_error_count.assign_add(batch_sse_grad[2][5])
            sum_td_error_count.assign_add(batch_sse_grad[2][6])
            
            ## Compute self-defined metrics--MSE
            model.dom_loss.update_state(batch_sse_grad[3][1])
            model.dbc_loss.update_state(batch_sse_grad[3][2])
            model.nbc_loss.update_state(batch_sse_grad[3][3])
            model.ibc_loss.update_state(batch_sse_grad[3][4])
            model.ic_loss.update_state(batch_sse_grad[3][5])
            model.td_loss['p'].update_state(batch_sse_grad[3][6][0])
            model.td_loss['sg'].update_state(batch_sse_grad[3][6][1])
            model.td_loss['so'].update_state(batch_sse_grad[3][6][2])
            model.total_loss.update_state(batch_sse_grad[3][0])
            
            # Update the losses and gradients
            tf.print("\r[INFO] starting batch {}/{}...total_loss: {:.4e} - dom_loss: {:.4e} - dbc_loss: {:.4e} - nbc_loss: {:.4e} - ibc_loss: {:.4e} - ic_loss: {:.4e} - td_loss(p): {:.4e} - td_loss(sg): {:.4e} - td_loss(so): {:.4e}"\
                  .format(i+1,no_bt,model.total_loss.result(),model.dom_loss.result(),model.dbc_loss.result(),model.nbc_loss.result(),model.ibc_loss.result(),model.ic_loss.result(),model.td_loss['p'].result(),model.td_loss['sg'].result(),model.td_loss['so'].result()),sep=" ",end="")
            
            # Could append batch losses if necessary
        
        # Check for zeros in the unique appearance sums--prevent division by zero
        sum_dom_error_count=zeros_to_ones(sum_dom_error_count)
        sum_dbc_error_count=zeros_to_ones(sum_dbc_error_count)
        sum_nbc_error_count=zeros_to_ones(sum_nbc_error_count)
        sum_ibc_error_count=zeros_to_ones(sum_ibc_error_count)
        sum_ic_error_count=zeros_to_ones(sum_ic_error_count)
        sum_td_error_count=zeros_to_ones(sum_td_error_count)
        
        # Compute the full batch Mean Squared Errors
        dom_mse=tf.math.divide(sum_dom_sse,sum_dom_error_count)
        dbc_mse=tf.math.divide(sum_dbc_sse,sum_dbc_error_count)
        nbc_mse=tf.math.divide(sum_nbc_sse,sum_nbc_error_count)
        ibc_mse=tf.math.divide(sum_ibc_sse,sum_ibc_error_count)
        ic_mse=tf.math.divide(sum_ic_sse,sum_ic_error_count)
        td_mse={'p':tf.math.divide(sum_td_sse['p'],sum_td_error_count),'sg':tf.math.divide(sum_td_sse['sg'],sum_td_error_count),'so':tf.math.divide(sum_td_sse['so'],sum_td_error_count)}
        
        batch_mse=dom_mse+dbc_mse+nbc_mse+ibc_mse+ic_mse+sum(td_mse.values())
        
        # Compute the full batch gradients. 
        dom_grad=tf.math.divide(sum_dom_sse_grad,sum_dom_error_count)
        dbc_grad=tf.math.divide(sum_dbc_sse_grad,sum_dbc_error_count)
        nbc_grad=tf.math.divide(sum_nbc_sse_grad,sum_nbc_error_count)
        ibc_grad=tf.math.divide(sum_ibc_sse_grad,sum_ibc_error_count)
        ic_grad=tf.math.divide(sum_ic_sse_grad,sum_ic_error_count)
        td_grad=tf.math.divide(sum_td_sse_grad,nT*sum_td_error_count)
        batch_grad=dom_grad+dbc_grad+nbc_grad+ibc_grad+ic_grad+td_grad
        
        # clips the gradient when above a certain norm

               
        #if tf.norm(batch_grad,ord='euclidean')>=model.cfd_type['Gradient_Norm']:
        #    batch_grad=tf.clip_by_norm(batch_grad, model.cfd_type['Gradient_Norm'])
        
        _grad=[batch_grad,dom_grad,dbc_grad,nbc_grad,ibc_grad,ic_grad,td_grad]
        
        f.iter.assign_add(1)
        # Create a losses and gradients list and append to history--tracking of gradients and losses
        _mse=[batch_mse,dom_mse,dbc_mse,nbc_mse,ibc_mse,ic_mse,td_mse['p'],td_mse['sg'],td_mse['so'],tf.norm(batch_grad,ord='euclidean')]
        f.hist_loss['loss_values'].append(_mse)
        f.hist_grads.append(_grad)
        f.hist_ck.append([no_bt,sum_dom_error_count,sum_dbc_error_count,sum_nbc_error_count,sum_ibc_error_count,sum_ic_error_count,sum_td_error_count])
        tf.print(" [INFO] Epoch--{} Iterations completed. total_mse: {:.4e} - Gradient_norm: {:.4e}\n".format(f.iter.numpy(),batch_mse.numpy(),tf.norm(batch_grad)), end="")
        
        # Call the validation function
        return batch_mse, batch_grad.numpy()                       #batch_grad.numpy() with scipy.optimize
    
    loss_keys=['loss','dom_loss','dbc_loss','nbc_loss','ibc_loss','ic_loss','td_loss(p)','td_loss(sg)','td_loss(so)','grad_norm']   
    # Create a iteration count variable; updates its counter anytime the f(params_1d) 
    f.iter=tf.Variable(0, trainable=False)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.hist_loss={'loss_keys':loss_keys,'loss_values':[]}
    f.hist_grads=[]
    f.hist_ck=[]
        
    return f

# A function that converts the model parameters, i.e., model.trainable variables, to a 1D Tensor.
def convert_1D(model):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to prepare required information first.
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)
    
    convert_1D.shapes=shapes
    convert_1D.n_tensors=n_tensors
    convert_1D.idx=idx
    convert_1D.part=part
    return convert_1D

# A function that converts the zeros in a tensor to ones.
#@tf.function
def zeros_to_ones(x):
    y=tf.where(tf.math.equal(x,tf.zeros_like(x)),tf.ones_like(x), x) 
    return y

# A function that analytically computes the derivative of a normalization function.
@tf.function(jit_compile=True)
def normfunc_derivative(model,stat_idx=0,compute=False):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Normalization function derivative: Linear scaling (a,b) = (b-a)/(xmax-xmin)
    #                           Nonnormalized function: z-score= 1/std
    def _lnk_linear_scaling():
        lin_scale_no_log=(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0])
        lin_scale_log=(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0])
        
        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)
   
    def _linear_scaling():
        return (model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0])
    
    def _z_score():
        return 1/model.ts[stat_idx,3]
    
    normfunc_der=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: tf.ones((),dtype=model.dtype))

    # Dropsout a derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    normfunc_der=tf.where(tf.logical_or(tf.math.is_nan(normfunc_der), tf.math.is_inf(normfunc_der)),tf.zeros_like(normfunc_der), normfunc_der)
    return normfunc_der

# A function that computes the derivative based on a two-point finite-difference scheme. 
@tf.function
def finite_difference_derivative(model,x=None, diff_type='central_difference',grid_spacing=0.01):
    # Compute the finite difference 
    def central_difference():
        return (tf.stack(model(x+grid_spacing),axis=0)-tf.stack(model(x-grid_spacing),axis=0))/(2*grid_spacing)
    def forward_difference():
        return (tf.stack(model(x+grid_spacing),axis=0)-tf.stack(model(x),axis=0))/grid_spacing

    def backward_difference():
        return (tf.stack(model(x),axis=0)-tf.stack(model(x-grid_spacing),axis=0))/grid_spacing    
    
    derivative=tf.cond(tf.math.equal(diff_type,'central_difference'),lambda: central_difference(),lambda: forward_difference())
    
    # Dropsout a finite-difference derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    derivative=tf.where(tf.logical_or(tf.math.is_nan(derivative), tf.math.is_inf(derivative)),tf.zeros_like(derivative), derivative)
    return derivative

# A function that normalizes a model pipeline dataset based on training statistics. 
@tf.function(jit_compile=True)
def normalize(model,nonorm_input,stat_idx=0,compute=False):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Nonnormalized function: Linear scaling (a,b)= (xmax-xmin)*((x_norm-a)/(b-a))+xmin
    #                           Nonnormalized function: z-score= (x_norm*xstd)+xmean
    nonorm_input=tf.convert_to_tensor(nonorm_input, dtype=model.dtype, name='nonorm_input')
    
    def _lnk_linear_scaling():
        lin_scale_no_log=(((nonorm_input-model.ts[stat_idx,0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]))*(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.cfd_type['Norm_Limits'][0]
        lin_scale_log=((tf.math.log(nonorm_input/model.ts[stat_idx,0])/tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0]))*(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.cfd_type['Norm_Limits'][0]
        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)

    def _linear_scaling():
        return (((nonorm_input-model.ts[stat_idx,0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]))*(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.cfd_type['Norm_Limits'][0]
    
    def _z_score():
        return ((nonorm_input-model.ts[stat_idx,2])/model.ts[stat_idx,3])
    
    #norm=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: nonorm_input)
    norm_func=tf.where(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),_lnk_linear_scaling(),_linear_scaling())
    norm=tf.where(tf.math.equal(compute,True),norm_func,nonorm_input)
    
    # Dropsout a normalized value in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm

# A function that unnormalizes a normalized model pipeline dataset based based on the training statistics. 
@tf.function
def nonormalize(model,norm_input,stat_idx=0,compute=False):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Nonnormalized function: Linear scaling (a,b)= (xmax-xmin)*((x_norm-a)/(b-a))+xmin
    #                           Nonnormalized function: z-score= (x_norm*xstd)+xmean
    def _lnk_linear_scaling():
        lin_scale_no_log=(model.ts[stat_idx,1]-model.ts[stat_idx,0])*((norm_input-model.cfd_type['Norm_Limits'][0])/(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.ts[stat_idx,0]
        lin_scale_log=tf.math.exp(tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0])*((norm_input-model.cfd_type['Norm_Limits'][0])/(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+tf.math.log(model.ts[stat_idx,0]))

        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)
    def _linear_scaling():
        return (model.ts[stat_idx,1]-model.ts[stat_idx,0])*((norm_input-model.cfd_type['Norm_Limits'][0])/(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.ts[stat_idx,0]
    
    def _z_score():
        return (norm_input*model.ts[stat_idx,3])+model.ts[stat_idx,2]
    
    nonorm_func=tf.where(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),_lnk_linear_scaling(),_linear_scaling())
    nonorm=tf.where(tf.math.equal(compute,True),nonorm_func,norm_input)
    
    # Dropsout a unnormalized value in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    nonorm=tf.where(tf.logical_or(tf.math.is_nan(nonorm), tf.math.is_inf(nonorm)),tf.zeros_like(nonorm), nonorm)
    return nonorm

# A function that computes the normalized difference, i.e., normdiff=norm(a)-norm(b).
@tf.function(jit_compile=True)
def normalize_diff(model,diff,stat_idx=0,compute=False,x0=3.):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Nonnormalized function: Linear scaling (a,b)= (xmax-xmin)*((x_norm-a)/(b-a))+xmin
    #                           Nonnormalized function: z-score= (x_norm*xstd)+xmean
    diff=tf.convert_to_tensor(diff, dtype=model.dtype, name='diff')
    
    def _lnk_linear_scaling():
        lin_scale_no_log=tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]),dtype=model.dtype)*diff
        lin_scale_log=tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0]),dtype=model.dtype)*tf.math.log((x0+diff)/x0)

        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)

    def _linear_scaling():
        return tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]),dtype=model.dtype)*diff
    
    def _z_score():
        return tf.convert_to_tensor(1/model.ts[stat_idx,3],dtype=model.dtype)*diff
    
    norm_func=tf.where(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),_lnk_linear_scaling(),_linear_scaling())
    norm=tf.where(tf.math.equal(compute,True),norm_func,diff)
    
    # Dropsout a normalized difference in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm
 
# A function that makes an intermediate layer watchable by a tf.gradient tape.  
def watch_layer(layer, tape):
    """
    Make an intermediate hidden `layer` watchable by the `tape`.
    After calling this function, you can obtain the gradient with
    respect to the output of the `layer` by calling:

        grads = tape.gradient(..., layer.result)

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result of `layer.call` internally.
            layer.result = func(*args, **kwargs)
            # From this point onwards, watch this tensor.
            tape.watch(layer.result)
            # Return the result to continue with the forward pass.
            return layer.result
        return wrapper
    layer.call = decorator(layer.call)
    return layer
  

# A function that computes the second-order derivative using gradient tape.
# This operation introduces significant overhead without accelerated linear algebra. 
@tf.function#(jit_compile=True)
def second_order_derivative_AD(model,x,y):
    #Use Gradient Tape
    with tf.GradientTape(persistent=True) as tape2:
        # watch the input variables
        tape2.watch(x[3])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x[3])
            p = model(x, training=True) # Forward pass
        dp_dtn=tape1.gradient(p[0], x[3], unconnected_gradients='zero')    
        del tape1
    # Compute the second order derivatives.
    # 'experimental use pfor' set to False to prevent parallel tf tracing
    d2p_dtn2=tape2.gradient(dp_dtn, x[3], unconnected_gradients='zero')    
    del tape2
    dtn_dt=normfunc_derivative(model,stat_idx=3,compute=True)
    d2p_dt2=d2p_dtn2*(dtn_dt)**2
    return p,d2p_dt2
 
# A function that shifts model pipeline data points.
# This can be used during the physics-based semi-supervised learning to shift time points to unseen time domains. 
def time_shifting(model,xi,shift_frac_mean=0.2,pred_cycle_mean=1.,stat_idx=3, random=False):
    xp=list(xi)
    # Randomly shifts all time steps positions to improve learning
    #rnd_tshift=tf.reduce_mean(tf.random.normal(tf.shape(xp[3]),mean=5.,stddev=5/4,dtype=model.dtype,seed=None),axis=[1,2,3],keepdims=True)
    #rnd_tshift=normalize_diff(model,rnd_tshift,stat_idx=3,compute=True)
    #xp[3]+=(tf.cast((xp[3]>10.),model.dtype)*rnd_tshift)

    # Shifts the end time step towards prediction
    pred_cycle=pred_cycle_mean
    #pred_cycle=tf.math.abs(tf.random.normal((),mean=pred_cycle_mean,stddev=pred_cycle_mean/5,dtype=tf.dtypes.float32,seed=model.cfd_type['Seed']))
    
    shift_frac=shift_frac_mean; np=4
    #shift_frac=tf.math.abs(tf.random.stateless_normal((),[model.cfd_type['Seed']]*2,mean=shift_fac,stddev=(0.33*shift_fac),dtype=model.dtype,alg='auto_select')); np=4
    t=nonormalize(model,xp[stat_idx],stat_idx=stat_idx,compute=True)
    tsf_0=(1-shift_frac)*model.cfd_type['Max_Train_Time']

    tshift_range=(((pred_cycle+shift_frac)*model.cfd_type['Max_Train_Time']))
    tshift=(((t-tsf_0)/(model.cfd_type['Max_Train_Time']-tsf_0))*(tshift_range))+tsf_0-t 
    tshift_fac=((pred_cycle+shift_frac)/(shift_frac))
    
    tsf_0_norm=normalize(model,tsf_0,stat_idx=stat_idx,compute=True)  #.995
    tpred_norm=normalize_diff(model,tshift,stat_idx=stat_idx,compute=True)
    xp_idx=(tf.cast(xp[stat_idx]<tsf_0_norm,model.dtype)*xp[stat_idx])+(tf.cast(xp[stat_idx]>=tsf_0_norm,model.dtype)*(xp[stat_idx]+tpred_norm))
    xp[stat_idx]+=-xp[stat_idx]+xp_idx
    return xp,tshift_fac,tsf_0_norm
          
# A function that computes the physics-based errors of a model pipeline dataset. 
#@tf.function()  #--Taking a  derivative from outside exposes the TensorArray to the boundary, and the conversion is not implemented in Tensorflow.
def physics_error_gas_2D(model,x,y):
    # 1D model adapted to 2D for fast computiation on graph mode
    with tf.device('/GPU:0'):                   # GPU is better if available
        dt_type=model.dtype
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        paddings = tf.constant([[0,0], [1, 1,], [1, 1],[0, 0]])
        paddings_dPVT=tf.constant([[0,0], [0,0], [1, 1,], [1, 1],[0, 0]])

        phi=tf.pad(nonormalize(model,x[4],stat_idx=4,compute=compute_),paddings,mode='SYMMETRIC')     
        # Add noise to the permeability
        # x[5]=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.1*normalize_diff(model,x[5],stat_idx=5,compute=True), dtype=dt_type)

        kx=tf.pad(nonormalize(model,x[5],stat_idx=5,compute=compute_),paddings,mode='SYMMETRIC') 
        ky=kx
        # kz=tf.pad(nonormalize(model,x[6],stat_idx=6,compute=compute_),paddings,mode='SYMMETRIC')     
         
        dx=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][0]
        dy=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][1]
        dz=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][2]

        dx_ij=dx[...,1:-1,1:-1,:]; dx_i1=dx[...,1:-1,2:,:]; dx_i_1=dx[...,1:-1,:-2,:]
        dy_ij=dy[...,1:-1,1:-1,:]; dy_j1=dy[...,2:,1:-1,:]; dy_j_1=dy[...,:-2,1:-1,:] 
        dz_ij=dz[...,1:-1,1:-1,:] 
        dv=(dx_ij*dy_ij*dz_ij)
        dx_avg_ih=(dx_i1+dx_ij)/2.; dx_avg_i_h=(dx_ij+dx_i_1)/2.   
        dy_avg_jh=(dy_j1+dy_ij)/2.; dy_avg_j_h=(dy_ij+dy_j_1)/2. 

        C=tf.constant(0.001127, dtype=model.dtype, shape=(), name='const1')
        D=tf.constant(5.6145833334, dtype=model.dtype, shape=(), name='const2')
        eps=tf.constant(1e-7, dtype=model.dtype, shape=(), name='epsilon')    
        
        # Create the Connection Index Tensor for the wells
        # Cast a tensor of 1 for the well block and 0 for other cells &(q_n1_ij==model.cfd_type['Init_Grate']). The tensorflow scatter and update function can be used
        q_well_idx=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], tf.ones_like(model.cfd_type['Init_Grate']), model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        q_t0_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Init_Grate'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        min_bhp_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Min_BHP'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        
        no_wells=tf.cast(tf.shape(model.cfd_type['Init_Grate']),model.dtype)
        area_ij=dx_ij*dy_ij
        area_res=tf.cast(tf.math.reduce_prod(model.cfd_type['Dimension']['Measurement'][:2]),model.dtype)
        hc=tf.constant(model.cfd_type['Completion_Ratio'], dtype=dt_type, shape=(), name='completion_ratio')                              # Completion ratio
        
        # Static parameters
        phi_n1_ij=phi[...,1:-1,1:-1,:]  
        kx_ij=kx[...,1:-1,1:-1,:]; kx_i1=kx[...,1:-1,2:,:]; kx_i_1=kx[...,1:-1,:-2,:]
        ky_ij=ky[...,1:-1,1:-1,:]; ky_j1=ky[...,2:,1:-1,:]; ky_j_1=ky[...,:-2,1:-1,:]
        kx_avg_ih=(2.*kx_i1*kx_ij)/(kx_i1+kx_ij); kx_avg_i_h=(2.*kx_ij*kx_i_1)/(kx_ij+kx_i_1)
        ky_avg_jh=(2.*ky_j1*ky_ij)/(ky_j1+ky_ij); ky_avg_j_h=(2.*ky_ij*ky_j_1)/(ky_ij+ky_j_1)
        ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx_ij,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy_ij,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
        rw=tf.constant(0.1905/2, dtype=dt_type, shape=(), name='rw')
        model.phi_0_ij=phi_n1_ij
        model.cf_ij=97.32e-6/(1+55.8721*model.phi_0_ij**1.428586)
        Sgi=tf.constant((1-model.cfd_type['SCAL']['End_Points']['Swmin']),dtype=model.dtype,shape=(),name='Sgi')
        Soi=1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sgi
        tmax=model.cfd_type['Max_Train_Time']
        Pi=model.cfd_type['Pi']
        invBgi=model.cfd_type['Init_InvBg'];dinvBgi=model.cfd_type['Init_DinvBg'];
        invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
        def shut_days(limits=None,time=None,dtype=None):
            return (tf.ones_like(time)-tf.cast((time>=limits[0])&(time<=limits[1]),dtype)) 
        
        def wel_open(limits=None,time=None,dtype=None,eps=1.e-7):
            return tf.cast((time>=limits[1]-eps)&(time<=limits[1]+eps),dtype)
        
        Ck_ij=(2*(22/7)*hc*kx_ij*dz_ij*C)/(tf.math.log(ro/rw))
                        
        def physics_error_gas(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            out_stack_list=[0,1,2,3]
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            shutins_idx=tf.reduce_mean([tf.reduce_mean([shut_days(limits=model.cfd_type['Connection_Shutins']['Days'][c][cidx],time=tn0,dtype=model.dtype) for cidx in model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)
            welopens_idx=tf.reduce_mean([tf.reduce_mean([wel_open(limits=model.cfd_type['Connection_Shutins']['Days'][c][cidx],time=tn0,dtype=model.dtype) for cidx in model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)

            out_n0=model(xn0, training=True)
            out_n0,dPVT_n0,fac_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in out_stack_list]),[tf.pad(out_n0[i],paddings_dPVT,mode='SYMMETRIC') for i in [4,]],out_n0[-4:]

            # out_n0: predicted pressure output.
            # dPVT_n0: fluid property derivatives with respect to out_n0. 
            # predicted timestep at time point n0. 
            
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]
            
            invBg_n0_ij=out_n0[2][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBgug_n0_ij=(out_n0[2]*out_n0[3])[...,1:-1,1:-1,:]
            
            # Compute the average predicted timestep at time point n0.  
            tstep=(tf.reduce_mean(fac_n0[0],axis=[1,2,3],keepdims=True))
            
            # Normalize this timestep, as a difference, which is added to the nth time point to create the (n+1) prediction time point. 
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)

            # Create the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)

            # Re-evaluate the model time point n1. 
            out_n1,dPVT_n1,fac_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in out_stack_list]),[tf.pad(out_n1[i],paddings_dPVT,mode='SYMMETRIC') for i in [4,]],out_n1[-4:]
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]; 
            invBg_n1_ij=out_n1[2][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[3][...,1:-1,1:-1,:]
                       
            tstep_n1=tstep
            
            # Compute the average predicted timestep at time point n1.  
            tstep_n2=(tf.reduce_mean(fac_n1[0],axis=[1,2,3],keepdims=True))

            # Re-evaluate the model time point n2. 
            # However, the pressure and fluid properties at n2 are obtained by extrapolation
            p_n2_ij=(p_n1_ij-p_n0_ij)*(1.+tf.math.divide_no_nan(tstep_n2,tstep_n1))+p_n0_ij

            #=============================Relative Permeability Function=========================================================
            krog_n0,krgo_n0=krog_n1,krgo_n1=model.cfd_type['Kr_gas_oil'](Sgi)              #Entries: oil, and gas
            #====================================================================================================================
            #Define pressure variables 
            p_n1_i1=out_n1[0][...,1:-1,2:,:]; p_n1_i_1=out_n1[0][...,1:-1,:-2,:]
            p_n1_j1=out_n1[0][...,2:,1:-1,:]; p_n1_j_1=out_n1[0][...,:-2,1:-1,:]
            #====================================================================================================================
            # Compute d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision.
            d_dp_invBg_n0=dPVT_n0[0][0]
            invBgug_n1=(out_n1[2]*out_n1[3])
            invBgug_n1_ij=invBgug_n1[...,1:-1,1:-1,:]; 
            invBgug_n1_i1=invBgug_n1[...,1:-1,2:,:]; invBgug_n1_i_1=invBgug_n1[...,1:-1,:-2,:]
            invBgug_n1_j1=invBgug_n1[...,2:,1:-1,:]; invBgug_n1_j_1=invBgug_n1[...,:-2,1:-1,:]
            #====================================================================================================================
            # Compute the grid block pressures and fluid properties at faces using the average value function weighting
            p_n1_ih=(p_n1_i1+p_n1_ij)*0.5; p_n1_i_h=(p_n1_ij+p_n1_i_1)*0.5 
            p_n1_jh=(p_n1_j1+p_n1_ij)*0.5; p_n1_j_h=(p_n1_ij+p_n1_j_1)*0.5 
            p_n1_h=[p_n1_ih,p_n1_jh,p_n1_i_h,p_n1_j_h]           

            invBgug_avg_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_avg_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_avg_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_avg_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            cr_n0_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  #tf.zeros_like(phi)  
            cp_n1_ij=Sgi*((phi_n1_ij*d_dp_invBg_n0[...,1:-1,1:-1,:])+cr_n0_ij)

            a1_n1=C*kx_avg_i_h*krgo_n1*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(1/dx_ij)
            a2_n1=C*ky_avg_j_h*krgo_n1*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(1/dy_ij)
            a3_n1=C*kx_avg_ih*krgo_n1*invBgug_avg_n1_ih*(1/dx_avg_ih)*(1/dx_ij)
            a4_n1=C*ky_avg_jh*krgo_n1*invBgug_avg_n1_jh*(1/dy_avg_jh)*(1/dy_ij)
            a5_n1=(1/D)*(cp_n1_ij/(tstep))
            
            b1_n1=C*kx_avg_i_h*krgo_n1*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(dz_ij*dy_ij)
            b2_n1=C*ky_avg_j_h*krgo_n1*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(dz_ij*dx_ij)
            b3_n1=C*kx_avg_ih*krgo_n1*invBgug_avg_n1_ih*(1/dx_avg_ih)*(dz_ij*dy_ij)
            b4_n1=C*ky_avg_jh*krgo_n1*invBgug_avg_n1_jh*(1/dy_avg_jh)*(dz_ij*dx_ij)
            
            # Define grid weights
            well_wt=(tf.cast((q_well_idx==1),model.dtype)*1.)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            tsf_wt=tf.cast(xn0[3]<tsn['Time'],model.dtype)*1.+tf.cast(xn0[3]>=tsn['Time'],model.dtype)*tsn['Shift_Fac']
            # ===================================================================================================================
            # Compute bottom hole pressure
            q_n1_ij,_=fac_n1[-2],fac_n1[-1]
            
            # Compute the truncation Error term.
            trn_err=(dv/D)*cp_n1_ij*((2e-0*tf.keras.backend.epsilon()/tstep_n1)+(((tstep_n2*(p_n0_ij))+(tstep_n1*(p_n2_ij))-((tstep_n1+tstep_n2)*(p_n1_ij)))/((tstep_n1*tstep_n2)+tstep_n2**2.)))
            # ===================================================================================================================
            # Compute the domain loss.
            dom_divq_gg=dv*((-a1_n1*p_n1_i_1)+(-a2_n1*p_n1_j_1)+((a1_n1+a2_n1+a3_n1+a4_n1)*p_n1_ij)+(-a3_n1*p_n1_i1)+(-a4_n1*p_n1_j1)+(q_n1_ij/dv))
            dom_acc_gg=dv*a5_n1*(p_n1_ij-p_n0_ij)+trn_err
            dom=well_wt*(dom_divq_gg+dom_acc_gg)
            # Debugging....
            # tf.print('d_dp_invBg_n1\n',d_dp_invBg_n1,'InvBg_n0\n',invBg_n0,'InvBg_n1\n',invBg_n1,'InvUg_n1\n',invug_n1,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
            # tf.print('TIMESTEP\n',tf.reduce_mean(tstep,axis=[1,2,3]),output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/tstep.out" )
            #======================================= DBC Solution ===============================================================
            # Compute the external dirichlet boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            dbc=tf.zeros_like(dom)                 # Set at zero for now!
            #======================================= NBC Solution =============================================================== 
            # Compute the external Neumann boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            nbc=tf.zeros_like(dom)                 # Set at zero for now!
            #======================================= IBC Solution ===============================================================
            # Compute the inner boundary condition loss (wells).
            shutins_wt=4*(1.-shutins_idx)+1.
            ibc_n=q_well_idx*((dom_divq_gg))       
            #======================================= Material Balance Check =====================================================
            # Compute the material balance loss. 
            mbc=(-tf.reduce_sum(q_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)),axis=[1,2,3],keepdims=kdims))
            #======================================= Cumulative Material Balance Check ==========================================
            # Optional array: Compute the cumulative material balance loss (this loss is not considered - set as zero)
            cmbc=tf.zeros_like(dom) 
            #======================================= Initial Condition ==========================================================
            # Optional array: Compute the initial condition loss. This loss is set as zero since it is already hard-enforced in the neural network layers. 
            ic=tf.zeros_like(dom)

            #=======================================================================================================
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom) #q_well_idx*(q_n1_ij-out_t1[-2])          # Rate output index: -2
            qrc_3=tf.zeros_like(dom)#q_well_idx*(1e-8*-out_t0[-2])
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            #=============================================================================
            return [dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]]
        
        # A function to stack the physics-based losses.
        def stack_physics_error():
            # Perform time point shifting (if necessary). Not used.
            x_i,tshift_fac_i,tsf_0_norm_i=time_shifting(model,x,shift_frac_mean=0.05,pred_cycle_mean=0.,random=False)
            tstep_wt=tf.cast(x_i[3]<=tsf_0_norm_i,model.dtype)+tf.cast(x_i[3]>tsf_0_norm_i,model.dtype)*tshift_fac_i

            # Dry Gas
            out_gg=physics_error_gas(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})
            dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=out_gg[0],out_gg[1],out_gg[2],out_gg[3],out_gg[4],out_gg[5],out_gg[6],out_gg[7],out_gg[8],out_gg[9]
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  
            return [dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i],[out_n0_i,out_n1_i],no_grid_blocks
        
        phy_error,out_n,no_blks=stack_physics_error()
        stacked_pinn_errors=phy_error[0:-2]
        stacked_outs=out_n
        checks=[phy_error[-2],phy_error[-1]]

        return stacked_pinn_errors,stacked_outs,checks,no_blks
    

def physics_error_gas_oil_2D(model,x,y):
    # 1D model adapted to 2D for fast computiation on graph mode
    with tf.device('/GPU:0'):                   # GPU is better if available
        dt_type=model.dtype
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        
        paddings = tf.constant([[0,0], [1, 1,], [1, 1],[0, 0]])
        paddings_dPVT=tf.constant([[0,0], [0,0], [1, 1,], [1, 1],[0, 0]])
        phi=tf.pad(nonormalize(model,x[4],stat_idx=4,compute=compute_),paddings,mode='SYMMETRIC')     
        
        # Add noise to the permeability
        #x5=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.05, dtype=dt_type)

        kx=tf.pad(nonormalize(model,x[5],stat_idx=5,compute=compute_),paddings,mode='SYMMETRIC') 
        ky=kx
               
        dx=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][0]
        dy=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][1]
        dz=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][2]

        dx_ij=dx[...,1:-1,1:-1,:]; dx_i1=dx[...,1:-1,2:,:]; dx_i_1=dx[...,1:-1,:-2,:]
        dy_ij=dy[...,1:-1,1:-1,:]; dy_j1=dy[...,2:,1:-1,:]; dy_j_1=dy[...,:-2,1:-1,:] 
        dz_ij=dz[...,1:-1,1:-1,:] 
        dv=(dx_ij*dy_ij*dz_ij)
        dx_avg_ih=(dx_i1+dx_ij)/2.; dx_avg_i_h=(dx_ij+dx_i_1)/2.   
        dy_avg_jh=(dy_j1+dy_ij)/2.; dy_avg_j_h=(dy_ij+dy_j_1)/2. 

        C=tf.constant(0.001127, dtype=model.dtype, shape=(), name='const1')
        D=tf.constant(5.6145833334, dtype=model.dtype, shape=(), name='const2')
        eps=tf.constant(1e-7, dtype=model.dtype, shape=(), name='epsilon')    
        
        # Create the Connection Index Tensor for the wells
        # Cast a tensor of 1 for the well block and 0 for other cells &(q_n1_ij==model.cfd_type['Init_Grate']). The tensorflow scatter and update function can be used
        q_well_idx=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], tf.ones_like(model.cfd_type['Init_Grate']), model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        q_t0_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Init_Grate'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        min_bhp_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Min_BHP'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])

        no_wells=tf.cast(tf.shape(model.cfd_type['Init_Grate']),model.dtype)
        area_ij=dx_ij*dy_ij
        area_res=tf.cast(tf.math.reduce_prod(model.cfd_type['Dimension']['Measurement'][:2]),model.dtype)
        hc=tf.constant(model.cfd_type['Completion_Ratio'], dtype=dt_type, shape=(), name='completion_ratio')                              # Completion ratio

        # Static parameters
        phi_n1_ij=phi[...,1:-1,1:-1,:]  
        kx_ij=kx[...,1:-1,1:-1,:]; kx_i1=kx[...,1:-1,2:,:]; kx_i_1=kx[...,1:-1,:-2,:]
        ky_ij=ky[...,1:-1,1:-1,:]; ky_j1=ky[...,2:,1:-1,:]; ky_j_1=ky[...,:-2,1:-1,:]
        kx_avg_ih=(2.*kx_i1*kx_ij)/(kx_i1+kx_ij); kx_avg_i_h=(2.*kx_ij*kx_i_1)/(kx_ij+kx_i_1)
        ky_avg_jh=(2.*ky_j1*ky_ij)/(ky_j1+ky_ij); ky_avg_j_h=(2.*ky_ij*ky_j_1)/(ky_ij+ky_j_1)
        ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx_ij,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy_ij,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
        rw=tf.constant(0.1905/2, dtype=dt_type, shape=(), name='rw')
        model.phi_0_ij=phi_n1_ij
        model.cf_ij=97.32e-6/(1+55.8721*model.phi_0_ij**1.428586)
        Sgi=tf.constant((1-model.cfd_type['SCAL']['End_Points']['Swmin']),dtype=model.dtype,shape=(),name='Sgi')
        Soi=1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sgi
        Sor=model.cfd_type['SCAL']['End_Points']['Sorg']
        tmax=model.cfd_type['Max_Train_Time']
        Pi=model.cfd_type['Pi']
        Pdew=model.cfd_type['Dew_Point']
        invBgi=model.cfd_type['Init_InvBg'];dinvBgi=model.cfd_type['Init_DinvBg'];
        invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
        rhg_std=model.cfd_type['Rhg_Std']
        rho_std=model.cfd_type['Rho_Std']
        
        # Define an optimal rate and BHP function
        Ck_ij=(2*(22/7)*hc*kx_ij*dz_ij*C)/(tf.math.log(ro/rw))

        # Normalized constants
        t0_norm=normalize(model,0.,stat_idx=3,compute=True)
        t1_norm=normalize(model,1.,stat_idx=3,compute=True)
        tmax_norm=normalize(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
        tmax_norm_diff=normalize_diff(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
                        
        def physics_error_gas_oil(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            out_n0=model(xn0, training=True)
            out_n0,dPVT_n0,fac_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8,9]]),[tf.pad(out_n0[i],paddings_dPVT,mode='SYMMETRIC') for i in [10]],out_n0[-4:]

            # out_n0: predicted pressure output.
            # dPVT_n0: fluid property derivatives with respect to out_n0. 
            # predicted timestep at time point n0. 
            
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]           
            Sg_n0_ij=out_n0[1][...,1:-1,1:-1,:]
            So_n0_ij=out_n0[2][...,1:-1,1:-1,:]
            invBg_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBo_n0_ij=out_n0[4][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[5][...,1:-1,1:-1,:]
            invuo_n0_ij=out_n0[6][...,1:-1,1:-1,:]
            Rs_n0_ij=out_n0[7][...,1:-1,1:-1,:]
            Rv_n0_ij=out_n0[8][...,1:-1,1:-1,:]
            Vro_n0_ij=out_n0[9][...,1:-1,1:-1,:]
            invBgug_n0_ij=(out_n0[3]*out_n0[5])[...,1:-1,1:-1,:]
            invBouo_n0_ij=(out_n0[4]*out_n0[6])[...,1:-1,1:-1,:]
            RsinvBo_n0_ij=(out_n0[7]*out_n0[4])[...,1:-1,1:-1,:]
            RvinvBg_n0_ij=(out_n0[8]*out_n0[3])[...,1:-1,1:-1,:]
            
            invBgi,invBoi,invugi,invuoi,Rsi,Rvi=model.cfd_type['Init_InvBg'],model.cfd_type['Init_InvBo'],\
                model.cfd_type['Init_Invug'],model.cfd_type['Init_Invuo'],model.cfd_type['Init_Rs'],model.cfd_type['Init_Rv']
            invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
            invBoiuoi=model.cfd_type['Init_InvBo']*model.cfd_type['Init_Invuo']
            RsinvBoi=model.cfd_type['Init_Rs']*model.cfd_type['Init_InvBo']
            RvinvBgi=model.cfd_type['Init_Rv']*model.cfd_type['Init_InvBg']
            
            krog_n0,krgo_n0=model.cfd_type['Kr_gas_oil'](out_n0[1])              #Entries: oil, and gas |out_n1[1]; sat_n1[0]
            krgo_n0_ij=krgo_n0[...,1:-1,1:-1,:]
            krog_n0_ij=krog_n0[...,1:-1,1:-1,:]

            # Compute the average predicted timestep at time point n0.  
            tstep=(tf.reduce_mean(fac_n0[0],axis=[1,2,3],keepdims=True))
            
            # Normalize this timestep, as a difference, which is added to the nth time point to create the (n+1) prediction time point. 
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)

            # Create the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)

            # Re-evaluate the model time point n1. 
            out_n1,fac_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8,9]]),out_n1[-4:]           
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]
            Sg_n1_ij=out_n1[1][...,1:-1,1:-1,:]
            So_n1_ij=out_n1[2][...,1:-1,1:-1,:]
            invBg_n1_ij=out_n1[3][...,1:-1,1:-1,:]
            invBo_n1_ij=out_n1[4][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[5][...,1:-1,1:-1,:]
            invuo_n1_ij=out_n1[6][...,1:-1,1:-1,:]
            Rs_n1_ij=out_n1[7][...,1:-1,1:-1,:]
            Rv_n1_ij=out_n1[8][...,1:-1,1:-1,:]
            Vro_n1_ij=out_n1[9][...,1:-1,1:-1,:]
            invBgug_n1=(out_n1[3]*out_n1[5])
            invBouo_n1=(out_n1[4]*out_n1[6])
            RsinvBo_n1=(out_n1[7]*out_n1[4])
            RvinvBg_n1=(out_n1[8]*out_n1[3])
            RsinvBouo_n1=(out_n1[7]*out_n1[4]*out_n1[6])
            RvinvBgug_n1=(out_n1[8]*out_n1[3]*out_n1[5])
            invBgug_n1_ij=invBgug_n1[...,1:-1,1:-1,:]
            invBouo_n1_ij=invBouo_n1[...,1:-1,1:-1,:]
            RsinvBo_n1_ij=RsinvBo_n1[...,1:-1,1:-1,:]
            RvinvBg_n1_ij=RvinvBg_n1[...,1:-1,1:-1,:]   
            
            tstep_n1=tstep
            
            # Compute the average predicted timestep at time point n1.  
            tstep_n2=(tf.reduce_mean(fac_n1[0],axis=[1,2,3],keepdims=True))

            # Re-evaluate the model time point n2. 
            # However, for a two-phase system, the mass accumuated (instead of pressure) and fluid properties at n2 are obtained by extrapolation
            # The mass accumulated per grid block is given as: 
            wt_mt=0.5                                # applying equal weights between the two phases.
            mg_n0_ij=phi_n1_ij*((rhg_std*invBg_n0_ij*(1000/D)*Sg_n0_ij)+(rho_std*RvinvBg_n0_ij*Sg_n0_ij))    # reservoir mass flow: Conversion from bbl/Mscf to cf/scf=(5.615 ft/1000Scf)
            mo_n0_ij=phi_n1_ij*((rho_std*invBo_n0_ij*So_n0_ij)+(rhg_std*RsinvBo_n0_ij*(1000/D)*So_n0_ij))
            mt_n0_ij=mg_n0_ij+mo_n0_ij
            
            mg_n1_ij=phi_n1_ij*((rhg_std*invBg_n1_ij*(1000/D)*Sg_n1_ij)+(rho_std*RvinvBg_n1_ij*Sg_n1_ij))
            mo_n1_ij=phi_n1_ij*((rho_std*invBo_n1_ij*So_n1_ij)+(rhg_std*RsinvBo_n1_ij*(1000/D)*So_n1_ij))
            mt_n1_ij=mg_n1_ij+mo_n1_ij
            
            mt_n2_ij=(mt_n1_ij-mt_n0_ij)*(1.+tf.math.divide_no_nan(tstep_n2,tstep_n1))+mt_n0_ij

            # Compute the truncation error term.           
            trn_err=(dv/D)*((2e-0*tf.keras.backend.epsilon()/tstep_n1)+(((tstep_n2*(mt_n0_ij))+(tstep_n1*(mt_n2_ij))-((tstep_n1+tstep_n2)*(mt_n1_ij)))/((tstep_n1*tstep_n2)+tstep_n2**2.)))
            # ============================================ Relative Permeability ======================================
            krog_n1,krgo_n1=model.cfd_type['Kr_gas_oil'](out_n1[1])              #Entries: oil, and gas |out_n1[1]; sat_n1[0]
            krgo_n1_ij=krgo_n1[...,1:-1,1:-1,:]
            krog_n1_ij=krog_n1[...,1:-1,1:-1,:]
            # =========================================================================================================
            # Compute bottom hole pressure and rates
            # qfg_n1_ij,qdg_n1_ij,qfo_n1_ij,qvo_n1_ij,pwf_n1_ij=compute_rate_bhp_gas_oil(p_n0_ij,Sg_n0_ij,_invBg_n1_ij,_invBo_n1_ij,invBgug_n0_ij,invBouo_n0_ij,Rs_n0_ij,Rv_n0_ij,krgo_n0_ij,\
            #                                                     krog_n0_ij,q_t0_ij,min_bhp_ij,Ck_ij,q_well_idx,_Sgi=Sgi,_p_dew=Pdew,_shutins_idx=1,_ctrl_mode=1,\
            #                                                         _lmd=fac_n0[1],pre_sat_model=None,rel_perm_model=model.cfd_type['Kr_gas_oil'],model_PVT=model.PVT)
            qfg_n1_ij,qdg_n1_ij,qfo_n1_ij,qvo_n1_ij,pwf_n1_ij=fac_n1[-2][0],fac_n1[-2][1],fac_n1[-2][2],fac_n1[-2][3],fac_n1[-1]
            # =========================================================================================================
            # Compute the chord slopes for pressure and saturation. d_dp_Sg;d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision           
            _d_dpg_Sg_n1_ij=tf.math.divide_no_nan((out_n1[1]-out_n0[1]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            _d_dpo_So_n1_ij=tf.math.divide_no_nan((out_n1[2]-out_n0[2]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpg_Sg_n1_ij,d_dpo_So_n1_ij=_d_dpg_Sg_n1_ij,-1*_d_dpg_Sg_n1_ij            

            # Define a binary grid index indicating grid blocks above (0) or below (1) dew point/saturation pressure.
            tdew_idx=1.#tf.cast((pg_n1_ij<=model.cfd_type['Dew_Point']),model.dtype) #&(pg_n1_ij>=0.)
                        
            # Define gas and oil pressure variables for the grid blocks and those of adjoining grid blocks. 
            # In the absence of capillary pressures, the gas pressure is equal to the oil pressure, i.e., pg=po
            pg_n0_ij=out_n0[0][...,1:-1,1:-1,:]; po_n0_ij=out_n0[0][...,1:-1,1:-1,:];
            pg_n1_ij=out_n1[0][...,1:-1,1:-1,:]; po_n1_ij=out_n1[0][...,1:-1,1:-1,:];
            pg_n1_i1=out_n1[0][...,1:-1,2:,:]; po_n1_i1=out_n1[0][...,1:-1,2:,:];
            pg_n1_i_1=out_n1[0][...,1:-1,:-2,:]; po_n1_i_1=out_n1[0][...,1:-1,:-2,:];   
            pg_n1_j1=out_n1[0][...,2:,1:-1,:]; po_n1_j1=out_n1[0][...,2:,1:-1,:];
            pg_n1_j_1=out_n1[0][...,:-2,1:-1,:]; po_n1_j_1=out_n1[0][...,:-2,1:-1,:];
            
            # Define gas and oil fluid property and derivative variables of the grid blocks and those of adjoining grid blocks.
            invBgug_n1_i1=invBgug_n1[...,1:-1,2:,:]; invBgug_n1_i_1=invBgug_n1[...,1:-1,:-2,:]
            invBgug_n1_j1=invBgug_n1[...,2:,1:-1,:]; invBgug_n1_j_1=invBgug_n1[...,:-2,1:-1,:]
            
            invBouo_n1_ij=invBouo_n1[...,1:-1,1:-1,:]; 
            invBouo_n1_i1=invBouo_n1[...,1:-1,2:,:]; invBouo_n1_i_1=invBouo_n1[...,1:-1,:-2,:]
            invBouo_n1_j1=invBouo_n1[...,2:,1:-1,:]; invBouo_n1_j_1=invBouo_n1[...,:-2,1:-1,:]
            
            RvinvBgug_n1_ij=RvinvBgug_n1[...,1:-1,1:-1,:]; 
            RvinvBgug_n1_i1=RvinvBgug_n1[...,1:-1,2:,:]; RvinvBgug_n1_i_1=RvinvBgug_n1[...,1:-1,:-2,:]
            RvinvBgug_n1_j1=RvinvBgug_n1[...,2:,1:-1,:]; RvinvBgug_n1_j_1=RvinvBgug_n1[...,:-2,1:-1,:]

            RsinvBouo_n1_ij=RsinvBouo_n1[...,1:-1,1:-1,:]; 
            RsinvBouo_n1_i1=RsinvBouo_n1[...,1:-1,2:,:]; RsinvBouo_n1_i_1=RsinvBouo_n1[...,1:-1,:-2,:]
            RsinvBouo_n1_j1=RsinvBouo_n1[...,2:,1:-1,:]; RsinvBouo_n1_j_1=RsinvBouo_n1[...,:-2,1:-1,:]
            
            # Derivatives. 
            # d_dpg_invBg_n1_ij=tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_invBg_n1_ij=tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpg_invBo_n1_ij=tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_invBo_n1_ij=tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            # d_dpg_RsinvBo_n1_ij=tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_RsinvBo_n1_ij=tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpg_RvinvBg_n1_ij=tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_RvinvBg_n1_ij=tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            d_dpg_invBg_n1_ij=dPVT_n0[0][0][...,1:-1,1:-1,:]
            d_dpo_invBg_n1_ij=dPVT_n0[0][0][...,1:-1,1:-1,:]
            d_dpg_invBo_n1_ij=dPVT_n0[0][1][...,1:-1,1:-1,:]
            d_dpo_invBo_n1_ij=dPVT_n0[0][1][...,1:-1,1:-1,:]

            d_dpg_RsinvBo_n1_ij=((out_n0[7]*dPVT_n0[0][1])+(out_n0[4]*dPVT_n0[0][4]))[...,1:-1,1:-1,:] 
            d_dpo_RsinvBo_n1_ij=((out_n0[7]*dPVT_n0[0][1])+(out_n0[4]*dPVT_n0[0][4]))[...,1:-1,1:-1,:]
            d_dpg_RvinvBg_n1_ij=((out_n0[8]*dPVT_n0[0][0])+(out_n0[3]*dPVT_n0[0][5]))[...,1:-1,1:-1,:]
            d_dpo_RvinvBg_n1_ij=((out_n0[8]*dPVT_n0[0][0])+(out_n0[3]*dPVT_n0[0][5]))[...,1:-1,1:-1,:]
            
            # Compute the fluid property variables at the grid block faces using the average value function weighting.         
            invBgug_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            invBouo_n1_ih=(invBouo_n1_i1+invBouo_n1_ij)/2.; invBouo_n1_i_h=(invBouo_n1_ij+invBouo_n1_i_1)/2.
            invBouo_n1_jh=(invBouo_n1_j1+invBouo_n1_ij)/2.; invBouo_n1_j_h=(invBouo_n1_ij+invBouo_n1_j_1)/2.
            
            RvinvBgug_n1_ih=(RvinvBgug_n1_i1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_i_h=(RvinvBgug_n1_ij+RvinvBgug_n1_i_1)/2.
            RvinvBgug_n1_jh=(RvinvBgug_n1_j1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_j_h=(RvinvBgug_n1_ij+RvinvBgug_n1_j_1)/2.
            RsinvBouo_n1_ih=(RsinvBouo_n1_i1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_i_h=(RsinvBouo_n1_ij+RsinvBouo_n1_i_1)/2.
            RsinvBouo_n1_jh=(RsinvBouo_n1_j1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_j_h=(RsinvBouo_n1_ij+RsinvBouo_n1_j_1)/2.
             
            # Compute the gas and oil relative permeability variables at the grid block faces. 
            # The upstream weighting is suitable for saturation-dependent terms like relative permeability. 
            # Only the upstream weighting method works for linearization of saturation dependent terms in numerical simulations; the
            # average function value weighting gives erroneus results (Abou-Kassem, 2006).
            krgo_n1_i1=krgo_n1[...,1:-1,2:,:]; krgo_n1_i_1=krgo_n1[...,1:-1,:-2,:]
            krgo_n1_j1=krgo_n1[...,2:,1:-1,:]; krgo_n1_j_1=krgo_n1[...,:-2,1:-1,:]    
            
            krog_n1_i1=krog_n1[...,1:-1,2:,:]; krog_n1_i_1=krog_n1[...,1:-1,:-2,:]
            krog_n1_j1=krog_n1[...,2:,1:-1,:]; krog_n1_j_1=krog_n1[...,:-2,1:-1,:]
            
            #For i to be upstream (i+1), pot_i<=0; i to be downstream (i+1), pot_i>0.
            potg_n1_i1=poto_n1_i1=(pg_n1_i1-pg_n1_ij)
            potg_n1_i_1=poto_n1_i_1=(pg_n1_ij-pg_n1_i_1)
            potg_n1_j1=poto_n1_j1=(pg_n1_j1-pg_n1_ij)
            potg_n1_j_1=poto_n1_j_1=(pg_n1_ij-pg_n1_j_1)
            
            krgo_n1_ih=tf.cast(potg_n1_i1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_i1>0.,model.dtype)*krgo_n1_i1
            krgo_n1_i_h=tf.cast(potg_n1_i_1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_i_1>0.,model.dtype)*krgo_n1_i_1
            krgo_n1_jh=tf.cast(potg_n1_j1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_j1>0.,model.dtype)*krgo_n1_j1
            krgo_n1_j_h=tf.cast(potg_n1_j_1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_j_1>0.,model.dtype)*krgo_n1_j_1

            krog_n1_ih=tf.cast(poto_n1_i1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_i1>0.,model.dtype)*krog_n1_i1
            krog_n1_i_h=tf.cast(poto_n1_i_1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_i_1>0.,model.dtype)*krog_n1_i_1
            krog_n1_jh=tf.cast(poto_n1_j1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_j1>0.,model.dtype)*krog_n1_j1
            krog_n1_j_h=tf.cast(poto_n1_j_1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_j_1>0.,model.dtype)*krog_n1_j_1
            
            # krgo_n1_ih=krgo_n1_i_h=krgo_n1_jh=krgo_n1_j_h=krgo_n1_ij
            # krog_n1_ih=krog_n1_i_h=krog_n1_jh=krog_n1_j_h=krgo_n1_ij

            # Compute the rock compressibility term. This is the product of the rock compressibility, porosity and inverse formation volume factor at n0.
            cprgg_n0_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  
            cprgo_n0_ij=(model.phi_0_ij*model.cf*RsinvBo_n0_ij)  
            cproo_n0_ij=(model.phi_0_ij*model.cf*invBo_n0_ij)  
            cprog_n0_ij=(model.phi_0_ij*model.cf*RvinvBg_n0_ij)  
            
            # Compute variables for the gas phase flow -- free gas in the gas phase, and gas dissolved in the oil phase.
            agg_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*invBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            agg_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*invBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            ago_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*RsinvBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            ago_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*RsinvBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            agg_n1_ih=C*kx_avg_ih*(krgo_n1_ih*invBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            agg_n1_jh=C*ky_avg_jh*(krgo_n1_jh*invBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            ago_n1_ih=C*kx_avg_ih*(krog_n1_ih*RsinvBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            ago_n1_jh=C*ky_avg_jh*(krog_n1_jh*RsinvBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpgg_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_invBg_n1_ij)+cprgg_n0_ij))*(pg_n1_ij-pg_n0_ij)
            cpgo_n1_ij=(1/(D*tstep))*((phi_n1_ij*RsinvBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_RsinvBo_n1_ij)+cprgo_n0_ij))*(po_n1_ij-po_n0_ij)
            
            # Compute variables for the oil phase flow -- free oil in the oil phase, and oil vapourized in the gas phase.
            aoo_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*invBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aoo_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*invBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aog_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*RvinvBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aog_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*RvinvBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aoo_n1_ih=C*kx_avg_ih*(krog_n1_ih*invBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aoo_n1_jh=C*ky_avg_jh*(krog_n1_jh*invBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            aog_n1_ih=C*kx_avg_ih*(krgo_n1_ih*RvinvBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aog_n1_jh=C*ky_avg_jh*(krgo_n1_jh*RvinvBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpoo_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_invBo_n1_ij)+cproo_n0_ij))*(po_n1_ij-po_n0_ij)
            cpog_n1_ij=(1/(D*tstep))*((phi_n1_ij*RvinvBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_RvinvBg_n1_ij)+cprog_n0_ij))*(pg_n1_ij-pg_n0_ij)

            # Compute the domain loss. 
            # Domain divergence terms for the gas flow. 
            dom_divq_gg=dv*((-agg_n1_i_h*pg_n1_i_1)+(-agg_n1_j_h*pg_n1_j_1)+((agg_n1_i_h+agg_n1_j_h+agg_n1_ih+agg_n1_jh)*pg_n1_ij)+\
                      (-agg_n1_ih*pg_n1_i1)+(-agg_n1_jh*pg_n1_j1)+(qfg_n1_ij/dv))
                
            dom_divq_go=dv*((-ago_n1_i_h*po_n1_i_1)+(-ago_n1_j_h*po_n1_j_1)+((ago_n1_i_h+ago_n1_j_h+ago_n1_ih+ago_n1_jh)*po_n1_ij)+\
                      (-ago_n1_ih*po_n1_i1)+(-ago_n1_jh*po_n1_j1)+(qdg_n1_ij/dv))
                
            # Domain accumulation terms for the gas flow. 
            dom_acc_gg=dv*(cpgg_n1_ij)
            dom_acc_go=dv*(cpgo_n1_ij) 
            
            dom_gg=dom_divq_gg+dom_acc_gg
            dom_go=dom_divq_go+dom_acc_go
            
            # Domain loss for the gas flow. 
            dom_g=dom_gg+dom_go

            # Domain divergence terms for the oil flow. 
            dom_divq_oo=dv*((-aoo_n1_i_h*po_n1_i_1)+(-aoo_n1_j_h*po_n1_j_1)+((aoo_n1_i_h+aoo_n1_j_h+aoo_n1_ih+aoo_n1_jh)*po_n1_ij)+\
                      (-aoo_n1_ih*po_n1_i1)+(-aoo_n1_jh*po_n1_j1)+(qfo_n1_ij/dv))
                
            dom_divq_og=dv*((-aog_n1_i_h*pg_n1_i_1)+(-aog_n1_j_h*pg_n1_j_1)+((aog_n1_i_h+aog_n1_j_h+aog_n1_ih+aog_n1_jh)*pg_n1_ij)+\
                      (-aog_n1_ih*pg_n1_i1)+(-aog_n1_jh*pg_n1_j1)+(qvo_n1_ij/dv))
                
            # Domain accumulation terms for the oil flow. 
            dom_acc_oo=dv*(cpoo_n1_ij) 
            dom_acc_og=dv*(cpog_n1_ij)
            
            dom_oo=dom_divq_oo+dom_acc_oo
            dom_og=dom_divq_og+dom_acc_og
            
            # Domain loss for the oil flow. 
            dom_o=dom_oo+dom_og

            well_wt=(tf.cast((q_well_idx==1),model.dtype)*1.)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            # Debugging...
            #tf.print('d_dpg_invBg_n1_ij\n',d_dpg_invBg_n1_ij,'InvBg_n0\n',invBg_n0_ij,'InvBg_n1\n',invBg_n1_ij,'InvUg_n1\n',invug_n1_ij,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/pre_pvt.out" )

            cf=(5.615/1000.)
            
            # Weighting values for the gas and oil flows -- equal weights appied, i.e., 0.5
            wt_dom=0.5
            
            # Total domain loss for the gas and oil flows. 
            dom=wt_dom*(dom_gg+dom_go)+(1.-wt_dom)*(dom_oo+dom_og)+trn_err
            # ====================================== DBC Solution ===============================================================
            # Compute the external dirichlet boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            dbc=tf.zeros_like(dom)                  # Set at zero for now!
            # ====================================== NBC Solution =============================================================== 
            # Compute the external Neumann boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            nbc=tf.zeros_like(dom)                  # Set at zero for now!
            # ====================================== IBC Solution ===============================================================
            # Compute the inner boundary condition loss (wells).
            # regu=0.0005*tf.linalg.global_norm(model.trainable_variables)**2
            wt_ibc=0.5                              # Equal weights between the gas and oil flows. 
            ibc_n=q_well_idx*(wt_ibc*(dom_divq_gg+dom_divq_go)+(1.-wt_ibc)*(dom_divq_oo+dom_divq_og))
            # ====================================== Material Balance Check =====================================================
            # Compute the material balance loss. 
            kdims=False
            wt_mbc=0.5                              # Equal weights between the gas and oil flows. 
            mbc_gg=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*invBg_n1_ij)-(Sg_n0_ij*invBg_n0_ij))
            mbc_go=dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*RsinvBo_n1_ij)-(So_n0_ij*RsinvBo_n0_ij))
            
            mbc_oo=dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*invBo_n1_ij)-(So_n0_ij*invBo_n0_ij))
            mbc_og=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*RvinvBg_n1_ij)-(Sg_n0_ij*RvinvBg_n0_ij))
            
            mbc_g=(-tf.reduce_sum(qfg_n1_ij+qdg_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_gg+mbc_go,axis=[1,2,3],keepdims=kdims))             
            mbc_o=(-tf.reduce_sum(qfo_n1_ij+qvo_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_oo+mbc_og,axis=[1,2,3],keepdims=kdims))
            mbc=wt_mbc*(mbc_g)+(1.-wt_mbc)*(mbc_o)            
            # ====================================== Cumulative Material Balance Check ==========================================
            # Optional array: Compute the cumulative material balance loss (this loss is not considered - set as zero)
            cmbc=tf.zeros_like(dom) 
            # ======================================= Initial Condition =========================================================
            # Optional array: Compute the initial condition loss. This loss is set as zero since it is already hard-enforced in the neural network layers. 
            ic=tf.zeros_like(dom)
            # ====================================== Other Placeholders =========================================================
            # Supports up to four loss variables. Not currently used. 
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom)
            qrc_3=tf.zeros_like(dom)
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            # ===================================================================================================================
            return [dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]]

        # Stack the physics-based loss (if any)
        def stack_physics_error():
            x_i,tshift_fac_i,tsf_0_norm_i=time_shifting(model,x,shift_frac_mean=0.05,pred_cycle_mean=0.,random=False)
            tstep_wt=tf.cast(x_i[3]<=tsf_0_norm_i,model.dtype)+tf.cast(x_i[3]>tsf_0_norm_i,model.dtype)*tshift_fac_i

            # Gas-Oil
            out_go=physics_error_gas_oil(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})
            dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=out_go[0],out_go[1],out_go[2],out_go[3],out_go[4],out_go[5],out_go[6],out_go[7],out_go[8],out_go[9]
         
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  #update later
            return [dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i],[out_n0_i,out_n1_i],no_grid_blocks
        
        phy_error,out_n,no_blks=stack_physics_error()
        stacked_pinn_errors=phy_error[0:-2]
        stacked_outs=out_n 
        checks=[phy_error[-2],phy_error[-1]]

        return stacked_pinn_errors,stacked_outs,checks,no_blks
    
# A function that outputs zeros for the physics-based losses, preventing the demanding physics-based computations. Used during non-physics-based supervised learning. 
#@tf.function
def zeros_like_pinn_error(model,x,y):
    out_n0=model.loss_func['Squeeze_Out'](tf.stack(model(x, training=True)))
    dom=tf.zeros_like(y[0],dtype=dt_type)
    dbc=dom
    nbc=dom
    ibc_n=dom
    ic=dom
    mbc=dom
    cmbc=dom
    qrc_1=dom;qrc_2=dom;qrc_3=dom;qrc_4=dom
    out_n1=out_n0
    no_grid_blocks=[0.,0.,0.,0.,0.]
    qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
    return [dom,dbc,nbc,ibc_n,ic,qrc,],[out_n0,out_n1],[mbc,cmbc],no_grid_blocks
        
# ===============================================================================================================================
@tf.function
def boolean_mask_cond(x=None,y=None,data=[],bool_mask=[],solu_count=None):
    output=tf.cond(tf.math.equal(x,y),lambda: [tf.boolean_mask(data,bool_mask,axis=0),tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0))],\
                           lambda: [tf.multiply(tf.boolean_mask(data,bool_mask,axis=0),0.),tf.multiply(tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0)),0.)])
    return output

# ================================================= Training Loss Computation ===================================================
# A function that computes the training loss of either non-physics-based supervised learning or physics-based semi-supervised learning. 
# Tensorflow graph mode (@tf.function), also accelerated linear algebra XLA (jit_compile=True) is utilized to improve the speed of computation.
@tf.function(jit_compile=True)
def pinn_batch_sse_grad(model,x,y):
    # Physics gradient for Arrangement Type 1: 
    # DATA ARRANGEMENT FOR TYPE 1
    # Training Features: Model Inputs: 
    # x is a list of inputs (numpy arrays or tensors) for the given batch_size
    # x[0] is the grid block x_coord in ft.
    # x[1] is the grid block y_coord in ft.
    # x[2] is the grid block z_coord in ft.
    # x[3] is the time in day.
    # x[4] is the grid block porosity.
    # x[5] is the grid block x-permeability in mD.   
   
    # Model Outputs:
    # out[0] is the predicted grid block pressure (psia).
    # out[1] is the predicted grid block gas saturation.
    # out[2] is the predicted grid block oil saturation.
    
    # Training Labels: 
    # y[0] is the training label grid block pressure (psia).
    # y[1] is the training label block saturation--gas.
    # y[2] is the training label block saturation--oil.
    
    with tf.GradientTape(persistent=True) as tape3:
        pinn_errors,outs,checks,no_blks=model.loss_func['Physics_Error'](model,x,y) 
        dom_pinn=pinn_errors[0]
        dbc_pinn=pinn_errors[1]
        nbc_pinn=pinn_errors[2]
        ibc_pinn=pinn_errors[3]
        ic_pinn=pinn_errors[4]
        qrc_pinn=tf.stack(pinn_errors[5:])         
        mbc_pinn=checks[0]                          # MBC: Tank Material Balance Check.
        cmbc_pinn=checks[1]                         # CMBC: Cumulative Tank Material Balance Check.
        # =============================================== Training Data ============================================================
        # Training data includes the pressure and gas saturation labels.
        y_label=tf.stack([model.loss_func['Reshape'](y[i]) for i in model.nT_list],axis=0)
        y_model=outs[0][0:model.nT]
        td=(y_label-y_model)
       
        # Calculate the (Euclidean norm)**2 of each solution term--i.e., the error term.
        dom_pinn_se=tf.math.square(dom_pinn)                 
        dbc_pinn_se=tf.math.square(dbc_pinn)
        nbc_pinn_se=tf.math.square(nbc_pinn) 
        ibc_pinn_se=tf.math.square(ibc_pinn)
        ic_pinn_se=tf.math.square(ic_pinn) 
        qrc_pinn_se=tf.math.square(qrc_pinn)
        mbc_pinn_se=tf.math.square(mbc_pinn)
        cmbc_pinn_se=tf.math.square(cmbc_pinn)
        
        # Calculate the (Euclidean norm)**2 of the training data term.
        td_se=tf.math.square(td)
       
        # Compute the Sum of Squared Errors (SSE). 
        dom_pinn_sse=tf.math.reduce_sum(dom_pinn_se)
        dbc_pinn_sse=tf.math.reduce_sum(dbc_pinn_se)
        nbc_pinn_sse=tf.math.reduce_sum(nbc_pinn_se)
        ibc_pinn_sse=tf.math.reduce_sum(ibc_pinn_se)
        ic_pinn_sse=tf.math.reduce_sum(ic_pinn_se)
        qrc_pinn_sse=tf.math.reduce_sum(qrc_pinn_se)
        mbc_pinn_sse=tf.math.reduce_sum(mbc_pinn_se)
        cmbc_pinn_sse=tf.math.reduce_sum(cmbc_pinn_se)

        # Compute the Sum of Squared Errors (SSE) of the training data term.
        td_sse=tf.math.reduce_sum(td_se,axis=model.loss_func['Reduce_Axis'])
        
        # Weight the regularization term. 
        dom_wsse=model.nwt[0]*dom_pinn_sse
        dbc_wsse=model.nwt[1]*dbc_pinn_sse
        nbc_wsse=model.nwt[2]*(nbc_pinn_sse+tf.reduce_sum(qrc_pinn_sse)) #Rate check is averaged with the NBC Loss.              # nbc_avg_pinn_sse
        ibc_wsse=model.nwt[3]*ibc_pinn_sse
        ic_wsse=model.nwt[4]*ic_pinn_sse                      
        mbc_wsse=model.nwt[5]*mbc_pinn_sse
        cmbc_wsse=model.nwt[6]*cmbc_pinn_sse
        
        # Compute the weighted training loss of the batch.
        td_wsse=model.nwt[7:(7+model.nT)]*td_sse
        batch_wsse = dom_wsse+dbc_wsse+nbc_wsse+ibc_wsse+ic_wsse+mbc_wsse+cmbc_wsse+tf.reduce_sum(td_wsse)

        # Count the unique appearance of each loss term that does not have a zero identifier
        dom_error_count=tf.math.reduce_sum(tf.ones_like(dom_pinn_se))
        dbc_error_count=tf.math.reduce_sum(tf.ones_like(dbc_pinn_se))
        nbc_error_count=tf.math.reduce_sum(tf.ones_like(nbc_pinn_se))
        ibc_error_count=tf.math.reduce_sum(tf.ones_like(ibc_pinn_se))
        ic_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))
        mbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))                   # Tank Model 
        cmbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))  
        td_error_count=tf.math.reduce_sum(tf.ones_like(td_se[0]))
                
        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        dom_wmse=dom_wsse/zeros_to_ones(dom_error_count)
        dbc_wmse=dbc_wsse/zeros_to_ones(dbc_error_count)
        nbc_wmse=nbc_wsse/zeros_to_ones(nbc_error_count)
        ibc_wmse=ibc_wsse/zeros_to_ones(ibc_error_count)
        ic_wmse=ic_wsse/zeros_to_ones(ic_error_count)
        mbc_wmse=mbc_wsse/zeros_to_ones(mbc_error_count)
        cmbc_wmse=cmbc_wsse/zeros_to_ones(cmbc_error_count)
        td_wmse=td_wsse/zeros_to_ones(td_error_count)
        
        # tf.print('DOM_WMSE\n',dom_wsse,'\nDBC_WMSE\n',dbc_wsse,'\nNBC_WMSE\n',nbc_wsse,'\nIBC_WMSE\n',ibc_wsse,'\nIC_WMSE\n',ic_wsse,'\nMBC_WMSE\n',mbc_wsse,'\nCMBC_WMSE\n',cmbc_wsse,'\nTD_WMSE\n',td_wmse,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
        batch_wmse = dom_wmse+dbc_wmse+nbc_wmse+ibc_wmse+ic_wmse+mbc_wmse+cmbc_wmse+tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix.
             
    # Compute the gradients of each loss term.
    dom_wsse_grad=tape3.gradient(dom_wsse, model.trainable_variables,unconnected_gradients='zero')
    dbc_wsse_grad=tape3.gradient(dbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    nbc_wsse_grad=tape3.gradient(nbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ibc_wsse_grad=tape3.gradient(ibc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ic_wsse_grad=tape3.gradient(ic_wsse, model.trainable_variables,unconnected_gradients='zero')
    mbc_wsse_grad=tape3.gradient(mbc_wsse, model.trainable_variables,unconnected_gradients='zero')   
    cmbc_wsse_grad=tape3.gradient(cmbc_wsse, model.trainable_variables,unconnected_gradients='zero') 
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.
    
    # Compute the gradient of the batched data.
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    del tape3
    
    # Arrange the variables as a list. 
    _wsse=[batch_wsse,dom_wsse,dbc_wsse,nbc_wsse,ibc_wsse,ic_wsse,mbc_wsse,cmbc_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,dom_wsse_grad,dbc_wsse_grad,nbc_wsse_grad,ibc_wsse_grad,ic_wsse_grad,mbc_wsse_grad,cmbc_wsse_grad,td_wsse_grad]
    error_count=[1,dom_error_count,dbc_error_count,nbc_error_count,ibc_error_count,ic_error_count,mbc_error_count,cmbc_error_count,tf.reduce_sum(td_error_count)]

    _wmse=[batch_wmse,dom_wmse,dbc_wmse,nbc_wmse,ibc_wmse,ic_wmse,mbc_wmse,cmbc_wmse,td_wmse]  
    return [_wsse,_wsse_grad,error_count,_wmse,model.loss_func['Squeeze_Out'](tf.reshape(outs[0][0:model.nT,...],(model.nT,-1,*model.cfd_type['Dimension']['Dim'])))]


# A function that computes the training loss of fluid property supervised learning. 
@tf.function
def nopinn_batch_sse_grad_pvt(model,x,y):
    # DATA ARRANGEMENT FOR TYPE 3
    # Training Features: INPUTS: 
    # x is a list of inputs (numpy arrays or tensors) for the given batch_size
    # x[0] is the pressure

    # Training Label: OUTPUTS:
    # y[0] is the training label gas Formation Volume Factor inverse in 1/bbl/MScf
    # y[1] is the training label oil Formation Volume Factor inverse in 1/bbl/STB
    # y[2] is the training label gas viscosity inverse in 1/cp
    # y[3] is the training label oil viscosity inverse in 1/cp
    
    # Model OUTPUTS:
    # out[0] is the predicted grid block gas Formation Volume Factor inverse (1/Bg)
    # out[1] is the predicted grid block oil Formation Volume Factor inverse (1/Bo)
    # out[2] is the predicted grid block gas viscosity inverse (1/ug)
    # out[3] is the predicted grid block oil viscosity inverse (1/uo)
    
    with tf.GradientTape(persistent=True) as tape3:
        out = model(x, training=True) # Forward pass
        # ====================================================== Training Data ==================================================
        # Training data includes the porosity, constitutive relationship, k and Quantities of Interest (QoI) like pressure and saturation
        nT=tf.shape(out)[0]   
        y_label=tf.stack(y,axis=1)[:,0:nT]
        y_model=tf.squeeze(tf.stack(out,axis=1),[-1])
        td=y_label-y_model

        # Calculate the (Euclidean norm)**2 of the training term
        td_se=tf.math.square(td)

        # Compute the Sum of Squared Errors (SSE) of the Training term
        td_sse=tf.math.reduce_sum(td_se,axis=0)

        # Compute the training loss for the batch
        td_wsse=model.nwt[5:(5+nT)]*td_sse
        
        batch_wsse = tf.reduce_sum(td_wsse)

        # Compute the training error count
        td_error_count=tf.cast(tf.shape(td_se)[0],dtype=model.dtype)

        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        td_wmse=td_wsse/zeros_to_ones(td_error_count)
        
        batch_wmse = tf.reduce_sum(td_wmse)                
            
    # Compute the gradients of each loss term
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.
    
    #Compute the gradient of the batch
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    del tape3

    _wsse=[batch_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,td_wsse_grad]
    error_count=[1,(td_error_count)]
    
    _wmse=[batch_wmse,td_wmse]   
    
    return [_wsse,_wsse_grad,error_count,_wmse,out]

# ===============================================================================================================================

# A function that computes the total mobility cost function and its gradient. The gradient is obtained using the gradient tape to watch the computations. 
# This is used during the physics-based semi-supervised learning, for two-phase systems below dew point to estimate the saturation 
# at a bottomhole pressure, using the well grid block pressure and total mobility. 
def cost_func_Sg(_invBgug_n0_ij,_invBouo_n0_ij,_Rs_n0_ij,_Rv_n0_ij,Mo_Mg_n1_ij,_q_well_idx=None,rel_perm_model=None,_eps=1.0e-16):
    def a(Sg_n0_ij):
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tapeS:
            tapeS.watch(Sg_n0_ij)
            _krog_n0_ij,_krgo_n0_ij=rel_perm_model(Sg_n0_ij)
            _tmg_n0_ij=(_krgo_n0_ij*_invBgug_n0_ij)+(_krog_n0_ij*_invBouo_n0_ij)*_Rs_n0_ij
            _tmo_n0_ij=(_krog_n0_ij*_invBouo_n0_ij)+(_krgo_n0_ij*_invBgug_n0_ij)*_Rv_n0_ij
            cost=_q_well_idx*(tf.math.divide_no_nan(_tmo_n0_ij,_tmg_n0_ij+_eps)-Mo_Mg_n1_ij)
        d_Sg=tapeS.gradient(cost,Sg_n0_ij, unconnected_gradients='zero')
        del tapeS
        return cost,d_Sg   
    return a

# A function that computes the total mobility cost function without its gradient. 
def cost_func_Sg_nograd(_invBgug_n0_ij,_invBouo_n0_ij,_Rs_n0_ij,_Rv_n0_ij,Mo_Mg_n1_ij,_q_well_idx=None,rel_perm_model=None,_eps=1.0e-16,):
    def a(Sg_n0_ij):
        _krog_n0_ij,_krgo_n0_ij=rel_perm_model(Sg_n0_ij)
        _tmg_n0_ij=(_krgo_n0_ij*_invBgug_n0_ij)+(_krog_n0_ij*_invBouo_n0_ij)*_Rs_n0_ij
        _tmo_n0_ij=(_krog_n0_ij*_invBouo_n0_ij)+(_krgo_n0_ij*_invBgug_n0_ij)*_Rv_n0_ij
        cost=_q_well_idx*(tf.math.divide_no_nan(_tmo_n0_ij,_tmg_n0_ij+_eps)-Mo_Mg_n1_ij)
        return cost  
    return a

# A function that computes the relative permeability ratio, i.e., krg/kro, cost function without its gradient. 
def cost_func_Sg_kr_nograd(krg_kro_ij,_q_well_idx=None,rel_perm_model=None,_eps=1.0e-16,):
    def a(Sg_n0_ij):
        _krog_n0_ij,_krgo_n0_ij=rel_perm_model(Sg_n0_ij)
        cost=_q_well_idx*(tf.math.divide_no_nan(_krgo_n0_ij,_krog_n0_ij+_eps)-krg_kro_ij)
        return cost  
    return a

# A function that finds the zeros of a function utilizing the Netwon-Raphson and bisection method. 
def Newton_Raphson_Bisection(obj_func=None,no_iter=5,init_like=0.5,lower=0.,upper=1.,_back_prop=False):
    lower=(tf.ones_like(init_like))*lower
    upper=(tf.ones_like(init_like))*upper
    sign_lower=tf.sign(lower)
    sign_upper=tf.sign(upper)

    init_like=(lower+upper)*0.5
    i, x_init = tf.constant(0), tf.convert_to_tensor(init_like)
    c = lambda i,_: tf.less(i,no_iter)

    def b(i,x):
        f,df=tfp.math.value_and_gradient(lambda :obj_func(x[0]),x[0],use_gradient_tape=True)
        f_df=tf.math.divide_no_nan(f,df)
        result1=x[0]-f_df
        #print(obj_func(result)[1])
        c1=tf.cast(((result1<lower)|(result1>upper)),init_like.dtype)
        c1_1=c1*tf.cast(((tf.sign(obj_func(x[0]))*sign_lower)>=0.),init_like.dtype)
        c1_2=c1*tf.cast(((tf.sign(obj_func(x[0]))*sign_lower)<0.),init_like.dtype)
        c2=tf.cast(((result1>=lower)&(result1<=upper)),init_like.dtype)
        
        x[1]=x[0]*c1_1+(1.-c1_1)*x[1]
        x[2]=x[0]*c1_2+(1.-c1_2)*x[2]
        # lower=c3*lower+(1-c3)*0.5
        # upper=c3*upper+(1-c3)*0.5
        x[0]=c2*result1+c1*(x[1]+x[2])*0.5
        return i+1,x
    out=tf.while_loop(c, b, (i, [x_init,lower,upper]),maximum_iterations=no_iter,back_prop=_back_prop)[1][0]
    return out

# A function that computes the blocking intergral for two-phase well grid block flow computations.
def compute_integral_mo_mg(_p_n1_ij,_Sg_n1_ij,_boil_n1_ij,_bgas_n1_ij,_mo_mg_n1_ij,_tm_n1_ij,_krog_n1_ij,_min_bhp_ij,\
                  _p_dew=4048.,_p_maxl=3000.,_shift=0.001,_Sgi=0.78,_Sorg=0.2,_Socr=0.27,_q_well_idx=None,_no_intervals=5,_no_iter=10,pre_sat_model=None,rel_perm_model=None,model_PVT=None,_back_prop=False):
        _thf=0.95
        DM=tf.constant((5.615/1000), dtype=_p_n1_ij.dtype, shape=(), name='DM') #  Convert the surface oil mobility from stb/bbl to MScf/bbl
        #_no_intervals=2

        _p_ij=tf.linspace(_p_n1_ij,_min_bhp_ij,(_no_intervals+1))
        _p_ij=tf.ensure_shape(_p_ij,[None,*_p_n1_ij.shape])

        # A function that removes zeros (if any) from the tensor variables. 
        def remove_nan_end_corr(_p_n1_ij,_Sg_n1_ij,_bo_bg_n1_ij,_krog_n1_ij,_p_ord,_Sg_ord,_p_dew=4048.,_p_maxl=3000.,_Sgi=0.78,_Sorg=0.2,_Sg_CVD=None,_thf=_thf,):
            clip_var=lambda l,u:lambda x:tf.math.maximum(tf.math.minimum(x,u),l)
            _p_ord_n=tf.math.log(tf.nn.relu(tf.math.divide_no_nan((_p_n1_ij-_p_dew),(14.7-_p_dew)))+0.1)
            p1=-0.022006; p2=-0.112589; p3=-0.216498; p4=-0.200310; p5=-0.187842; p6=-0.194430; p7=0.1927090
            _Sg_CCE_n1_ij=lambda x:_Sgi-(1-0.22)**(1)*((p1*x**6)+(p2*x**5)+(p3*x**4)+(p4*x**3)+(p5*x**2)+(p6*x**1)+(p7*x**0))
            # p1=-8.9883E-21; p2=1.0141E-16; p3=-4.3145E-13; p4=8.4888E-10; p5=-7.6145E-07; p6=2.9704E-04; p7=1.6158E-01
            # _Sg_CCE_n1_ij=lambda x: _Sgi-((p1*x**6)+(p2*x**5)+(p3*x**4)+(p4*x**3)+(p5*x**2)+(p6*x**1)+(p7*x**0))
            _Sg_corr_ord=tf.where(tf.math.is_nan(_Sg_ord),_Sgi,_Sg_ord)
            
            #_Sg_corr_ord=tf.where(tf.math.greater_equal(_p_n1_ij,_p_dew),_Sgi,tf.where(tf.math.logical_or(tf.math.greater_equal(_p_n1_ij,0.99*_p_dew),tf.math.greater(_Sg_n1_ij,_Sgi-0.2)),(_Sg_n1_ij),_Sg_corr_ord))  
            _Sg_corr_ord=tf.where(tf.math.greater_equal(_p_n1_ij,_p_dew),_Sgi,tf.where(tf.math.logical_or(tf.math.greater_equal(_p_ord,0.99*_p_dew),tf.math.greater(_Sg_n1_ij,_Sgi-0.2)),tf.zeros_like(_Sg_n1_ij),(_Sg_corr_ord))) 
            return _Sg_corr_ord
        
        # Dew point
        # _krog_dew,_krgo_dew=rel_perm_model(_Sgi)
        # _p_dew=tf.constant(_p_dew,dtype=_p_ij.dtype,shape=(1,))
        # _PVT_dew=model_PVT(_p_dew)
        # _invBgug_dew=tf.reshape(_PVT_dew[0][0]*_PVT_dew[0][2],(1,)) 
        # _invBouo_dew=tf.reshape(_PVT_dew[0][1]*_PVT_dew[0][3],(1,))                
        # _Rs_dew=tf.reshape(_PVT_dew[0][4],(1,)) 
        # _Rv_dew=tf.reshape(_PVT_dew[0][5],(1,))
        # _mgg_dew=(_krgo_dew*_invBgug_dew)
        # _mgo_dew=0.#(_krog_dew*_invBouo_dew)*_Rs_dew
        # _moo_dew=0.#(_krog_dew*_invBouo_dew)
        # _mog_dew=(_krgo_dew*_invBgug_dew)*_Rv_dew
        # _tm_dew=_mgg_dew+_mgo_dew+_moo_dew+_mog_dew
        
        # Critical oil saturation at 0.95 dew point
        # _p_socr=_p_dew*_thf   # at maximum liquid dropout
        # _PVT_socr=model_PVT(_p_socr)
        # _invBgug_socr=tf.reshape(_PVT_socr[0][0]*_PVT_socr[0][2],(1,)) 
        # _invBouo_socr=tf.reshape(_PVT_socr[0][1]*_PVT_socr[0][3],(1,))                
        # _Rs_socr=tf.reshape(_PVT_socr[0][4],(1,)) 
        # _Rv_socr=tf.reshape(_PVT_socr[0][5],(1,))
        
        i, _sum, _tm_0 = tf.constant(0), tf.zeros_like(_p_n1_ij),_tm_n1_ij
        c = lambda i,*_: tf.less(i,_no_intervals)
        _bo_bg_n1_ij=tf.math.divide_no_nan(_boil_n1_ij,_bgas_n1_ij)

        def b(i,x):
            _p0=_p_ij[i]
            _p1=_p_ij[i+1]
            _PVT1=model_PVT(tf.reshape(_p1,(-1,)))
                      
            _invBg_1=tf.reshape(_PVT1[0][0],tf.shape(_p1)) 
            _invBo_1=tf.reshape(_PVT1[0][1],tf.shape(_p1)) 
            _invBgug_1=tf.reshape(_PVT1[0][0]*_PVT1[0][2],tf.shape(_p1))  
            _invBouo_1=tf.reshape(_PVT1[0][1]*_PVT1[0][3],tf.shape(_p1))                
            _Rs_1=tf.reshape(_PVT1[0][4],tf.shape(_p1))   
            _Rv_1=tf.reshape(_PVT1[0][5],tf.shape(_p1))
            
            #_VroCVD_1=tf.math.divide_no_nan((_invBg_1*(_bo_bg_n1_ij-_Rv_1)),(_invBo_1-(_Rv_1*_invBg_1)-_bo_bg_n1_ij*((_Rs_1*_invBo_1)-_invBg_1)))
            _VroCVD_1=tf.math.divide_no_nan((_boil_n1_ij-_bgas_n1_ij*_Rv_1),(1.-_Rv_1*_Rs_1))*(1./_invBo_1)
            _Sg_CVD=_Sgi*(1.-_VroCVD_1)        
                           
            _cf_1=cost_func_Sg_nograd(_invBgug_1,_invBouo_1,_Rs_1,_Rv_1,_mo_mg_n1_ij,_q_well_idx=_q_well_idx,rel_perm_model=rel_perm_model,_eps=0.)   
            #_cf_1=cost_func_Sg(_invBgug_1,_invBouo_1,_Rs_1,_Rv_1,_mo_mg_n1_ij,_q_well_idx=_q_well_idx,rel_perm_model=rel_perm_model,_eps=1.0e-7)

            _Sg_1=tf.ensure_shape(chp.find_root_chandrupatla(_cf_1,low=(tf.zeros_like(_p0)),high=(tf.ones_like(_p0)*(_Sgi)),position_tolerance=1e-08,value_tolerance=0.0,max_iterations=_no_iter,stopping_policy_fn=tf.reduce_all,validate_args=False,name='find_root_chandrupatla')[0],_p_n1_ij.get_shape())
            #_Sg_1=Newton_Raphson_Bisection(obj_func=_cf_1,no_iter=_no_iter,init_like=tf.zeros_like(_p0),lower=0.,upper=(_Sgi),_back_prop=False)  #tf.zeros_like(_p0)
            _Sg_1=remove_nan_end_corr(_p_n1_ij,_Sg_n1_ij,_bo_bg_n1_ij,_krog_n1_ij,_p1,_Sg_1,_p_dew=_p_dew,_p_maxl=_p_maxl,_Sgi=_Sgi,_Sorg=_Sorg,_Sg_CVD=_Sg_CVD)
            #tf.print('\n_Sg_1\n',_Sg_1[...,29,9,0],output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/_Sg_1.out" )
         
            _krog_1,_krgo_1=rel_perm_model(_Sg_1)
            _mgg_1=_krgo_1*_invBgug_1
            _mgo_1=_krog_1*_invBouo_1*_Rs_1
            _moo_1=_krog_1*_invBouo_1
            _mog_1=_krgo_1*_invBgug_1*_Rv_1
            _tm_1=_mgg_1+_mgo_1+(_moo_1+_mog_1)

            #_area_p0_dew_p1=(0.5*(x[1]+_tm_dew)*(_p0-_p_dew))+((0.5*(_tm_dew+_tm_1)*(_p_dew-_p1)))            
            _area=0.5*(x[1]+_tm_1)*(_p0-_p1)
            
            x[0]+=_area
            x[1]=_tm_1
            return i+1,x
        return tf.while_loop(c, b, (i, [_sum,_tm_0  ]),maximum_iterations=_no_intervals,back_prop=_back_prop)[1][0]   
            
# A function that computes the blocking intergral for single-phase low-pressure well grid block flow computations.      
def compute_integral_mg(_p_n1_ij,_invBgug_n1_ij,_min_bhp_ij,_no_intervals=5,model_PVT=None,_back_prop=False):    
    _p_ij=tf.linspace(_p_n1_ij,_min_bhp_ij,(_no_intervals+1))
    _p_ij=tf.ensure_shape(_p_ij,[None,*_p_n1_ij.shape])

    _tm_n1_ij=_invBgug_n1_ij

    i, _sum, _tm_0 = tf.constant(0), tf.zeros_like(_p_n1_ij),_tm_n1_ij
    c = lambda i,*_: tf.less(i,_no_intervals)
    def b(i,x):
        # sum = x[0]; tm = x[1]
        _p0=_p_ij[i]
        _p1=_p_ij[i+1]
        _PVT1=model_PVT(tf.reshape(_p1,(-1,)))
       
        _invBgug_1=tf.reshape(_PVT1[0][0]*_PVT1[0][1],tf.shape(_p1))  
        _tm_1=_invBgug_1

        _area=0.5*(x[1]+_tm_1)*(_p0-_p1)
        x[0]+=_area
        x[1]=_tm_1

        return i+1,x
    return tf.while_loop(c, b, (i, [_sum,_tm_0]),maximum_iterations=_no_intervals,back_prop=_back_prop)[1][0]    

# A function that computes the cost function and its gradient, in a two-phase system, using the well grid block rate. The rate is dependendent on the specified mobility phase m_opt_n1_ij.        
def cost_func_pwf(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=None,_m_opt_n1_ij=None,\
                   _p_dew=4048.,_shift=50.,_Sgi=0.78,_Sorg=0.2,_lmd=1.,_no_intervals=5,_no_iter=10,pre_sat_model=None,rel_perm_model=None,model_PVT=None,_back_prop=False):
    def a(pwf_n0_ij):
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape_pwf:
            tape_pwf.watch(pwf_n0_ij)
            _area_pwf_ij=compute_integral_mo_mg(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,pwf_n0_ij,\
                  _p_dew=_p_dew,_Sgi=_Sgi,_Sorg=_Sorg,_no_intervals=_no_intervals,_no_iter=_no_iter,_q_well_idx=_q_well_idx,pre_sat_model=None,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=_back_prop)
            _blk_fac=tf.math.divide_no_nan((_area_pwf_ij),(_tm_n1_ij*(_p_n1_ij-pwf_n0_ij))) #_lmd
            cost=(tf.ensure_shape(_q_well_idx*(_Ck_ij*(_blk_fac*_m_opt_n1_ij)*(_p_n1_ij-pwf_n0_ij)-_q_opt_n1_ij),_p_n1_ij.get_shape()))
        d_pwf=tape_pwf.gradient(cost,pwf_n0_ij, unconnected_gradients='zero')
        del tape_pwf
        return cost,d_pwf   
    return a

# A function that computes the cost function of the well grid block rate.    
def cost_func_pwf_NoGrad(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=None,_m_opt_n1_ij=None,\
                   _p_dew=4048.,_shift=50.,_Sgi=0.78,_Sorg=0.2,_lmd=1.,_no_intervals=12,_no_iter=4,pre_sat_model=None,rel_perm_model=None,model_PVT=None,_back_prop=False):
    def a(pwf_n0_ij):
        _area_pwf_ij=compute_integral_mo_mg(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,pwf_n0_ij,\
              _p_dew=_p_dew,_Sgi=_Sgi,_Sorg=_Sorg,_no_intervals=_no_intervals,_no_iter=_no_iter,_q_well_idx=_q_well_idx,pre_sat_model=None,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=_back_prop)
        _blk_fac=tf.math.divide_no_nan((_area_pwf_ij),(_tm_n1_ij*(_p_n1_ij-pwf_n0_ij))) #_lmd
        _q_pred_n1_ij=_q_well_idx*(_Ck_ij*(_blk_fac*_m_opt_n1_ij)*(_p_n1_ij-pwf_n0_ij))
        _q_obs_n1_ij=_q_well_idx*_q_opt_n1_ij
        cost=tf.ensure_shape((_q_pred_n1_ij-_q_obs_n1_ij),_p_n1_ij.get_shape())
        return cost  
    return a

# A function that iteratively estimates the well grid block bottomhole pressure, in a two-phase system, using a cost function. 
def compute_pwf(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,_min_bhp_ij,_q_well_idx,_Ck_ij,_pwf_n1_ij=None,_q_opt_n1_ij=None,_qmax_n1_ij_min_bhp=None,_m_opt_n1_ij=None,\
                   _p_dew=4048.,_shift=0.001,_pi=5000.,_Sgi=0.78,_Sorg=0.2,_lmd=1.,_no_intervals=12,_no_iter=[4,4],pre_sat_model=None,rel_perm_model=None,model_PVT=None,_back_prop=False):
    obj_func=cost_func_pwf_NoGrad(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=_q_opt_n1_ij,_m_opt_n1_ij=_m_opt_n1_ij,\
                       _p_dew=_p_dew,_shift=_shift,_Sgi=_Sgi,_Sorg=_Sorg,_lmd=_lmd,_no_intervals=_no_intervals,_no_iter=_no_iter[1],pre_sat_model=pre_sat_model,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=_back_prop)
    #_pwf_n0_ij=Newton_Raphson_ld(obj_func=obj_func,no_iter=_no_iter[0],init_like=tf.zeros_like(_p_n1_ij)+_pi,lower=_min_bhp_ij,upper=_p_n1_ij,ld=1,_back_prop=_back_prop)   #tf.zeros_like(_p_n1_ij)
    _pwf_n0_ij=tf.ensure_shape(chp.find_root_chandrupatla(obj_func,low=_min_bhp_ij,high=_p_n1_ij,position_tolerance=1e-08,value_tolerance=0.0,max_iterations=_no_iter[0],stopping_policy_fn=tf.reduce_all,validate_args=False,name='find_root_chandrupatla')[0],_p_n1_ij.get_shape())    
    return _pwf_n0_ij

# A function that computes the cost function, in a low-pressure dry gas system, using the well grid block rate. 
def cost_func_pwf_nograd_gas(_p_n1_ij,_invBgug_n1_ij,_kr_n1_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=None,\
                  _lmd=1.,_no_intervals=5,model_PVT=None,_back_prop=False):
    def a(pwf_n0_ij):
        _area_pwf_ij=compute_integral_mg(_p_n1_ij,_invBgug_n1_ij,pwf_n0_ij,_no_intervals=_no_intervals,model_PVT=model_PVT,_back_prop=_back_prop)
        _blk_fac=tf.math.divide_no_nan((_area_pwf_ij),(_invBgug_n1_ij*(_p_n1_ij-pwf_n0_ij)))
        cost=(tf.ensure_shape(_q_well_idx*(_Ck_ij*(_blk_fac*_kr_n1_ij*_invBgug_n1_ij)*(_p_n1_ij-pwf_n0_ij)-_q_opt_n1_ij),_p_n1_ij.get_shape()))
        return cost  
    return a

# A function that iteratively estimates the well grid block bottomhole pressure, in a low-pressure single-phase system, uing a cost function. 
def compute_pwf_gas(_p_n1_ij,_invBgug_n1_ij,_kr_n1_ij,_min_bhp_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=None,\
                  _lmd=1.,_no_intervals=5,_no_iter=[10],model_PVT=None,_back_prop=False):
    obj_func=cost_func_pwf_nograd_gas(_p_n1_ij,_invBgug_n1_ij,_kr_n1_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=_q_opt_n1_ij,\
                      _lmd=1.,_no_intervals=_no_intervals,model_PVT=model_PVT,_back_prop=_back_prop)
    _pwf_n0_ij=tf.ensure_shape(chp.find_root_chandrupatla(obj_func,low=_min_bhp_ij,high=_p_n1_ij,position_tolerance=1e-08,value_tolerance=0.0,max_iterations=_no_iter[0],stopping_policy_fn=tf.reduce_all,validate_args=False,name='find_root_chandrupatla')[0],_p_n1_ij.get_shape())
    return _pwf_n0_ij

# A function that computes the well grid block flow rate and bottomhole pressure for a dry gas system.
def compute_rate_bhp_gas(_p_n1_ij,_invBgug_n1_ij,_kr_n1_ij,_q_t0_ij,_min_bhp_ij,_Ck_ij,_shutins_idx=1,_invBg_n1_ij=0.,_ctrl_mode=1,_no_intervals=5,_no_iter_per_interval=5,_q_well_idx=None,model_PVT=None):
    _qc_t0_ij=tf.ensure_shape(_q_t0_ij,_p_n1_ij.get_shape())*_shutins_idx
    _min_bhp_ij=tf.ensure_shape(_min_bhp_ij,_p_n1_ij.get_shape())
    PVT_min_bhp_ij=model_PVT(tf.reshape(_min_bhp_ij,(-1,)))
    _invBgug_min_bhp_ij=tf.reshape(PVT_min_bhp_ij[0][0]*PVT_min_bhp_ij[0][1],tf.shape(_p_n1_ij))
    
    _invBg_min_bhp_ij=tf.reshape(PVT_min_bhp_ij[0][0],tf.shape(_p_n1_ij))
    _area_n1_min_bhp_ij=compute_integral_mg(_p_n1_ij,_invBgug_n1_ij,_min_bhp_ij,_no_intervals=_no_intervals,model_PVT=model_PVT,_back_prop=True)

    blk_fac=tf.math.divide_no_nan(_area_n1_min_bhp_ij,(_invBgug_n1_ij*(_p_n1_ij-_min_bhp_ij)))               
    _qmax_n1_ij_min_bhp=_q_well_idx*_Ck_ij*(blk_fac*_kr_n1_ij*_invBgug_n1_ij)*(_p_n1_ij-_min_bhp_ij)
    _q_n1_ij_opt=tf.where(tf.math.equal(_ctrl_mode,1),tf.math.maximum(tf.math.minimum(_qc_t0_ij,_qmax_n1_ij_min_bhp),0.),_qmax_n1_ij_min_bhp) 
    
    #_pwf_n1_ij=_q_well_idx*(_p_n1_ij-(_q_n1_ij_opt/(_Ck_ij*(blk_fac*_kr_n1_ij*_invBgug_n1_ij)))) 
    _pwf_n1_ij=compute_pwf_gas(_p_n1_ij,_invBgug_n1_ij,_kr_n1_ij,_min_bhp_ij,_q_well_idx,_Ck_ij,_q_opt_n1_ij=_q_n1_ij_opt,\
                      _no_intervals=_no_intervals,_no_iter=_no_iter_per_interval,model_PVT=model_PVT,_back_prop=False)
    return _q_n1_ij_opt,_pwf_n1_ij     
   
# A function that computes the well grid block flow rate and bottomhole pressure for a gas-condensate system.
def compute_rate_bhp_gas_oil(_p_n1_ij,_Sg_n1_ij,_invBg_n1_ij,_invBo_n1_ij,_invBgug_n1_ij,_invBouo_n1_ij,_Rs_n1_ij,_Rv_n1_ij,_krgo_n1_ij,_krog_n1_ij,_q_t0_ij,_min_bhp_ij,_Ck_ij,_q_well_idx,_pwf_n1_ij=0.,_Sgi=0.78,_Sorg=0.2,_p_dew=4048.485,\
                     _pi=5000.,_pbase=0.,_shutins_idx=1,_shift=0.001,_lmd=0.,_ctrl_mode=1,_no_intervals=5,_no_iter_per_interval=5,pre_sat_model=None,rel_perm_model=None,model_PVT=None):
    
    DM=tf.constant((5.615/1000.), dtype=_p_n1_ij.dtype, shape=(), name='const22') #  Convert the mobility from stb/bbl to MScf/bbl
    _qc_t0_ij=tf.ensure_shape(_q_t0_ij,_p_n1_ij.get_shape())*_shutins_idx
    _q_well_idx=tf.ensure_shape(_q_well_idx,_p_n1_ij.get_shape())
    _min_bhp_ij=tf.ensure_shape(_min_bhp_ij,_p_n1_ij.get_shape())
    
    __Sg_n1_ij=(_Sg_n1_ij)
   
    __krog_n1_ij,__krgo_n1_ij=rel_perm_model(__Sg_n1_ij)
    
    _mgg_n1_ij=(__krgo_n1_ij*_invBgug_n1_ij)
    _mgo_n1_ij=(__krog_n1_ij*_invBouo_n1_ij)*_Rs_n1_ij
    _moo_n1_ij=(__krog_n1_ij*_invBouo_n1_ij)
    _mog_n1_ij=(__krgo_n1_ij*_invBgug_n1_ij)*_Rv_n1_ij
    _mg_n1_ij=_mgg_n1_ij+_mgo_n1_ij
    _tm_n1_ij=_mgg_n1_ij+_mgo_n1_ij+(_moo_n1_ij+_mog_n1_ij)
    _mo_mg_n1_ij=(tf.math.divide_no_nan((_moo_n1_ij+_mog_n1_ij),(_mgg_n1_ij+_mgo_n1_ij)))
    
    _VroCVD_n1_ij=tf.math.divide_no_nan(_Sgi-_Sg_n1_ij,_Sgi)
    _boil_n1_ij=(_VroCVD_n1_ij*_invBo_n1_ij)+(1.-_VroCVD_n1_ij)*(_Rv_n1_ij*_invBg_n1_ij)
    _bgas_n1_ij=(_VroCVD_n1_ij*_Rs_n1_ij*_invBo_n1_ij)+(1.-_VroCVD_n1_ij)*(_invBg_n1_ij)   
    
    # Max total rate
    _qfg_t0_ij=_qc_t0_ij*tf.math.divide_no_nan(_mgg_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))
    _qvo_t0_ij=_qc_t0_ij*tf.math.divide_no_nan(_mog_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))    
    _qfo_t0_ij=_qc_t0_ij*tf.math.divide_no_nan(_moo_n1_ij,(_mgg_n1_ij+_mgo_n1_ij)) 
    _qdg_t0_ij=_qc_t0_ij*tf.math.divide_no_nan(_mgo_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))
    _qt_t0_ij=_qfg_t0_ij+_qvo_t0_ij+_qfo_t0_ij+_qdg_t0_ij
    
    #so_idx=tf.cast(_p_n1_ij<=_p_dew,_p_n1_ij.dtype)
    _area_n1_min_bhp_ij=compute_integral_mo_mg(_p_n1_ij,__Sg_n1_ij,_boil_n1_ij,_bgas_n1_ij,_mo_mg_n1_ij,_tm_n1_ij,__krog_n1_ij,_min_bhp_ij,\
                       _p_dew=_p_dew,_shift=_shift,_Sgi=_Sgi,_Sorg=_Sorg,_q_well_idx=_q_well_idx,_no_intervals=_no_intervals,_no_iter=_no_iter_per_interval,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=False)

    # _area_n1_hl_bhp_ij=compute_integral_mo_mg(_p_n1_ij,__Sg_n1_ij,_boil_n1_ij,_bgas_n1_ij,_mo_mg_n1_ij,_tm_n1_ij,__krog_n1_ij,(_min_bhp_ij)*1.5,\
    #                   _p_dew=_p_dew,_shift=_shift,_Sgi=_Sgi,_Sorg=_Sorg,_q_well_idx=_q_well_idx,_no_intervals=_no_intervals,_no_iter=_no_iter_per_interval,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=False)

    blk_fac=_lmd+tf.math.divide_no_nan((_area_n1_min_bhp_ij),(_tm_n1_ij*(_p_n1_ij-_min_bhp_ij)))
    #scaled_sigmoid=tfp.bijectors.Sigmoid(low=-10., high=10., validate_args=False, name='sigmoid')
    blk_fac=tf.where(_p_n1_ij>_p_dew,1.,blk_fac) #tf.math.minimum(blk_fac,10.)   
    
    _qg_max_n1_min_bhp_ij=_q_well_idx*_Ck_ij*(blk_fac*(_mgg_n1_ij+_mgo_n1_ij))*(_p_n1_ij-_min_bhp_ij)
    #_qo_max_n1_min_bhp_ij=_q_well_idx*_Ck_ij*(blk_fac*(_moo_n1_ij+_mog_n1_ij))*(_p_n1_ij-_min_bhp_ij)
    _qt_max_n1_min_bhp_ij=_q_well_idx*_Ck_ij*(blk_fac*(_mgg_n1_ij+_mgo_n1_ij+_moo_n1_ij+_mog_n1_ij))*(_p_n1_ij-_min_bhp_ij)
        
    _qg_n1_ij_opt=tf.where(tf.math.equal(_ctrl_mode,1),tf.math.maximum(tf.math.minimum(_qc_t0_ij,_qg_max_n1_min_bhp_ij),0.),_qg_max_n1_min_bhp_ij)
    #_qo_n1_ij_opt=tf.where(tf.math.equal(_ctrl_mode,1),tf.math.maximum(tf.math.minimum(_qc_t0_ij*_Rvi,_qo_max_n1_min_bhp_ij),0.),_qo_max_n1_min_bhp_ij)
    
    #_qt_n1_ij_opt=tf.where(tf.math.equal(_ctrl_mode,1),tf.math.maximum(tf.math.minimum(_qt_t0_ij,_qt_max_n1_min_bhp_ij),0.),_qt_max_n1_min_bhp_ij)
    
    blk_fac_pwf=((tf.math.divide_no_nan(_qg_n1_ij_opt,_qg_max_n1_min_bhp_ij)*blk_fac))
    _pwf_n1_ij=tf.math.maximum(_q_well_idx*(_p_n1_ij-tf.math.divide_no_nan(_qg_n1_ij_opt,(_Ck_ij*((blk_fac_pwf)*(_mgg_n1_ij+_mgo_n1_ij))))),_min_bhp_ij)
    # _pwf_n1_ij=compute_pwf(_p_n1_ij,_Sg_n1_ij,_tm_n1_ij,_mo_mg_n1_ij,_krog_n1_ij,_min_bhp_ij,_q_well_idx,_Ck_ij,_pwf_n1_ij=None,_q_opt_n1_ij=_qg_n1_ij_opt,_qmax_n1_ij_min_bhp=_qg_max_n1_min_bhp_ij,_m_opt_n1_ij=(_mgg_n1_ij+_mgo_n1_ij),\
    #                     _p_dew=4048.,_shift=0.001,_pi=5000.,_Sgi=0.78,_Sorg=0.2,_lmd=1.,_no_intervals=_no_intervals,_no_iter=[4,4],pre_sat_model=pre_sat_model,rel_perm_model=rel_perm_model,model_PVT=model_PVT,_back_prop=False)

    _qfg_n1_ij=_qg_n1_ij_opt*tf.math.divide_no_nan(_mgg_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))
    _qvo_n1_ij=_qg_n1_ij_opt*tf.math.divide_no_nan(_mog_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))  
    _qfo_n1_ij=_qg_n1_ij_opt*tf.math.divide_no_nan(_moo_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))
    _qdg_n1_ij=_qg_n1_ij_opt*tf.math.divide_no_nan(_mgo_n1_ij,(_mgg_n1_ij+_mgo_n1_ij))
    
    # _qfg_n1_ij=_qt_n1_ij_opt*tf.math.divide_no_nan(_mgg_n1_ij,(_tm_n1_ij))
    # _qvo_n1_ij=_qt_n1_ij_opt*tf.math.divide_no_nan(_mog_n1_ij,(_tm_n1_ij))    
    # _qfo_n1_ij=_qt_n1_ij_opt*tf.math.divide_no_nan(_moo_n1_ij,(_tm_n1_ij))
    # _qdg_n1_ij=_qt_n1_ij_opt*tf.math.divide_no_nan(_mgo_n1_ij,(_tm_n1_ij))

    return _qfg_n1_ij,_qdg_n1_ij,_qfo_n1_ij,_qvo_n1_ij,_pwf_n1_ij    
