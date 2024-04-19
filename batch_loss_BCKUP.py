#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
# Edited © 2020 Victor Molokwu <vcm1@hw.ac.uk>
# Distributed under terms of the MIT license.

"""An example of using tfp.optimizer.lbfgs_minimize to optimize a TensorFlow model.
This code shows a naive way to wrap a tf.keras.Model and optimize it with the L-BFGS
optimizer from TensorFlow Probability.
Python interpreter version: 3.6.9
TensorFlow version: 2.0.0
TensorFlow Probability version: 0.8.0
NumPy version: 1.17.2
Matplotlib version: 3.1.1
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot
from pickle import load
import time
dt_type='float64'
#================================================================================================================================
#df = pd.DataFrame({"sum weighted input": np.linspace(-5,5,points)})
#df["sigmoid"] = K.eval(linear(df["sum weighted input"]))

#================================================================================================================================
# Function for computing the Regularized loss using the domain and boundary physics (PINN)
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

# Convert initial parameters--i.e., model.trainable variables, to a 1D Tensor
def convert_1D(model):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
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

#@tf.function
def zeros_to_ones(x):
    y=tf.where(tf.math.equal(x,tf.zeros_like(x)),tf.ones_like(x), x) 
    return y

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
    
    #normfunc_der=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: _z_score()),lambda: tf.ones((),dtype=dt_type))
    normfunc_der=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: tf.ones((),dtype=model.dtype))

    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    normfunc_der=tf.where(tf.logical_or(tf.math.is_nan(normfunc_der), tf.math.is_inf(normfunc_der)),tf.zeros_like(normfunc_der), normfunc_der)
    return normfunc_der

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
    
    #nonorm=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: norm_input)
    nonorm_func=tf.where(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),_lnk_linear_scaling(),_linear_scaling())
    nonorm=tf.where(tf.math.equal(compute,True),nonorm_func,norm_input)
    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    
    nonorm=tf.where(tf.logical_or(tf.math.is_nan(nonorm), tf.math.is_inf(nonorm)),tf.zeros_like(nonorm), nonorm)

    return nonorm

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
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    derivative=tf.where(tf.logical_or(tf.math.is_nan(derivative), tf.math.is_inf(derivative)),tf.zeros_like(derivative), derivative)
    return derivative

@tf.function(jit_compile=True)
def normalize_diff(model,diff,stat_idx=0,compute=False):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Nonnormalized function: Linear scaling (a,b)= (xmax-xmin)*((x_norm-a)/(b-a))+xmin
    #                           Nonnormalized function: z-score= (x_norm*xstd)+xmean
    diff=tf.convert_to_tensor(diff, dtype=model.dtype, name='diff')
    
    def _lnk_linear_scaling():
        lin_scale_no_log=tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]),dtype=model.dtype)*diff
        lin_scale_log=tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0]),dtype=model.dtype)*tf.math.log(diff)

        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)

    def _linear_scaling():
        return tf.convert_to_tensor((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]),dtype=model.dtype)*diff
    
    def _z_score():
        return tf.convert_to_tensor(1/model.ts[stat_idx,3],dtype=model.dtype)*diff
    
    #norm=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: diff)
    norm_func=tf.where(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),_lnk_linear_scaling(),_linear_scaling())
    norm=tf.where(tf.math.equal(compute,True),norm_func,diff)
    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm

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
    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm

# Hard limit activation function
@tf.function
def hard_limit_func(x=None, lower_limit=14.7,upper_limit=5000.,alpha=0.):
    x=tf.convert_to_tensor(x, name='input')
    lower_limit=tf.convert_to_tensor(lower_limit, dtype=x.dtype, name='lower_limit') 
    upper_limit=tf.convert_to_tensor(upper_limit, dtype=x.dtype, name='upper_limit') 
    alpha=tf.convert_to_tensor(alpha, dtype=x.dtype, name='alpha') 
    return tf.where(tf.less(x, lower_limit),lower_limit+tf.math.abs(x-lower_limit)*alpha,tf.where(tf.greater(x, upper_limit),upper_limit+(x-upper_limit)*alpha,x))
   
@tf.function
def tstep_func(model=None,etime=None,error=0.005):
    log10=lambda x:tf.math.log(x)/tf.math.log(10.)
    etime=tf.convert_to_tensor(etime,dtype=model.dtype)
    
    num=int(model.cfd_type['Max_Train_Time']/model.cfd_type['Timestep_Log_Fac'])
    #a=tf.cast(tf.unique(np.logspace(0,np.log10(max_time_step+1),num=tf.cast((max_time_step/4.),dtype=tf.int64)).astype(np.int32),out_idx=tf.int64)[0]-tf.constant(1),dtype=model.dtype)
    a=tf.cast(tf.unique(tf.cast(tf.experimental.numpy.logspace(0,log10(model.cfd_type['Max_Train_Time']+1),num=num),dtype=tf.int64))[0]-tf.constant(1,dtype=tf.int64),dtype=model.dtype)
    b=a[1:]-a[:-1]

    # Duplicate/pad the last index
    c=tf.pad(b,[[0,1]],constant_values=b[-1])

    #Expand the dimesion
    expand_etime=etime[...,None]
    
    #Tile the expanded tensor
    tiled_a = tf.tile(a[None, ...], [tf.shape(etime)[0], 1])
    tiled_a_low=(1-error)*tiled_a
    tiled_a_high=(1+error)*tiled_a

    # Now expanded_time and tiled_a are broadcastable so we can compare
    # each element of time to all elements in a in parallel
    #mult = tf.cast(tf.equal(expand_time, tiled_a), tf.float64) 
    mult = tf.cast(tf.logical_and(tf.greater_equal(expand_etime, tiled_a_low),tf.less_equal(expand_etime, tiled_a_high)), tf.float64) 

    # from mult we need first index from axis -1 that is != 0 (using argmax)
    # sub shows which rows have all zeros (no element of time in a)
    # for such rows we put value as the last index
    sub = tf.cast(tf.math.equal(tf.reduce_sum(mult, -1), 0), dtype=tf.int64)*tf.cast((tf.shape(a)[0]-1),dtype=tf.int64)
    
    # result
    res = tf.argmax(mult, axis=-1) + sub    

    return tf.gather(c,res)
    

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

def watch_layer_1(layer):
    def decorator(func):   
        def wrapper(*args, **kwargs):
            # Store the result of `layer.call` internally.
            layer.result = func(*args, **kwargs)
            return layer.result
        return wrapper
    layer.call = decorator(layer.call)
    return layer

@tf.function
def well_bound_input_tensors(model,x,dx,dy,angle_deg=None):
    # Convert the angle to radians
    angle_rad=tf.cast((angle_deg*22)/(7*180),dtype=dt_type)
    ro_dx=dx*tf.math.cos(angle_rad)
    ro_dy=dy*tf.math.sin(angle_rad)
    
    # Using a linear scaling a=-1 anc b=1
    b_a=tf.cast(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0],dt_type)
    ro_dxn=ro_dx*(b_a/(model.ts[0,1]-model.ts[0,0]))
    ro_dyn=ro_dy*(b_a/(model.ts[1,1]-model.ts[1,0]))

    # Unstack the n-1 tuple
    _x=x[0]+(ro_dxn)
    _y=x[1]+(ro_dyn)
    x_restk=tf.tuple([_x,_y,x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15]])
    return x_restk

   
@tf.function
def pinn_error(model,x,y):
    with tf.device('/GPU:0'):                   # GPU is better if available
        # Physics gradient for Arrangement Type 1: 
        # DATA ARRANGEMENT FOR TYPE 1
        # Training Features: INPUTS: 
        # x is a list of inputs (numpy arrays or tensors) for the given batch_size
        # x[0] is the grid block x_coord in ft
        # x[1] is the grid block y_coord in ft
        # x[2] is the grid block z_coord in ft
        # x[3] is the time in day
        # x[4] is the grid block porosity 
        # x[5] is the grid block x-permeability in mD
        # x[6] is the grid block z-permeability in mD
        
        # x[7] is the segment vector in the x-direction in ft (used for approximating the outer boundary)
        # x[8] is the segment vector in the y-direction in ft (used for approximating the outer boundary)
        # x[9] is the segment vector in the z-direction in ft (used for approximating the outer boundary)        
        # x[10] is the grid block x-dimension in ft (used for Inner Boundary Condition)--Average values can be used
        # x[11] is the grid block y-dimension in ft (used for Inner Boundary Condition)--Average values can be used
        # x[12] is the grid block z-dimension in ft (used for Inner Boundary Condition)--Average values can be used
        # x[13] is the harmonic average capacity ratio (i.e., kxdz(i+1)/kxdz(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
        # x[14] is the harmonic average capacity ratio (i.e., kzdx(i+1)/kzdx(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
        # x[15] is the input label as float indicating whether DOM(0), DBC(1), NBC(2), IBC(3), IC(4) or Train Data(5)   Label x[0-DOM|1-DBC|2-NBC|3-IBC|4-IC|5-TD-Full|6-TD-Sample]

        # Training Label: OUTPUTS:
        # y[0] is the training label grid block pressure (psia)
        # y[1] is the training label block saturation--gas
        # y[2] is the training label gas Formation Volume Factor in bbl/MScf
        # y[3] is the training label gas viscosity in cp
        # y[4] is the training label gas rate in Mscf/D
        
        # Model OUTPUTS:
        # out[0] is the predicted grid block pressure (psia)
        # out[1] is the predicted grid block gas saturation
        # out[2] is the predicted grid block gas Formation Volume Factor inverse (1/Bg)
        # out[3] is the predicted grid block gas viscosity inverse (1/ug)
        #======================================================================================================
        # Peaceman (1978, 1983) showed that the actual flowing pressure equals the numerically calculated well-block pressure at the well-block radius
        # Wellblock_radius, ro=0.14*((grid_dx**2)+(grid_dy**2))**(1/2)         
        #ro=0.14*((x[10]**2)+(x[11]**2))**(1/2)
        #ro=tf.constant(0.09525, dtype=dt_type, shape=(1,),name='well_radius')
        #ro_t=tf.cast((22./(6.*7.)),dt_type)
        #ro_dx=ro*tf.cos(ro_t); ro_dy=ro*tf.sin(ro_t)
    
        # Using a linear scaling a=-1 anc b=1
        #b_a=tf.cast(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0],dt_type)
        #ro_dxn=ro_dx*(b_a/(model.ts[0,1]-model.ts[0,0])); ro_dyn=ro_dy*(b_a/(model.ts[1,1]-model.ts[1,0]))

        # Using z-score standardization
        # ro_dxn=ro_dx*(1/model.ts[0,3]); ro_dyn=ro_dy*(1/model.ts[1,3])
        
        # Get the datatype using the first index of the features
        dt_type=model.dtype
        #======================================================================================================
        # Compute the DOM, DBC, NBC, IBC errors. IC is not computed as it is considered a unique training data
        # Debugger ... ... ...
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        # Unnormalize (if any),reshape and pad the outer boundary for the constitutive relationships (k,phi)
        # The paddings for the outer boundary is 'SYMMETRIC'--implies a no flow boundary
        paddings = tf.constant([[0, 0], [1, 1], [1, 1],[0, 0]])

        phi=tf.pad(tf.reshape(nonormalize(model,x[4],stat_idx=4,compute=compute_),(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     
        # Add noise to the permeability
        #x5=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.5, dtype=dt_type)
        kx=tf.pad(tf.reshape(nonormalize(model,x[5],stat_idx=5,compute=compute_),(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     
        ky=kx
        #kz=tf.pad(tf.reshape(nonormalize(model,x[6],stat_idx=6,compute=compute_),(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     
        
        dx=tf.pad(tf.reshape(x[7],(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     
        dy=tf.pad(tf.reshape(x[8],(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     
        dz=tf.pad(tf.reshape(x[9],(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC')     

        dx_ij=dx[:,1:-1,1:-1,:]; dx_i1=dx[:,2:,1:-1,:]; dx_i_1=dx[:,:-2,1:-1,:]
        dy_ij=dy[:,1:-1,1:-1,:]; dy_j1=dy[:,1:-1,2:,:]; dy_j_1=dy[:,1:-1,:-2,:] 
        dz_ij=dz[:,1:-1,1:-1,:] 
        dv=(dx_ij*dy_ij*dz_ij)
        
        dx_avg_ih=(dx_i1+dx_ij)/2.; dx_avg_i_h=(dx_ij+dx_i_1)/2.   
        dy_avg_jh=(dy_j1+dy_ij)/2.; dy_avg_j_h=(dy_ij+dy_j_1)/2.  
        
        def physics_error():
            # Reshape the well rates
            q_n1_ij=tf.reshape(y[-2],(model.cfd_type['Dimension']['Reshape']))
            #cq_n1_ij=tf.reshape(y[-1],(model.cfd_type['Dimension']['Reshape'])) 
            #=======================================================================================================
            # Compute the solutions for the Domain, Boundary, Initial Conditions
            # Create a list for storing domain, DBC, NBC, IBC, IC and TrainData solutions
            # Conversion constant 1 -- 0.001127
            # Conversion constant 2 -- 5.6145833334 from rb/D to cf/D
            C=tf.constant(0.001127, dtype=dt_type, shape=(), name='const1')
            D=tf.constant(5.6145833334, dtype=dt_type, shape=(), name='const2')
            rw=tf.constant(0.1905/2, dtype=dt_type, shape=(), name='rw')
        
            #====================================Domain Solution====================================================
            # PDE: div.[(k/ugBg)*(grad(p)*C)=c(p)*(dp/dt) + q  (Mscf/D)        
            # PDE: (C*kx)*((d2p/dx2)*(1/ugBg)+(dp/dx)^2*d/dp(1/ugBg))+
            #      (C*ky)*((d2p/dy2)*(1/ugBg)+(dp/dy)^2*d/dp(1/ugBg))+
            #      (C*kz)*((d2p/dz2)*(1/ugBg)+(dp/dz)^2*d/dp(1/ugBg))+
            #      ((phi*d/dp(1/Bg))+(phi_0*cr*(1/Bg)))*dp/dt-
            #      (q/Vb)=0
            
            #      C=0.001127
            #      Vb is the volume of block. Vb=dx*dy*dz
            # Compute by applying product rule: d_dp_invBgug=(1/Bg)*d/dp(1/ug)+(1/ug)*d/dp(1/Bg)
     
            # Compute the model forward pass at n, reshape and pad the outer boundary for the QoIs (pressure, saturation...):
            xn0=x
            out_n0=model(xn0, training=True)
            out_n0=tf.stack([tf.pad(tf.reshape(out_n0[i],(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5]])
            
            # Compute the model forward pass at n+1, reshape and pad the outer boundary:
            # At n+1, the timestep interval is normalized before used as a feature
            tstep=model.cfd_type['Timestep']*(1-((1-model.cfd_type['Timestep_Fac'])*tf.cast(tf.math.reduce_mean(kx)<model.cfd_type['K_Timestep'],model.dtype)))
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)
            xn1=list(x)
            xn1[3]+=tstep_norm                  # Update the timestep
            
            """
            with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape1:
                p_int=model.get_layer('pressure')
                watch_layer(p_int,tape1)
                out_n1=model(xn1, training=True)
    
            # Compute d_dp_invBg at p(n+1), reshape and then pad the outer boundary
            d_dp_invBg_n1=tf.pad(tf.reshape(tape1.gradient(out_n1[2],p_int.result,unconnected_gradients='zero'),model.cfd_type['Dimension']['Reshape']),paddings,mode='SYMMETRIC')
            """
            out_n1=model(xn1, training=True)
            out_n1=tf.stack([tf.pad(tf.reshape(out_n1[i],(model.cfd_type['Dimension']['Reshape'])),paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5]])  

            # Compute d_dp_invBg at p(n+1) using the chord slope
            d_dp_invBg_n1=(out_n1[2]-out_n0[2])/(out_n1[0]-out_n0[0])
            
            """# Compution using Finite Difference
            grid_spacing=0.01; p_in=tf.keras.Input(tensor=model.get_layer('invBg_split_hlayer_1').input)
            model_Bu=tf.keras.Model(inputs=p_in,outputs=[model.get_layer('invBg').output,model.get_layer('invug').output])
            p_n1_flat=tf.reshape(out_n1[0],(-1,1))
            d_dp_invBg_n1_fdiff=tf.reshape(model_Bu(p_n1_flat+grid_spacing)[0]-model_Bu(p_n1_flat-grid_spacing)[0]/(2*grid_spacing),tf.shape(out_n1[0]))
            """
            invBg_n0=(out_n0[2])
                    
            invBg_n1=(out_n1[2])
            invug_n1=(out_n1[3])
            invBgug_n1=(invBg_n1*invug_n1)
            model.phi_0=phi
            model.cf=97.32e-6/(1+55.8721*model.phi_0**1.428586)
            cr_n0_ij=(model.phi_0*model.cf*invBg_n0)
            cp_n1=Sg*((phi*d_dp_invBg_n1)+cr_n0_ij)
            cp_n1_ij=cp_n1[:,1:-1,1:-1,:]
    
            kx_ij=kx[:,1:-1,1:-1,:]; kx_i1=kx[:,2:,1:-1,:]; kx_i_1=kx[:,:-2,1:-1,:]
            ky_ij=ky[:,1:-1,1:-1,:]; ky_j1=ky[:,1:-1,2:,:]; ky_j_1=ky[:,1:-1,:-2,:]
            kx_avg_ih=(2.*kx_i1*kx_ij)/(kx_i1+kx_ij); kx_avg_i_h=(2.*kx_ij*kx_i_1)/(kx_ij+kx_i_1)
            ky_avg_jh=(2.*ky_j1*ky_ij)/(ky_j1+ky_ij); ky_avg_j_h=(2.*ky_ij*ky_j_1)/(ky_ij+ky_j_1)

            # Using Peaceman solution (1973) to calculate the equivalent wellblock radius (For IBC solution): 
            ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx_ij,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy_ij,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
            
            invBgug_n1_ij=invBgug_n1[:,1:-1,1:-1,:]; invBgug_n1_i1=invBgug_n1[:,2:,1:-1,:]; invBgug_n1_i_1=invBgug_n1[:,:-2,1:-1,:]
            invBgug_n1_j1=invBgug_n1[:,1:-1,2:,:]; invBgug_n1_j_1=invBgug_n1[:,1:-1,:-2,:]
            invBgug_avg_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_avg_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_avg_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_avg_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            
            p_n1_ij=out_n1[0][:,1:-1,1:-1,:]; p_n1_i1=out_n1[0][:,2:,1:-1,:]; p_n1_i_1=out_n1[0][:,:-2,1:-1,:]
            p_n1_j1=out_n1[0][:,1:-1,2:,:]; p_n1_j_1=out_n1[0][:,1:-1,:-2,:]
            p_n0_ij=out_n0[0][:,1:-1,1:-1,:]
            
            a1=C*kx_avg_i_h*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(1/dx_ij)
            a2=C*ky_avg_j_h*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(1/dy_ij)
            a3=C*kx_avg_ih*invBgug_avg_n1_ih*(1/dx_avg_ih)*(1/dx_ij)  
            a4=C*ky_avg_jh*invBgug_avg_n1_jh*(1/dy_avg_jh)*(1/dy_ij)
            a5=(1/D)*(cp_n1_ij/tstep)
            # Full discretized equation
            # dom_idx=tf.pad(tf.ones_like(q_n1_ij[1:-1,1:-1,:]),paddings,mode='CONSTANT')
            # qconst_well_idx=tf.cast((q_n1_ij>0.)&(q_n1_ij==model.cfd_type['Init_Grate']),model.dtype)
            # Add noise to the constant rate
            # qnoise=qconst_well_idx*tf.random.normal(tf.shape(q_n1_ij),mean=0.0,stddev=0.10,dtype=dt_type,seed=None)
            # q_n1_ij=q_n1_ij*(1.+qnoise)
      
            minbhp_well_idx=tf.cast((q_n1_ij>0.)&(q_n1_ij<model.cfd_type['Init_Grate']),model.dtype)
            #qconst_rate=qconst_well_idx*(q_n1_ij)
            #qconst_bhp=minbhp_well_idx*(2*(22/7)*kx_ij*dz_ij*C*invBgug_n1_ij*(1./tf.math.log(ro/rw))*(p_n1_ij-model.cfd_type['Min_BHP']))
            
            #q_var=qconst_rate+qconst_bhp
            
            dom=dv*((-a1*p_n1_i_1)+(-a2*p_n1_j_1)+((a1+a2+a3+a4+a5)*p_n1_ij)+(-a3*p_n1_i1)+(-a4*p_n1_j1)+(q_n1_ij/dv)-(a5*p_n0_ij))
            #====================================DBC Solution======================================================= 
            # Solution on the surface and same as pressure
            dbc=tf.zeros_like(dom)                 # Set at zero for now!
            #print(dbc)
            #===================================IBC_NEUMANN Solution========================================================
            # Node Equation at a well is given by: qw+(q1+q2+q3+q4)=0
            # Four faces are given in anticlockwise order: 1,2,3 and 4
            # Under steadystate: [q(i-1/2,j)-q(i+1/2,j)]+[q(i,j-1/2)-q(i,j+1/2)]-q(i,j)=0
            # Using Peaceman solution for pressure at the grid
            b1=C*kx_avg_i_h*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(dz_ij*dy_ij)
            b2=C*ky_avg_j_h*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(dz_ij*dx_ij)
            b3=C*kx_avg_ih*invBgug_avg_n1_ih*(1/dx_avg_ih)*(dz_ij*dy_ij)
            b4=C*ky_avg_jh*invBgug_avg_n1_jh*(1/dy_avg_jh)*(dz_ij*dx_ij)
            
            well_block_rates=(b1*p_n1_i_1)+(b2*p_n1_j_1)+(-(b1+b2+b3+b4)*p_n1_ij)+(b3*p_n1_i1)+(b4*p_n1_j1)
     
            # Cast a tensor of 1 for the well block and 0 for other cells &(q_n1_ij==model.cfd_type['Init_Grate'])
            q_well_idx=tf.cast((q_n1_ij>0.),model.dtype)
            
            ibc_n=(well_block_rates*q_well_idx)-q_n1_ij
            #===================================IBC_DIRICHLET Solution========================================================
            # This solution is when the inner boundary is under a constant bottom hole pressure
            # pwf=p_n1_ij-((well_block_rates*minbhp_well_idx)/(2*(22/7)*kx_ij*dz_ij*C))*(tf.math.log(ro/rw))*(1/(invBgug_n1_ij))  
            # nbc=minbhp_well_idx*(model.cfd_type['Min_BHP']-pwf)
            nbc=tf.zeros_like(dom)   # Debugging...
            #====================================NBC Solution======================================================= 
            # Neumann Boundary Condition--solution on the normal derivative to the surface
            # Expressed in equation form: qrate(obs)-qrate(nbc)=0
            # q_bd should be in MScf/D and gfvf in rb/MScf: Const1 gives result in rb/D 
            # nbc_idx=tf.pad(tf.zeros_like(q_n1_ij[1:-1,1:-1,:]),paddings,mode='CONSTANT',constant_values=1.)
            # nbc_block_rates=(-b1*(p_n1_ij-p_n1_i_1))+(-b2*(p_n1_ij-p_n1_j_1))+(-b3*(p_n1_i1-p_n1_ij))+(-b4*(p_n1_j1-p_n1_ij))
            # nbc=(nbc_idx*nbc_block_rates)                # Set at zero for now!
            # nbc=tf.zeros_like(dom) 
            
            #===================================Material Balance Check==============================================
            invBg_n0_ij=invBg_n0[:,1:-1,1:-1,:]
            invBg_n1_ij=invBg_n1[:,1:-1,1:-1,:]
            phi_n1_ij=phi[:,1:-1,1:-1,:]
            mbc=-tf.reduce_sum(q_n1_ij)-tf.reduce_sum(dv*Sg*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)))

            #===================================Cumulative Material Balance Check===================================
            xn_t0=list(x)
            xn_t0[3]=model.cfd_type['Norm_Limits'][0]*tf.ones_like(xn_t0[0])
            out_ic=model(xn_t0, training=True)
            out_ic=tf.stack([tf.reshape(out_ic[i],(model.cfd_type['Dimension']['Reshape'])) for i in [0,2]])          
            #invBg_t0_ij=out_ic[-1]
            #model.cum['Gas_Obs'].assign_add(tf.reduce_sum(q_n1_ij)*tstep)
            #cmbc=-tf.reduce_sum(cq_n1_ij)-tf.reduce_sum(dv*Sg*phi_n1_ij*(invBg_n1_ij-invBg_t0_ij)*(1/(D)))
            cmbc=tf.zeros_like(dom)
            
            #===================================Cumulative Material Balance Check===================================
            ic=model.cfd_type['Pi']-out_ic[0]
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  #update later    
            return [dom,dbc,nbc,ibc_n,ic,mbc,cmbc],[out_n0[:,:,1:-1,1:-1,:],out_n1[:,:,1:-1,1:-1,:]],no_grid_blocks

        def physics_error_zeros_like():
            dom=tf.reshape(tf.zeros_like(y[0],dtype=dt_type),(model.cfd_type['Dimension']['Reshape']))
            dbc=dom
            nbc=dom
            ibc_n=dom
            mbc=dom
            cmbc=dom
            out_n0=tf.stack([dom]*4) 
            out_n1=tf.stack([dom]*4) 
            no_grid_blocks=[0.,0.,0.,0.,0.]
            return [dom,dbc,nbc,ibc_n,mbc,cmbc],[out_n0,out_n1],no_grid_blocks
        
        def ic_error():
            #=====================================IC Solution======================================================= 
            # Forward pass is used as it still the model solution but at initial condition (t=0) 
            # Initial Condition (IC) feature
            xn_t0=list(x)
            xn_t0[3]=model.cfd_type['Norm_Limits'][0]*tf.ones_like(xn_t0[0])
            xn_t0t=tf.tuple(xn_t0)
            #breakpoint()
            out_ic=model(xn_t0, training=True)
            out_ic=tf.reshape(out_ic[0],model.cfd_type['Dimension']['Reshape'])  # Idx 0: Pressure; 2: 1/Bg
           
            ic=model.cfd_type['Pi']-out_ic
       
            # out_ic=model(x)
            # xn_t0_idx=tf.reshape(tf.cast(x[3]==model.cfd_type['Norm_Limits'][0],model.dtype),model.cfd_type['Dimension']['Reshape'])
            # ic=xn_t0_idx*tf.reshape((model.cfd_type['Pi']-tf.squeeze(out_ic[0])),model.cfd_type['Dimension']['Reshape']) 
            return [ic]
        
        #phy_error_out_n=tf.cond(tf.math.equal(model.nwt[0],0.),lambda: physics_error_zeros_like(),lambda: physics_error())
        phy_error,out_n,no_blks=physics_error()
  
        stacked_pinn_errors=tf.stack(phy_error[0:-2],axis=0)     #+ic_error()
        stacked_outs=tf.stack(out_n,axis=0)
        checks=[phy_error[-2],phy_error[-1]]
        #breakpoint()
        return stacked_pinn_errors,stacked_outs,checks,no_blks

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
    # Compute the second order derivatives 
    d2p_dtn2=tape2.gradient(dp_dtn, x[3], unconnected_gradients='zero')    #Experimental use pfor set to False to prevent parallel tf tracing
    del tape2
    dtn_dt=normfunc_derivative(model,stat_idx=3,compute=True)
    d2p_dt2=d2p_dtn2*(dtn_dt)**2
    return p,d2p_dt2
           
#@tf.function()  #--Taking a  derivative from outside exposes the TensorArray to the boundary, and the conversion is not implemented in Tensorflow
def pinn_error_2D(model,x,y):

    #Stack list
    '''
    x1=[list(i) for i in [x,x]]; y1=[list(i) for i in [y,y]]
    x1[1][3]+=1--1
    x=[tf.concat([x1[i][j] for i in [0,1]],0) for j in [0,1,2,3,4,5,6,7,8,9,10]] #tf.range(0,tf.shape(x[:-1])[0]+1)'''
    # 1D model adapted to 2D for fast computiation on graph mode
    with tf.device('/GPU:0'):                   # GPU is better if available
        dt_type=model.dtype
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        
        paddings = tf.constant([[0,0], [1, 1,], [1, 1],[0, 0]])
        
        #t=nonormalize(model,x[3],stat_idx=3,compute=compute_)
        phi=tf.pad(nonormalize(model,x[4],stat_idx=4,compute=compute_),paddings,mode='SYMMETRIC')     
        # Add noise to the permeability
        #x5=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.05, dtype=dt_type)

        kx=tf.pad(nonormalize(model,x[5],stat_idx=5,compute=compute_),paddings,mode='SYMMETRIC') 
        #kx=tf.random.normal((),mean=kx,stddev=0.005,dtype=tf.dtypes.float32,seed=model.cfd_type['Seed'])
        ky=kx
        #kz=tf.pad(nonormalize(model,x[6],stat_idx=6,compute=compute_),paddings,mode='SYMMETRIC')     
        
        dx=tf.pad(x[7],paddings,mode='SYMMETRIC')     
        dy=tf.pad(x[8],paddings,mode='SYMMETRIC')     
        dz=tf.pad(x[9],paddings,mode='SYMMETRIC')     

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
        q_n1_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Init_Grate'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        p_min=tf.math.reduce_min(model.cfd_type['Min_BHP']) #tf.reduce_mean(p_n0_ij,axis=[1,2,3])
        p_max=model.cfd_type['Pi']   # p_t0_ij
        no_wells=tf.cast(tf.shape(model.cfd_type['Init_Grate']),model.dtype)
        area_ij=dx_ij*dy_ij
        area_res=tf.cast(tf.math.reduce_prod(model.cfd_type['Dimension']['Measurement'][:2]),model.dtype)
        # Define an optimal rate and BHP function
        def compute_rate_bhp(_p_n1_ij,_invBgug_n1_ij,_p_t0_ij,_invBgug_t0_ij,_kr_n1_ij,_q_n1_ij):
            #_invBgug_t0_ij=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
            min_bhp_ij=p_min
            _p_n1_ij=(tf.cast(_p_n1_ij>=_p_t0_ij,model.dtype)*_p_t0_ij)+(tf.cast((_p_n1_ij<_p_t0_ij)&(_p_n1_ij>min_bhp_ij),model.dtype)*_p_n1_ij)+(tf.cast(_p_n1_ij<=min_bhp_ij,model.dtype)*min_bhp_ij)
            _invBgug_n1_ij=(tf.cast(_p_n1_ij>=_p_t0_ij,model.dtype)*_invBgug_t0_ij)+(tf.cast((_p_n1_ij<_p_t0_ij)&(_p_n1_ij>min_bhp_ij),model.dtype)*_invBgug_n1_ij)+(tf.cast(_p_n1_ij<=min_bhp_ij,model.dtype)*_invBgug_n1_ij)
           
            # Calculate the maximum rate at minimum bottomhole pressure
            _qmax_n1_ij_min_bhp=q_well_idx*((2*(22/7)*_kr_n1_ij*kx_ij*dz_ij*C)*(1/(tf.math.log(ro/rw)))*(_invBgug_n1_ij)*(_p_n1_ij-min_bhp_ij))
            _q_n1_ij_opt=_q_n1_ij#tf.math.minimum(_q_n1_ij,tf.math.maximum(_qmax_n1_ij_min_bhp,0.))
            _pwf_n1_ij=(_p_n1_ij-(_q_n1_ij_opt/(2*(22/7)*kx_ij*dz_ij*C))*(tf.math.log(ro/rw))*(1/(_invBgug_n1_ij))) 
            return _q_n1_ij_opt,_pwf_n1_ij        
        #Reset the weight

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
        
        # Normalized constants
        t0_norm=normalize(model,0.,stat_idx=3,compute=True)
        t1_norm=normalize(model,1.,stat_idx=3,compute=True)
        t10_norm=normalize(model,10.,stat_idx=3,compute=True)
        t20_norm=normalize(model,10.,stat_idx=3,compute=True)
        t75_norm=normalize(model,75.,stat_idx=3,compute=True)
        tmax_norm=normalize(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
        tmax_norm_diff=normalize_diff(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
                        
        #Time shifting
        def time_shifting(xi,shift_frac_mean=0.2,pred_cycle_mean=1.,random=False):
            xp=list(xi)
            pred_cycle=pred_cycle_mean
            #pred_cycle=tf.math.abs(tf.random.normal((),mean=pred_cycle_mean,stddev=(pred_cycle_mean-1)/5,dtype=tf.dtypes.float32,seed=model.cfd_type['Seed']))
            shift_frac=shift_frac_mean; np=4
            #shift_frac=tf.math.abs(tf.random.stateless_normal((),[model.cfd_type['Seed']]*2,mean=shift_fac,stddev=(0.33*shift_fac),dtype=model.dtype,alg='auto_select')); np=4
            #shift_frac=tf.math.abs(tf.random.normal((),mean=shift_frac_mean,stddev=(shift_frac_mean/3),dtype=tf.dtypes.float32,seed=model.cfd_type['Seed']))
            t=nonormalize(model,xp[3],stat_idx=3,compute=compute_)
            tsf_0=(1-shift_frac)*model.cfd_type['Max_Train_Time']
        
            tshift_range=(((pred_cycle+shift_frac)*model.cfd_type['Max_Train_Time']))
            tshift=(((t-tsf_0)/(model.cfd_type['Max_Train_Time']-tsf_0))*(tshift_range))+tsf_0-t 
            tshift_fac=((pred_cycle+shift_frac)/(shift_frac))
            #tshift_fac=tf.math.abs(tf.random.normal((),mean=tshift_fac,stddev=tshift_fac/5,dtype=tf.dtypes.float32,seed=model.cfd_type['Seed']))
            tsf_0_norm=normalize(model,tsf_0,stat_idx=3,compute=True)  #.995
            tpred_norm=normalize_diff(model,tshift,stat_idx=3,compute=True)
            xp3=(tf.cast(xp[3]<tsf_0_norm,model.dtype)*xp[3])+(tf.cast(xp[3]>=tsf_0_norm,model.dtype)*(xp[3]+tpred_norm))
            xp[3]+=-xp[3]+xp3
            return xp,tshift_fac,tsf_0_norm
        
        def physics_error(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            out_n0=model(xn0, training=True)
            #out_n0,d2p_dt=second_order_derivative_AD(model,x,y)
            out_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5]])
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]
            invBg_n0_ij=out_n0[2][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBgug_n0_ij=(out_n0[2]*out_n0[3])[...,1:-1,1:-1,:]
            # ============================Compute the tuning model network============================================
            # At n+1, the timestep interval is normalized before used as a feature
            p_n0_ij_norm=(((p_n0_ij-p_min)/(model.cfd_type['Pi']-p_min))*(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.cfd_type['Norm_Limits'][0]
            trn_err_tstep_fac=model.trn_model([xn0[3],xn0[5]])  #[tn0,p_n0_ij] [xn0[3],(xn0[5])] [xn0[3],p_n0_ij_norm]  [tn0,(kx/(out_n0[2]*out_n0[3]))]
            prec=0.01#
            #tstep_fac=tf.reduce_mean(trn_err_tstep_fac[0],axis=[1,2,3],keepdims=True) #5e-3       # 1e-4 to 1e-3
            step_fac=trn_err_tstep_fac[1]#tf.reduce_mean(trn_err_tstep_fac[1],axis=[1,2,3],keepdims=True)
            min_tstep=(48.*eps/prec)**(1/4)#(model.cfd_type['Timestep'])*step_fac
            min_tstep_norm=normalize_diff(model,min_tstep,stat_idx=3,compute=True)
            
            xn2=list(xi)    
            xn2[3]+=min_tstep_norm#tstep_norm*step_fac #tstep_norm
            out_n2=model(xn2, training=True)
            p_n2_ij=out_n2[0]  
            invBg_n2_ij=out_n2[2]
            invBgug_n2_ij=out_n2[2]*out_n2[3]
            #phi_n2_ij=phi[...,1:-1,1:-1,:]
            
            xn_2=list(xi)
            xn_2[3]-=min_tstep_norm#tstep_norm*step_fac #tstep_norm
            out_n_2=model(xn_2, training=True)
            p_n_2_ij=out_n_2[0]
            invBg_n_2_ij=out_n_2[2]
            invBgug_n_2_ij=out_n_2[2]*out_n_2[3]
            
            #=========================================================================================================
            #Define a timestep
            # Compute the Max gridblock pressure change -- weighted around the well
            #max_gridblk_pres_change=0.0002; max_tstep=model.cfd_type['Timestep']*1.; min_tstep=1.
            #tstep_est=tf.reduce_mean(tf.math.abs(tf.math.divide_no_nan((max_gridblk_pres_change*p_n0_ij),(dp_dt))),axis=[1,2,3],keepdims=True)
            #tstep_est=tf.maximum((tf.math.divide_no_nan((max_gridblk_pres_change*tf.reduce_mean(p_n0_ij,axis=[1,2,3],keepdims=True)),(dp_dt))),model.cfd_type['Timestep'])
            #tstep=model.cfd_type['Timestep']
            #tstep=step_fac*model.cfd_type['Timestep']
            tstep=(tf.cast(xn0[3]<t10_norm,model.dtype)*(model.cfd_type['Timestep']*1.))+(tf.cast((xn0[3]>=t10_norm)&(xn0[3]<tsn['Time']),model.dtype)*model.cfd_type['Timestep'])+\
                        (tf.cast((xn0[3]>=tsn['Time']),model.dtype)*model.cfd_type['Timestep']*tsn['Shift_Fac'])#*(tshift_fac) tsn['Shift_Fac']
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)
            tstep_norm_diff=normalize_diff(model,tstep,stat_idx=3,compute=True)
            
            # Update the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)
            out_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5]])  
            
            # Update the timestep (n-1)
            xn_1=list(xi)
            xn_1[3]-=tstep_norm #tstep_norm
            out_n_1=model(xn_1, training=True)
            p_n_1_ij=out_n_1[0]
            invBg_n_1_ij=out_n_1[2]
            
            # Initial Condition
            xn_t0=list(xi)
            t0_idx=tf.cast(xn0[3]<normalize(model,0.005,stat_idx=3,compute=True),model.dtype)
            tstep_idx=tf.cast(xn0[3]<normalize(model,tstep,stat_idx=3,compute=True),model.dtype)
            xn_t0[3]+=-xn_t0[3]+(tf.ones_like(xn_t0[0])*t0_norm)#+tf.random.normal(shape=tf.shape(xn_t0[0]), mean=0.0, stddev=0.0005, dtype=dt_type)
            out_t0=model(xn_t0)
            p_t0_ij=out_t0[0]
            invBg_t0_ij=out_t0[2]
            invug_t0_ij=out_t0[3]
            invBgug_t0_ij=(out_t0[2]*out_t0[3]) 
            
            #====================================================================================================================
            #Compute the second order time derivatives to check for PSS
            #dp_dt=(tf.reduce_mean(p_n2_ij,axis=[1,2,3],keepdims=True)-tf.reduce_mean(p_n_2_ij,axis=[1,2,3],keepdims=True))/(2*tstep*step_fac)
            #d2p_dt2=(tf.math.divide_no_nan((p_n2_ij-2*(p_n0_ij)+p_n_2_ij),(min_tstep)**2))
            d2p_dt2=(tf.math.divide_no_nan((p_n2_ij-2*(p_n0_ij)+p_n_2_ij),(min_tstep)**2))
            #dp_dt=(tf.math.divide_no_nan((p_n2_ij-p_n_2_ij),(2*min_tstep)))
            #d2p_dt2=(tf.math.divide_no_nan((pwf_n2_ij-2*(pwf_n0_ij)+pwf_n_2_ij),(min_tstep)**2))
            
            # Compute bottom hole pressure
            _,pwf_n_2_ij=compute_rate_bhp(p_n_2_ij,invBgug_n_2_ij,p_t0_ij,invBgug_t0_ij,1.,q_n1_ij)
            _,pwf_n0_ij=compute_rate_bhp(p_n0_ij,invBgug_n0_ij,p_t0_ij,invBgug_t0_ij,1.,q_n1_ij)
            _,pwf_n2_ij=compute_rate_bhp(p_n2_ij,invBgug_n2_ij,p_t0_ij,invBgug_t0_ij,1.,q_n1_ij)
            #====================================================================================================================
            #Define pressure variables 
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]; 
            p_n1_i1=out_n1[0][...,1:-1,2:,:]; p_n1_i_1=out_n1[0][...,1:-1,:-2,:]
            p_n1_j1=out_n1[0][...,2:,1:-1,:]; p_n1_j_1=out_n1[0][...,:-2,1:-1,:]
            #====================================================================================================================
            # Compute d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision
            d_dp_invBg_n1=tf.math.divide_no_nan((out_n1[2]-out_n0[2]),(out_n1[0]-out_n0[0]))
            invBg_n1_ij=out_n1[2][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[3][...,1:-1,1:-1,:]

            invBgug_n1=(out_n1[2]*out_n1[3])
            invBgug_n1_ij=invBgug_n1[...,1:-1,1:-1,:]; 
            invBgug_n1_i1=invBgug_n1[...,1:-1,2:,:]; invBgug_n1_i_1=invBgug_n1[...,1:-1,:-2,:]
            invBgug_n1_j1=invBgug_n1[...,2:,1:-1,:]; invBgug_n1_j_1=invBgug_n1[...,:-2,1:-1,:]
            invBgug_avg_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_avg_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_avg_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_avg_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            cr_n0_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  #tf.zeros_like(phi)  
            cp_n1_ij=Sgi*((phi_n1_ij*d_dp_invBg_n1[...,1:-1,1:-1,:])+cr_n0_ij)

            a1_n1=C*kx_avg_i_h*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(1/dx_ij)
            a2_n1=C*ky_avg_j_h*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(1/dy_ij)
            a3_n1=C*kx_avg_ih*invBgug_avg_n1_ih*(1/dx_avg_ih)*(1/dx_ij)  
            a4_n1=C*ky_avg_jh*invBgug_avg_n1_jh*(1/dy_avg_jh)*(1/dy_ij)
            a5_n1=(1/D)*(cp_n1_ij/(tstep))
            
            b1_n1=C*kx_avg_i_h*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(dz_ij*dy_ij)
            b2_n1=C*ky_avg_j_h*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(dz_ij*dx_ij)
            b3_n1=C*kx_avg_ih*invBgug_avg_n1_ih*(1/dx_avg_ih)*(dz_ij*dy_ij)
            b4_n1=C*ky_avg_jh*invBgug_avg_n1_jh*(1/dy_avg_jh)*(dz_ij*dx_ij)
            
            well_wt=(tf.cast((q_well_idx==1),model.dtype)*no_wells)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            tsf_wt=tf.cast(xn0[3]<tsn['Time'],model.dtype)*1.+tf.cast(xn0[3]>=tsn['Time'],model.dtype)*tsn['Shift_Fac']
            # Calculate the maximum bottomhole pressure rate
            #pwf_n0_ij
            #tf.print('d_dp_invBg_n1\n',d_dp_invBg_n1,'InvBg_n0\n',invBg_n0,'InvBg_n1\n',invBg_n1,'InvUg_n1\n',invug_n1,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
            #tf.print('TIMESTEP\n',tstep,'(1/D)*(cp_n1_ij/tstep)\n','\na1\21`n',tf.reduce_max(a1),'\na2\n',tf.reduce_max(a2),'\na3\n',tf.reduce_max(a3),'\na4\n',tf.reduce_max(a4),'\na5\n',tf.reduce_max(a5),output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )

            #minbhp_well_idx=tf.cast((y[-2]>0.)&(y[-2]<model.cfd_type['Init_Grate'][0]),model.dtype)
            #dom=dv*((-a1_n1*p_n1_i_1)+(-a2_n1*p_n1_j_1)+((a1_n1+a2_n1+a3_n1+a4_n1)*p_n1_ij)+(-a3_n1*p_n1_i1)+(-a4_n1*p_n1_j1)+(q_n1_ij/dv)+(Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep))))
            dom=well_wt*dv*((-a1_n1*p_n1_i_1)+(-a2_n1*p_n1_j_1)+((a1_n1+a2_n1+a3_n1+a4_n1+a5_n1)*p_n1_ij)+(-a3_n1*p_n1_i1)+(-a4_n1*p_n1_j1)+(q_n1_ij/dv)-(a5_n1*p_n0_ij)) #+((a1+a2+a3+a4)*trn_err_tstep_fac[0])
            #tf.print('DOM_LOSS\n',dom,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
            #=======================================DBC Solution============================================================
            # Solution on the surface and same as pressure
            dbc=tf.zeros_like(dom)                 # Set at zero for now!
            #=======================================NBC Solution =========================================================== 
            # pwf=p_n1_ij-((well_block_rates*minbhp_well_idx)/(2*(22/7)*kx_ij*dz_ij*C))*(tf.math.log(ro/rw))*(1/(invBgug_n1_ij))  
            # nbc=minbhp_well_idx*(model.cfd_type['Min_BHP']-pwf) 
            nbc=tf.zeros_like(dom)                 # Set at zero for now!
            #===================================IBC_NEUMANN Solution========================================================
            well_block_rates_n1=(b1_n1*p_n1_i_1)+(b2_n1*p_n1_j_1)+(-(b1_n1+b2_n1+b3_n1+b4_n1)*p_n1_ij)+(b3_n1*p_n1_i1)+(b4_n1*p_n1_j1)
            #regu=0.0005*tf.linalg.global_norm(model.trainable_variables)**2
            ibc_n=(no_wells)*q_well_idx*(well_block_rates_n1-q_n1_ij)           # 
            #===================================Material Balance Check==============================================
            t_pss=15; 
            tpss_idx_hf=normalize(model,t_pss,stat_idx=3,compute=True)
            # Compute the radius of investigation
            #===========================================================================================================================================
            cgi=(model.cfd_type['Init_InvBg']*model.cfd_type['Init_DinvBg']+model.cf)   
            #cpt_t0_ij=Sgi*(phi_n1_ij*(1./model.cfd_type['Pi'])*(1.))+model.cf_ij  #cpr_t0_ij=Sgi*(phi_n1_ij*(1./tf.reduce_mean(pi))*0.999)+model.cf_ij
            cpt_t0_ij=(Sgi*cgi+model.cf_ij)*phi_n1_ij
            kx_mean_harmonic_ij=tf.math.reciprocal(tf.math.reduce_mean(tf.math.reciprocal(kx_ij),axis=[1,2,3],keepdims=True))  #Reciprocal of the arithmetic mean of the reciprocals
            rad_inv_n1_ij_transient=tf.math.sqrt(tf.math.maximum((kx_mean_harmonic_ij*tf.math.divide_no_nan((tn0*24.*model.cfd_type['Init_Invug']),(948.*cpt_t0_ij))),0)) #model.cfd_type['Init_Invug']
            rad_inv_ratio=tf.minimum((((22/7)*(rad_inv_n1_ij_transient**2)*no_wells)/area_res),1.)#tf.shape(model.cfd_type['Init_Grate']
            rad_inv_ratio_1well=tf.math.tanh((((22/7)*(rad_inv_n1_ij_transient**2)*1.)/area_res))
            
            rad_inv_t1_ij=tf.math.sqrt(tf.math.maximum((kx_ij*tf.math.divide_no_nan((1.*24.*model.cfd_type['Init_Invug']),(948.*cpt_t0_ij))),0.)) #model.cfd_type['Init_Invug']
            rad_inv_t1_ratio=tf.math.tanh(((22/7)*(rad_inv_t1_ij**2)*no_wells)/area_res)
            rad_inv_wblk_ratio=tf.math.tanh((area_ij/area_res)*no_wells)

            #t_pss_fac=(tf.cast(step_fac>=0.025,model.dtype)*step_fac)+(tf.cast(step_fac<0.025,model.dtype)*-step_fac)
            rad_inv_n1_ij=tf.math.sqrt((area_res/no_wells)/(22/7)); rad_inv_n1_ij_1well=tf.math.sqrt((area_res/1.)/(22/7)); rad_inv_wblk_ij=tf.math.sqrt((area_ij)/(22/7))
            t_pss_n1_ij=((948.*cpt_t0_ij*((rad_inv_n1_ij**2)+(rad_inv_n1_ij_1well**2))*0.50*(1+0.05))/(24.*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            rad_res=tf.math.sqrt((area_res/1.)/(22/7))
            
            
            #breakpoint()
            t_pss_n1_ij_min=((948.*cpt_t0_ij*rad_inv_n1_ij**2)/(24.*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            t_pss_n1_ij_max=((948.*cpt_t0_ij*rad_inv_n1_ij_1well**2)/(24.*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            
            #t_pss_n1_ij_max=((0.05*cpt_t0_ij*area_res)/(24.*0.0002637*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            #t_pss_n1_ij_max*=tf.math.abs(tf.random.normal((),mean=1.,stddev=(tstep/t_pss_n1_ij_max),dtype=dt_type))
            
            t_pss_wblk_ij=((948.*cpt_t0_ij*rad_inv_wblk_ij**2)/(24.*kx_ij*model.cfd_type['Init_Invug']))
            tn_pss_n1_ij_min=normalize(model,t_pss_n1_ij_min,stat_idx=3,compute=True)
            tn_pss_n1_ij_max=normalize(model,t_pss_n1_ij_max,stat_idx=3,compute=True)
            tn_pss_wblk_ij=normalize(model,t_pss_wblk_ij,stat_idx=3,compute=True)
            tn_pss_n1_ij_max_norm_diff=normalize_diff(model,t_pss_n1_ij_max,stat_idx=3,compute=True)
            
            dp_dt_pss=tf.math.abs(0.234*(-tf.reduce_sum(q_n1_ij,axis=[1,2,3],keepdims=True)/tf.reduce_sum(dv*Sgi*((phi_n1_ij*model.cfd_type['Init_DinvBg'])+(model.phi_0_ij*model.cf*model.cfd_type['Init_InvBg']))*(1/(D*tstep)),axis=[1,2,3],keepdims=True)))
            dp_dt_max=(1/tstep)*q_n1_ij/(2*(22/7)*kx_ij*dz_ij*C*(invBgug_n0_ij)*(1./tf.math.log(ro/rw)))
            dp_dt=(p_n1_ij-p_n0_ij)/tstep
            #tf.print('dp_dt_pss\n',dp_dt_pss,'dp_dt\n',dp_dt,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/dp_dt.out" )
            #==========================================================================================================================================
            tun_p=0.001 
            p_tpss=p_min#tf.reduce_min(tf.math.abs(p_n0_ij),axis=[1,2,3],keepdims=True)#p_min#tf.maximum(out_tpss[0],p_min) #p_min 
            #t_pss_n1_ij_max=nonormalize(model,step_fac,stat_idx=3,compute=True)
            pss_idx=tf.math.tanh(tf.math.divide_no_nan(tf.math.log(tf.math.abs(p_n0_ij)/p_tpss),tf.math.log(p_max/p_tpss))*(-1.)+1.)   #erf((sqrt(pi)/2)
            #pss_idx=tf.math.tanh(tf.math.divide_no_nan((tf.math.abs(p_n0_ij)-p_tpss),(p_max-p_tpss))*(-1.)+1.)   #erf((sqrt(pi)/2)
            #pss_idx=tf.math.tanh(tf.math.exp(tf.math.divide_no_nan(tf.math.log(tn0/1.),tf.math.log(t_pss_n1_ij_max/1.))*(tf.math.log(1./tun_p))+tf.math.log(tun_p)))

            rnd_trn_err=((4*eps)/min_tstep**2)+((prec*min_tstep**2)/12)

            '''tmbc_idx=tf.cast((xn0[3]>tn_pss_n1_ij_max)&(xn0[3]>(t0_norm))&(d2p_dt2_abs<=prec),model.dtype)\
                +tf.cast((xn0[3]>tn_pss_n1_ij_max)&(xn0[3]>(t0_norm))&(d2p_dt2_abs>prec),model.dtype)*pss_idx\
                +tf.cast((xn0[3]>tn_pss_n1_ij_min)&(xn0[3]<=tn_pss_n1_ij_max)&(xn0[3]>(t0_norm))&(d2p_dt2_abs<=prec),model.dtype)\
                +tf.cast((xn0[3]>tn_pss_n1_ij_min)&(xn0[3]<=tn_pss_n1_ij_max)&(xn0[3]>(t0_norm))&(d2p_dt2_abs>prec),model.dtype)*pss_idx\
                +tf.cast((xn0[3]<=tn_pss_n1_ij_min)&(d2p_dt2_abs>prec)&(xn0[3]>(t0_norm)),model.dtype)*pss_idx
            '''
            #div_abs=tf.reduce_sum(dv*((a1_n1*p_n1_i_1)+(a2_n1*p_n1_j_1)+(-(a1_n1+a2_n1+a3_n1+a4_n1)*p_n1_ij)+(a3_n1*p_n1_i1)+(a4_n1*p_n1_j1)),axis=[1,2,3])
            std=tf.math.reduce_std(tf.math.abs(d2p_dt2),axis=[1,2,3])
            d2p_dt2_=(tf.math.divide_no_nan((p_n1_ij-2*(p_n0_ij)+p_n_1_ij),(tstep)**2))
            
            #tn_pss_n1_ij_max=model.cfd_type['Norm_Limits'][0]+tmax_norm_diff*tf.math.abs(tf.random.normal((),mean=0.5,stddev=0.5/3,dtype=tf.dtypes.float32,seed=model.cfd_type['Seed'])) 
            eps_pss=tf.math.maximum(tf.math.abs(tf.random.normal((),mean=0.,stddev=1e-2,dtype=dt_type)),1e-4 )
            #model.eps_pss['Average'].update_state(eps_pss)
            #model.eps_pss.assign(tf.reduce_mean(tf.cast(xn0[3]<normalize(model,0.005,stat_idx=3,compute=True),model.dtype)*tf.math.abs(d2p_dt2)+tf.cast(xn0[3]>=normalize(model,0.005,stat_idx=3,compute=True),model.dtype)*model.eps_pss))
            
            #tmbc_idx=tf.cast((xn0[3]>(tn_pss_n1_ij_max)),model.dtype)\
             #       +tf.cast((xn0[3]>t1_norm)&(xn0[3]<=tn_pss_n1_ij_max)&(tf.math.abs(d2p_dt2_)<=eps_pss),model.dtype)\
              #      +tf.cast((xn0[3]>t1_norm)&(xn0[3]<=tn_pss_n1_ij_max)&(tf.math.abs(d2p_dt2_)>eps_pss),model.dtype)*pss_idx
                    #+tf.cast((xn0[3]>t0_norm)&(xn0[3]<=t1_norm),model.dtype)*pss_idx'''

            #tpss_idx=tf.cast(xn0[3]>(tn_pss_n1_ij_max),model.dtype)*tf.math.tanh(1.)+tf.cast(xn0[3]<=(tn_pss_n1_ij_max),model.dtype)*0.
            #tpss_idx=tf.cast((tf.math.abs(dp_dt)<=dp_dt_pss),model.dtype)*1.+tf.cast((tf.math.abs(dp_dt)>dp_dt_pss),model.dtype)*pss_idx

            #tmbc_idx=tf.cast((xn0[3]>(t1_norm))&(tf.math.abs(d2p_dt2)<=eps_pss),model.dtype)*1.\
                             # +tf.cast((xn0[3]>(t0_norm))&(tf.math.abs(d2p_dt2)>eps_pss),model.dtype)*tf.math.maximum(eps_pss,pss_idx)

            tmbc_idx=tf.cast((xn0[3]>(t0_norm))&(tf.math.abs(d2p_dt2)<=eps_pss),model.dtype)*0.75\
                              +tf.cast((xn0[3]>(t0_norm))&(tf.math.abs(d2p_dt2)>eps_pss),model.dtype)*pss_idx
            
            
            #tmbc_idx=tf.cast(xn0[3]>t10_norm,model.dtype)+tf.cast(xn0[3]<=t10_norm,model.dtype)*0.
            tmbc_t0_idx=tf.reduce_mean(tf.cast((xn0[3]>=t0_norm)&(xn0[3]<=t1_norm),model.dtype),axis=[1,2,3])
            tmbc_well_idx=(tf.reduce_sum(q_well_idx*tmbc_idx,axis=[1,2,3],keepdims=False)/no_wells) #tf.reduce_mean(tmbc_idx,axis=[1,2,3])#
            mbc=tmbc_well_idx*(-tf.reduce_sum(q_n1_ij,axis=[1,2,3],keepdims=False)-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)),axis=[1,2,3],keepdims=False))
            #mbc=tmbc_well_idx*(-tf.reduce_sum(q_n1_ij,axis=[1,2,3])-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n_1_ij)*(1/(2*D*tstep)),axis=[1,2,3]))
            #mbc=tmbc_well_idx*(-tf.reduce_sum(q_n1_ij,axis=[1,2,3])-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n2_ij-invBg_n0_ij)*(1/(D*min_tstep)),axis=[1,2,3]))
            #dom+=tf.ones_like(xn0[3])*tmbc_well_idx*(-tf.reduce_sum(q_n1_ij,axis=[1,2,3])-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)),axis=[1,2,3]))
            #mbc=dv*(((tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)),axis=[1,2,3],keepdims=False))/-tf.reduce_sum(q_n1_ij,axis=[1,2,3],keepdims=False))-tf.ones_like(xn0[3]))
            #===================================Cumulative Material Balance Check===================================
            #model.cum['Gas_Obs'].assign_add(tf.reduce_sum(q_n1_ij)*tstep)
            #cq_n1_ij=q_n1_ij*tn1
            #cmbc=tmbc_well_idx*(-tf.reduce_sum(cq_n1_ij,axis=[1,2,3])-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_t0_ij)*(1/(D)),axis=[1,2,3]))
            cmbc=tf.zeros_like(dom)
            #==========================================Initial Condition============================================
            ic=(model.cfd_type['Pi']-out_t0[0])
            #ic=tf.stack([ic,ic])
            #ic_wt=0.5  #0.75
            #ic=((1-ic_wt)*(1-q_well_idx)*(model.cfd_type['Pi']-out_t0[0]))+(ic_wt*q_well_idx*(model.cfd_type['Pi']-out_t0[0]))
            #ic=(no_wells*(1-q_well_idx)*(model.cfd_type['Pi']-out_t0[0]))+(q_well_idx*(model.cfd_type['Pi']-out_t0[0]))
            #dbc=q_well_idx*(model.cfd_type['Pi']-out_t0[0])
            #==========================================Rate Optimization Solution)================================== 
            # This is an optimization for the variable rate-pressure 
            # ======================================================================================================
            s1_n1_ij=out_n1[-1][...,1:-1,1:-1,:]
            
            # Initial Timestep, t1
            xn_t1=list(xi)
            xn_t1[3]=tf.ones_like(xn_t1[0])*t1_norm
            out_t1=model(xn_t1)
            
            #=======================================================================================================
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom)#q_well_idx*(q_n1_ij-out_t1[-2])          # Rate output index: -2
            qrc_3=tf.zeros_like(dom)#q_well_idx*(1e-8*-out_t0[-2])
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            #=============================================================================
            return dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]
        
        def physics_error_gas_oil(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            out_n0=model(xn0, training=True)
            #out_n0,d2p_dt=second_order_derivative_AD(model,x,y)
            out_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8]])
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]
            Sg_n0_ij=out_n0[1][...,1:-1,1:-1,:]
            So_n0_ij=(1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sg_n0_ij)#out_n0[2][...,1:-1,1:-1,:]
            invBg_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBo_n0_ij=out_n0[4][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[5][...,1:-1,1:-1,:]
            invuo_n0_ij=out_n0[6][...,1:-1,1:-1,:]
            Rs_n0_ij=out_n0[7][...,1:-1,1:-1,:]
            Rv_n0_ij=out_n0[8][...,1:-1,1:-1,:]
            
            invBgug_n0_ij=(out_n0[3]*out_n0[5])[...,1:-1,1:-1,:]
            invBouo_n0_ij=(out_n0[4]*out_n0[6])[...,1:-1,1:-1,:]
            RsinvBo_n0_ij=(out_n0[7]*out_n0[4])[...,1:-1,1:-1,:]
            RvinvBg_n0_ij=(out_n0[8]*out_n0[3])[...,1:-1,1:-1,:]
            
            # Initial Condition
            xn_t0=list(xi)
            t0_idx=tf.cast(xn0[3]<normalize(model,0.005,stat_idx=3,compute=True),model.dtype)
            xn_t0[3]+=-xn_t0[3]+(tf.ones_like(xn_t0[0])*t0_norm)#+tf.random.normal(shape=tf.shape(xn_t0[0]), mean=0.0, stddev=0.0005, dtype=dt_type)
            out_t0=model(xn_t0)
            p_t0_ij=out_t0[0]
            Sg_t0_ij=out_t0[1]
            So_t0_ij=(1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sg_t0_ij)
            '''
            invBg_t0_ij=out_t0[3]
            invBo_t0_ij=out_t0[4]
            invug_t0_ij=out_t0[5]
            invuo_t0_ij=out_t0[6]
            invBgug_t0_ij=(out_t0[3]*out_t0[5]) 
            RsinvBo_t0_ij=(out_t0[7]*out_t0[4])
            RvinvBg_t0_ij=(out_t0[8]*out_t0[3])'''
            
            invBg_t0_ij,invBo_t0_ij,invug_t0_ij,invuo_t0_ij,Rs_t0_ij,Rv_t0_ij=model.cfd_type['Init_InvBg'],model.cfd_type['Init_InvBo'],\
                model.cfd_type['Init_Invug'],model.cfd_type['Init_Invuo'],model.cfd_type['Init_Rs'],model.cfd_type['Init_Rv']
            invBgug_t0_ij=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
            RsinvBo_t0_ij=model.cfd_type['Init_Rs']*model.cfd_type['Init_InvBo']
            RvinvBg_t0_ij=model.cfd_type['Init_Rv']*model.cfd_type['Init_InvBg']
            # ============================Compute the tuning model network============================================
            # At n+1, the timestep interval is normalized before used as a feature
            #p_n0_ij_norm=(((p_n0_ij-p_min)/(model.cfd_type['Pi']-p_min))*(model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0]))+model.cfd_type['Norm_Limits'][0]
            trn_err_tstep_fac=model.trn_model([xn0[3],(xn0[5])])  #[p_n0_ij,tn0] [xn0[3],(xn0[5])] [xn0[3],p_n0_ij_norm]  [tn0,(kx/(out_n0[2]*out_n0[3]))]
            prec=0.01#tf.reduce_mean(trn_err_tstep_fac[0],axis=[1,2,3],keepdims=True) #5e-3       # 1e-4 to 1e-3
            tstep_fac=tf.reduce_mean(trn_err_tstep_fac[0],axis=[1,2,3],keepdims=True)
            step_fac=tf.reduce_mean(trn_err_tstep_fac[1],axis=[1,2,3],keepdims=True)
            min_tstep=(48.*eps/prec)**(1/4)#(model.cfd_type['Timestep'])*step_fac
            min_tstep_norm=normalize_diff(model,min_tstep,stat_idx=3,compute=True)

            xn2=list(xi)    
            xn2[3]+=min_tstep_norm#tstep_norm*step_fac #tstep_norm
            out_n2=model(xn2, training=True)
            p_n2_ij=out_n2[0]
            Sg_n2_ij=out_n2[1]
            So_n2_ij=(1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sg_n2_ij)
            invBg_n2_ij=out_n2[3]
            invBgug_n2_ij=out_n2[3]*out_n2[5]
            invBouo_n2_ij=out_n2[4]*out_n2[6]
            #phi_n2_ij=phi[...,1:-1,1:-1,:]
            
            xn_2=list(xi)
            xn_2[3]-=min_tstep_norm#tstep_norm*step_fac #tstep_norm
            out_n_2=model(xn_2, training=True)
            p_n_2_ij=out_n_2[0]
            Sg_n_2_ij=out_n_2[1]
            So_n_2_ij=(1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sg_n_2_ij)
            invBg_n_2_ij=out_n_2[3]
            invBgug_n_2_ij=out_n_2[3]*out_n_2[5]
            invBouo_n_2_ij=out_n_2[4]*out_n_2[6]
            
            krog_n2_ij,krgo_n2_ij=model.kr_gas_oil(So_n2_ij,Sg_n2_ij)              #Entries: oil, and gas
            krog_n_2_ij,krgo_n_2_ij=model.kr_gas_oil(So_n_2_ij,Sg_n_2_ij)              #Entries: oil, and gas
            #=========================================================================================================
            #Compute the second order time derivatives to check for PSS
            #_,pwf_n_2_ij=compute_rate_bhp(p_n_2_ij,invBgug_n_2_ij,p_t0_ij,invBgug_t0_ij,krgo_n_2_ij,q_n1_ij)
            #_,pwf_n2_ij=compute_rate_bhp(p_n2_ij,invBgug_n2_ij,p_t0_ij,invBgug_t0_ij,krgo_n2_ij,q_n1_ij)
            d2p_dt2=(tf.math.divide_no_nan((p_n2_ij-2*(p_n0_ij)+p_n_2_ij),(min_tstep)**2))
            #d2p_dt2=(tf.math.divide_no_nan((pwf_n2_ij-2*(pwf_n0_ij)+pwf_n_2_ij),(min_tstep)**2))

            cgi=(model.cfd_type['Init_InvBg']*model.cfd_type['Init_DinvBg']+model.cf)   
            dp_dt=tf.math.divide_no_nan((p_n2_ij-p_n0_ij),(min_tstep))
            dp_max=q_n1_ij/(2*(22/7)*kx_ij*dz_ij*C*(model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug'])*(1./tf.math.log(ro/rw)))
            #dp_max=tf.reduce_sum(D*q_n1_ij*(1/model.cfd_type['Init_InvBg'])*1,axis=[1,2,3],keepdims=True)/tf.reduce_sum(dv*phi_n1_ij*Sgi*cgi,axis=[1,2,3],keepdims=True)
            
            dt_t0=tf.minimum(tf.math.abs(tf.math.divide_no_nan(dp_max,dp_dt)),365.)
            dt_t0=tf.reduce_sum(dt_t0,axis=[1,2,3],keepdims=True)/no_wells
            #tf.print('dt_t0\n',dt_t0,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/dt_t0.out" )
            #=========================================================================================================
            #Define a timestep
            #tstep=model.cfd_type['Timestep']
            tstep=(tf.cast(xn0[3]<t1_norm,model.dtype)*model.cfd_type['Timestep'])+\
                        (tf.cast((xn0[3]>=t1_norm)&(xn0[3]<t10_norm),model.dtype)*(model.cfd_type['Timestep']*1.))+\
                        (tf.cast((xn0[3]>=t10_norm)&(xn0[3]<tsn['Time']),model.dtype)*model.cfd_type['Timestep'])+\
                        (tf.cast((xn0[3]>=tsn['Time']),model.dtype)*model.cfd_type['Timestep']*tsn['Shift_Fac'])#*(tshift_fac) tsn['Shift_Fac']
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)

            # Update the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)
            out_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8]]) 
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]
            Sg_n1_ij=out_n1[1][...,1:-1,1:-1,:]
            So_n1_ij=(1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sg_n1_ij)#out_n1[2][...,1:-1,1:-1,:]
            invBg_n1_ij=out_n1[3][...,1:-1,1:-1,:]
            invBo_n1_ij=out_n1[4][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[5][...,1:-1,1:-1,:]
            invuo_n1_ij=out_n1[6][...,1:-1,1:-1,:]
            Rs_n1_ij=out_n1[7][...,1:-1,1:-1,:]
            Rv_n1_ij=out_n1[8][...,1:-1,1:-1,:]
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
            #=============================Relative Permeability Function==============================================
            krog_n1,krgo_n1=model.kr_gas_oil(out_n1[2],out_n1[1])              #Entries: oil, and gas
            krgo_n1_ij=krgo_n1[...,1:-1,1:-1,:]
            krog_n1_ij=krog_n1[...,1:-1,1:-1,:]
            #=========================================================================================================
            # Update the timestep (n-1)
            xn_1=list(xi)
            xn_1[3]-=tstep_norm #tstep_norm
            out_n_1=model(xn_1, training=True)
            p_n_1_ij=out_n_1[0]
            invBg_n_1_ij=out_n_1[2]
            #IC
            #====================================================================================================================
            # Compute bottom hole pressure and rates
            _,pwf_n1_ij=compute_rate_bhp(p_n1_ij,invBgug_n1_ij,p_t0_ij,invBgug_t0_ij,krgo_n1_ij,q_n1_ij)

            qg_n1_ij=q_n1_ij
            qo_n1_ij=(qg_n1_ij/(krgo_n1_ij*invBgug_n1_ij))*(krog_n1_ij*invBouo_n1_ij)     #tf.zeros_like(qg_n1_ij)      #To be updated
            qo_n1_ij=tf.cast((pwf_n1_ij<=model.cfd_type['Dew_Point'])&(pwf_n1_ij>0.),model.dtype)*qo_n1_ij
            #====================================================================================================================
            #Define pressure variables 
            # In the absence of capillary pressures, pg=po
            pg_n0_ij=po_n0_ij=out_n0[0][...,1:-1,1:-1,:];
            pg_n1_ij=po_n1_ij=out_n1[0][...,1:-1,1:-1,:];
            pg_n1_i1=po_n1_i1=out_n1[0][...,1:-1,2:,:]; pg_n1_i_1=po_n1_i_1=out_n1[0][...,1:-1,:-2,:]    #Check grid rotation
            pg_n1_j1=po_n1_j1=out_n1[0][...,2:,1:-1,:]; pg_n1_j_1=po_n1_j_1=out_n1[0][...,:-2,1:-1,:]
            #====================================================================================================================
            tdew_idx=tf.cast((pg_n1_ij<=model.cfd_type['Dew_Point'])&(pg_n1_ij>0.),model.dtype)
            #====================================================================================================================
            # Compute the chord slopes for pressure and saturation. d_dp_Sg;d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision
            d_dpg_Sg_n1_ij=tf.math.divide_no_nan((out_n1[1]-out_n0[1]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_Sg_n1_ij=tf.math.divide_no_nan((out_n1[1]-out_n0[1]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpg_So_n1_ij=tf.math.divide_no_nan((out_n1[2]-out_n0[2]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_So_n1_ij=tf.math.divide_no_nan((out_n1[2]-out_n0[2]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            breakpoint()
            d_dpg_invBg_n1_ij=model.PVT(pg_n0_ij)[1][0]#tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_invBg_n1_ij=model.PVT(po_n0_ij)[1][0]#tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpg_invBo_n1_ij=model.PVT(pg_n0_ij)[1][1]#tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_invBo_n1_ij=model.PVT(po_n0_ij)[1][1]#tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            d_dpg_RsinvBo_n1_ij=(model.PVT(pg_n0_ij)[0][4]*model.PVT(pg_n0_ij)[1][1])+(model.PVT(pg_n0_ij)[0][1]*model.PVT(pg_n0_ij)[1][4]) #tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_RsinvBo_n1_ij=(model.PVT(po_n0_ij)[0][4]*model.PVT(po_n0_ij)[1][1])+(model.PVT(po_n0_ij)[0][1]*model.PVT(po_n0_ij)[1][4]) #tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpg_RvinvBg_n1_ij=(model.PVT(pg_n0_ij)[0][5]*model.PVT(pg_n0_ij)[1][0])+(model.PVT(pg_n0_ij)[0][0]*model.PVT(pg_n0_ij)[1][5]) #tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            d_dpo_RvinvBg_n1_ij=(model.PVT(po_n0_ij)[0][5]*model.PVT(po_n0_ij)[1][0])+(model.PVT(po_n0_ij)[0][0]*model.PVT(po_n0_ij)[1][5]) #tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            #Average function value weighting for pressure-dependent terms
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
            
            invBgug_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            invBouo_n1_ih=(invBouo_n1_i1+invBouo_n1_ij)/2.; invBouo_n1_i_h=(invBouo_n1_ij+invBouo_n1_i_1)/2.
            invBouo_n1_jh=(invBouo_n1_j1+invBouo_n1_ij)/2.; invBouo_n1_j_h=(invBouo_n1_ij+invBouo_n1_j_1)/2.
            
            RvinvBgug_n1_ih=(RvinvBgug_n1_i1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_i_h=(RvinvBgug_n1_ij+RvinvBgug_n1_i_1)/2.
            RvinvBgug_n1_jh=(RvinvBgug_n1_j1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_j_h=(RvinvBgug_n1_ij+RvinvBgug_n1_j_1)/2.
            RsinvBouo_n1_ih=(RsinvBouo_n1_i1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_i_h=(RsinvBouo_n1_ij+RsinvBouo_n1_i_1)/2.
            RsinvBouo_n1_jh=(RsinvBouo_n1_j1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_j_h=(RsinvBouo_n1_ij+RsinvBouo_n1_j_1)/2.

            # Upstream weighting for saturation-dependent terms  -- only the upstream weighting method works for linearization of saturation dependent terms
            # Average function value weighting gives erroneus results on reservoir simulators (Abou-Kassem)
            krgo_n1_i1=krgo_n1[...,1:-1,2:,:]; krgo_n1_i_1=krgo_n1[...,1:-1,:-2,:]
            krgo_n1_j1=krgo_n1[...,2:,1:-1,:]; krgo_n1_j_1=krgo_n1[...,:-2,1:-1,:]    #(j,i,)
            
            krog_n1_i1=krog_n1[...,1:-1,2:,:]; krog_n1_i_1=krog_n1[...,1:-1,:-2,:]
            krog_n1_j1=krog_n1[...,2:,1:-1,:]; krog_n1_j_1=krog_n1[...,:-2,1:-1,:]
            
            #For i to be upstream (i+1), pot_i<=0; i to be downstream (i+1), pot_i>0
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

            cprgg_n1_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  #tf.zeros_like(phi)
            cprgo_n1_ij=(model.phi_0_ij*model.cf*RsinvBo_n0_ij)  #tf.zeros_like(phi)
            cproo_n1_ij=(model.phi_0_ij*model.cf*invBo_n0_ij)  #tf.zeros_like(phi)
            cprog_n1_ij=(model.phi_0_ij*model.cf*RvinvBg_n0_ij)  #tf.zeros_like(phi)
            
            # Gas Phase Flow
            agg_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*invBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            agg_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*invBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            ago_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*RsinvBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            ago_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*RsinvBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            agg_n1_ih=C*kx_avg_ih*(krgo_n1_ih*invBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            agg_n1_jh=C*ky_avg_jh*(krgo_n1_jh*invBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            ago_n1_ih=C*kx_avg_ih*(krog_n1_ih*RsinvBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            ago_n1_jh=C*ky_avg_jh*(krog_n1_jh*RsinvBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpgg_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_invBg_n1_ij)+cprgg_n1_ij))
            cpgo_n1_ij=(1/(D*tstep))*((phi_n1_ij*RsinvBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_RsinvBo_n1_ij)+cprgo_n1_ij))

            qfg_n1_ij=qg_n1_ij
            qdg_n1_ij=tdew_idx*Rs_n1_ij*qo_n1_ij
            
            # Oil Phase Flow
            aoo_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*invBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aoo_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*invBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aog_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*RvinvBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aog_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*RvinvBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aoo_n1_ih=C*kx_avg_ih*(krog_n1_ih*invBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aoo_n1_jh=C*ky_avg_jh*(krog_n1_jh*invBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            aog_n1_ih=C*kx_avg_ih*(krgo_n1_ih*RvinvBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aog_n1_jh=C*ky_avg_jh*(krgo_n1_jh*RvinvBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpoo_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_invBo_n1_ij)+cproo_n1_ij))
            cpog_n1_ij=(1/(D*tstep))*((phi_n1_ij*RvinvBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_RvinvBg_n1_ij)+cprog_n1_ij))

            qfo_n1_ij=tdew_idx*qo_n1_ij
            qvo_n1_ij=Rv_n1_ij*qg_n1_ij

            #            
            #dom_divq_g=dv*((-agg_n1_i_h*pg_n1_i_1)+(-agg_n1_j_h*pg_n1_j_1)+(-ago_n1_i_h*po_n1_i_1)+(-ago_n1_j_h*po_n1_j_1)+\
             #         ((agg_n1_i_h+agg_n1_j_h+agg_n1_ih+agg_n1_jh)*pg_n1_ij)+((ago_n1_i_h+ago_n1_j_h+ago_n1_ih+ago_n1_jh)*po_n1_ij)+\
              #        ((-agg_n1_ih*pg_n1_i1)+(-agg_n1_jh*pg_n1_j1)+(-ago_n1_ih*po_n1_i1)+(-ago_n1_jh*po_n1_j1))+\
               #        ((qfg_n1_ij+qdg_n1_ij)/dv))

            #dom_acc_g=dv*((cpgg_n1_ij*(pg_n1_ij-pg_n0_ij))+(cpgo_n1_ij*(po_n1_ij-po_n0_ij))) 
            dom_divq_gg=dv*((-agg_n1_i_h*pg_n1_i_1)+(-agg_n1_j_h*pg_n1_j_1)+((agg_n1_i_h+agg_n1_j_h+agg_n1_ih+agg_n1_jh)*pg_n1_ij)+\
                      (-agg_n1_ih*pg_n1_i1)+(-agg_n1_jh*pg_n1_j1)+(qfg_n1_ij/dv))
                
            dom_divq_go=tdew_idx*dv*((-ago_n1_i_h*po_n1_i_1)+(-ago_n1_j_h*po_n1_j_1)+((ago_n1_i_h+ago_n1_j_h+ago_n1_ih+ago_n1_jh)*po_n1_ij)+\
                      (-ago_n1_ih*po_n1_i1)+(-ago_n1_jh*po_n1_j1)+(qdg_n1_ij/dv))
                
            dom_acc_gg=dv*(cpgg_n1_ij*(pg_n1_ij-pg_n0_ij))
            dom_acc_go=tdew_idx*dv*(cpgo_n1_ij*(po_n1_ij-po_n0_ij)) 
            
            dom_gg=dom_divq_gg+dom_acc_gg
            dom_go=dom_divq_go+dom_acc_go
            dom_g=dom_gg+dom_go

            #dom_divq_o=dv*((-aoo_n1_i_h*po_n1_i_1)+(-aoo_n1_j_h*po_n1_j_1)+(-aog_n1_i_h*pg_n1_i_1)+(-aog_n1_j_h*pg_n1_j_1)+\
             #         ((aoo_n1_i_h+aoo_n1_j_h+aoo_n1_ih+aoo_n1_jh)*po_n1_ij)+((aog_n1_i_h+aog_n1_j_h+aog_n1_ih+aog_n1_jh)*pg_n1_ij)+\
              #        ((-aoo_n1_ih*po_n1_i1)+(-aoo_n1_jh*po_n1_j1)+(-aog_n1_ih*pg_n1_i1)+(-aog_n1_jh*pg_n1_j1))+\
               #        ((qfo_n1_ij+qvo_n1_ij)/dv))
            #dom_acc_o=dv*((cpoo_n1_ij*(po_n1_ij-po_n0_ij))+(cpog_n1_ij*(pg_n1_ij-pg_n0_ij))) 

            dom_divq_oo=tdew_idx*dv*((-aoo_n1_i_h*po_n1_i_1)+(-aoo_n1_j_h*po_n1_j_1)+((aoo_n1_i_h+aoo_n1_j_h+aoo_n1_ih+aoo_n1_jh)*po_n1_ij)+\
                      (-aoo_n1_ih*po_n1_i1)+(-aoo_n1_jh*po_n1_j1)+(qfo_n1_ij/dv))
                
            dom_divq_og=dv*((-aog_n1_i_h*pg_n1_i_1)+(-aog_n1_j_h*pg_n1_j_1)+((aog_n1_i_h+aog_n1_j_h+aog_n1_ih+aog_n1_jh)*pg_n1_ij)+\
                      (-aog_n1_ih*pg_n1_i1)+(-aog_n1_jh*pg_n1_j1)+(qvo_n1_ij/dv))
                
            dom_acc_oo=tdew_idx*dv*(cpoo_n1_ij*(po_n1_ij-po_n0_ij)) 
            dom_acc_og=dv*(cpog_n1_ij*(pg_n1_ij-pg_n0_ij))
            
            dom_oo=dom_divq_oo+dom_acc_oo
            dom_og=dom_divq_og+dom_acc_og
            dom_o=dom_oo+dom_og

            well_wt=(tf.cast((q_well_idx==1),model.dtype)*1.)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            #tf.print('d_dpg_invBg_n1_ij\n',d_dpg_invBg_n1_ij,'InvBg_n0\n',invBg_n0_ij,'InvBg_n1\n',invBg_n1_ij,'InvUg_n1\n',invug_n1_ij,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/pre_pvt.out" )
            #tf.print('TIMESTEP\n',tstep,'(1/D)*(cp_n1_ij/tstep)\n','\na1\21`n',tf.reduce_max(a1),'\na2\n',tf.reduce_max(a2),'\na3\n',tf.reduce_max(a3),'\na4\n',tf.reduce_max(a4),'\na5\n',tf.reduce_max(a5),output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/pre_pvt.out" )
            #tf.print('d_dpg_invBg_n1_ij\n',d_dpg_invBg_n1_ij,'pg_n0_ij\n',pg_n0_ij,'InvBg_n0\n',invBg_n0_ij,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/pre_pvt.out" )

            dom=dom_g+dom_o
            #dom=[dom_g,dom_o]
            #=======================================DBC Solution============================================================
            # Solution on the surface and same as pressure
            dbc=tf.zeros_like(dom)                 # Set at zero for now!
            #=======================================NBC Solution =========================================================== 
            # pwf=p_n1_ij-((well_block_rates*minbhp_well_idx)/(2*(22/7)*kx_ij*dz_ij*C))*(tf.math.log(ro/rw))*(1/(invBgug_n1_ij))  
            # nbc=minbhp_well_idx*(model.cfd_type['Min_BHP']-pwf) 
            nbc=tf.zeros_like(dom)                 # Set at zero for now!
            #===================================IBC_NEUMANN Solution========================================================
            #regu=0.0005*tf.linalg.global_norm(model.trainable_variables)**2
            ibc_n=(no_wells)*q_well_idx*(dom_divq_gg+dom_divq_go+dom_divq_oo+dom_divq_og)           # 
            #ibc_n=(no_wells)*q_well_idx*([dom_divq_g,dom_divq_o]) 
            #===================================Material Balance Check==============================================
            t_pss=15; 
            tpss_idx_hf=normalize(model,t_pss,stat_idx=3,compute=True)
            # Compute the radius of investigation
            #===========================================================================================================================================
            #cpt_t0_ij=Sg_n0_ij*(phi_n1_ij*(1./model.cfd_type['Pi'])*(1.))+model.cf_ij  #cpr_t0_ij=Sg*(phi_n1_ij*(1./tf.reduce_mean(pi))*0.999)+model.cf_ij
            cpt_t0_ij=(Sgi*cgi+model.cf_ij)*phi_n1_ij
            kx_mean_harmonic_ij=tf.math.reciprocal(tf.math.reduce_mean(tf.math.reciprocal(kx_ij),axis=[1,2,3],keepdims=True))  #Reciprocal of the arithmetic mean of the reciprocals
            rad_inv_n1_ij_1well=tf.math.sqrt((area_res/1.)/(22/7))

            t_pss_n1_ij_max=((948.*cpt_t0_ij*rad_inv_n1_ij_1well**2)/(24.*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            #t_pss_n1_ij_max*=tf.math.abs(tf.random.normal((),mean=1.,stddev=(tstep/t_pss_n1_ij_max),dtype=dt_type))
            #t_pss_n1_ij_max=((0.05*cpt_t0_ij*area_res)/(24.*0.0002637*kx_mean_harmonic_ij*model.cfd_type['Init_Invug']))
            tn_pss_n1_ij_max=normalize(model,t_pss_n1_ij_max,stat_idx=3,compute=True)
            tn_pss_n1_ij_max_norm_diff=normalize_diff(model,t_pss_n1_ij_max,stat_idx=3,compute=True)

            #==========================================================================================================================================
            tun_p=0.001 
            p_tpss=p_min#tf.maximum(out_tpss[0],p_min) #p_min 
            #t_pss_n1_ij_max=nonormalize(model,step_fac,stat_idx=3,compute=True)
            pss_idx=tf.math.tanh(tf.math.divide_no_nan(tf.math.log(tf.math.abs(p_n0_ij)/p_tpss),tf.math.log(p_max/p_tpss))*(-1.)+1.)   #erf((sqrt(pi)/2)
            #pss_idx=tf.math.tanh(tf.math.divide_no_nan((tf.reduce_mean(tf.math.abs(p_n0_ij),axis=[1,2,3],keepdims=True)-p_tpss),(p_max-p_tpss))*(-1.)+1.)   #erf((sqrt(pi)/2)

            eps_pss=step_fac#tf.math.maximum(tf.math.abs(tf.random.normal((),mean=0.,stddev=1e-2,dtype=dt_type)),1e-4 )
            #model.eps_pss['Average'].update_state(eps_pss)

            tpss_idx=tf.cast(xn0[3]>(tn_pss_n1_ij_max),model.dtype)*1.+tf.cast(xn0[3]<=(tn_pss_n1_ij_max),model.dtype)*pss_idx
            tmbc_idx=tf.cast((xn0[3]>t0_norm)&(tf.math.abs(d2p_dt2)<=eps_pss),model.dtype)*1.\
                               +tf.cast((xn0[3]>t0_norm)&(tf.math.abs(d2p_dt2)>eps_pss),model.dtype)*pss_idx#model.eps_pss['Average'].result()
            
            kdims=False
            tmbc_well_idx=(tf.reduce_sum(q_well_idx*tmbc_idx,axis=[1,2,3],keepdims=kdims)/no_wells) #tf.reduce_mean(tmbc_idx,axis=[1,2,3])#
            mbc_gg=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*invBg_n1_ij)-(Sg_n0_ij*invBg_n0_ij))
            mbc_go=tdew_idx*dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*RsinvBo_n1_ij)-(So_n0_ij*RsinvBo_n0_ij))
            mbc_g=(-tf.reduce_sum(qfg_n1_ij+qdg_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_gg+mbc_go,axis=[1,2,3],keepdims=kdims))
                   
            mbc_oo=tdew_idx*dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*invBo_n1_ij)-(So_n0_ij*invBo_n0_ij))
            mbc_og=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*RvinvBg_n1_ij)-(Sg_n0_ij*RvinvBg_n0_ij))
            mbc_o=(-tf.reduce_sum(qfo_n1_ij+qvo_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_oo+mbc_og,axis=[1,2,3],keepdims=kdims))

            mbc=tmbc_well_idx*(mbc_g+mbc_o)
            #mbc=tf.zeros_like(dom)
            #===================================Cumulative Material Balance Check===================================
            #model.cum['Gas_Obs'].assign_add(tf.reduce_sum(q_n1_ij)*tstep)
            cmbc_gg=dv*(1/(D))*phi_n1_ij*((Sg_n1_ij*invBg_n1_ij)-(Sg_t0_ij*invBg_t0_ij))
            cmbc_go=tdew_idx*dv*(1/(D))*phi_n1_ij*((So_n1_ij*RsinvBo_n1_ij)-(So_t0_ij*RsinvBo_t0_ij))
            cmbc_g=((-tf.reduce_sum(qfg_n1_ij+qdg_n1_ij,axis=[1,2,3],keepdims=kdims)*tn1)-tf.reduce_sum(cmbc_gg+cmbc_go,axis=[1,2,3],keepdims=kdims))
                   
            cmbc_oo=tdew_idx*dv*(1/(D))*phi_n1_ij*((So_n1_ij*invBo_n1_ij)-(So_t0_ij*invBo_t0_ij))
            cmbc_og=dv*(1/(D))*phi_n1_ij*((Sg_n1_ij*RvinvBg_n1_ij)-(Sg_t0_ij*RvinvBg_t0_ij))
            cmbc_o=((-tf.reduce_sum(qfo_n1_ij+qvo_n1_ij,axis=[1,2,3],keepdims=kdims)*tn1)-tf.reduce_sum(cmbc_oo+cmbc_og,axis=[1,2,3],keepdims=kdims))

            #cmbc=tmbc_well_idx*(cmbc_g+cmbc_o)
            cmbc=tf.zeros_like(dom)
            #==========================================Initial Condition============================================
            ic_p=(model.cfd_type['Pi']-p_t0_ij)
            ic_Sg=0.5*(Sgi-Sg_t0_ij)+0.5*((1.-tdew_idx)*(Sgi-Sg_n0_ij))
            ic_So=(0.-So_t0_ij)                             # Oil Saturation is Zero at initial conditions 
            ic=ic_p+ic_Sg#+ic_So
            #==========================================Rate Optimization Solution)================================== 
            # This is an optimization for the variable rate-pressure 
            # ======================================================================================================
            s1_n1_ij=out_n1[-1][...,1:-1,1:-1,:]

            # Initial Timestep, t1
            xn_t1=list(xi)
            xn_t1[3]=tf.ones_like(xn_t1[0])*t1_norm
            out_t1=model(xn_t1)
            
            #=======================================================================================================
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom)#q_well_idx*(q_n1_ij-out_t1[-2])          # Rate output index: -2
            qrc_3=tf.zeros_like(dom)#q_well_idx*(1e-8*-out_t0[-2])
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            #=============================================================================
            return dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]

        # Stack the physics-based loss (if any)
        def stack_physics_error():
            x_i,tshift_fac_i,tsf_0_norm_i=time_shifting(x,shift_frac_mean=0.5,pred_cycle_mean=1.,random=False)
            tstep_wt=tf.cast(x_i[3]<=tsf_0_norm_i,model.dtype)+tf.cast(x_i[3]>tsf_0_norm_i,model.dtype)*tshift_fac_i
            #dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=tf.where(tf.math.equal(model.cfd_type['Fluid_Type'],'GC'),physics_error_gas_oil(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.}),physics_error(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.}))
            
            dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=physics_error_gas_oil(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})
            #dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=physics_error(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  #update later
            return [dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i],[out_n0_i,out_n1_i],no_grid_blocks
        
        def physics_error_zeros_like():
            dom=tf.zeros_like(y[0],dtype=dt_type)
            dbc=dom
            nbc=dom
            ibc_n=dom
            ic=dom
            mbc=dom
            cmbc=dom
            qrc_1=dom;qrc_2=dom;qrc_3=dom,qrc_4=dom
            out_n0=tf.stack([dom]*4) 
            out_n1=tf.stack([dom]*4) 
            no_grid_blocks=[0.,0.,0.,0.,0.]
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            return [dom,dbc,nbc,ibc_n,ic,*qrc,mbc,cmbc],[out_n0,out_n1],no_grid_blocks
        
        
        phy_error,out_n,no_blks=stack_physics_error()
        stacked_pinn_errors=phy_error[0:-2]# tf.stack(phy_error[0:-2],axis=0)    #+ic_error()
        stacked_outs=out_n#tf.stack(out_n,axis=0)  
        checks=[phy_error[-2],phy_error[-1]]

        #breakpoint()
        return stacked_pinn_errors,stacked_outs,checks,no_blks

@tf.function
def zeros_like_pinn_error(model,x,y):
    # Create a zero-based tensor corresponding to batch shape
    dom=tf.zeros_like(y[0],dtype=dt_type)             # First label column is used
    dbc=tf.zeros_like(y[0],dtype=dt_type) 
    nbc=tf.zeros_like(y[0],dtype=dt_type) 
    ibc=tf.zeros_like(y[0],dtype=dt_type) 
    ic=tf.zeros_like(y[0],dtype=dt_type) 
    stacked_zeros_like_pinn_errors=tf.stack([dom,dbc,nbc,ibc,ic],axis=1)
    return stacked_zeros_like_pinn_errors
#==================================================================================================================================================
@tf.function
def boolean_mask_cond(x=None,y=None,data=[],bool_mask=[],solu_count=None):
    output=tf.cond(tf.math.equal(x,y),lambda: [tf.boolean_mask(data,bool_mask,axis=0),tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0))],\
                           lambda: [tf.multiply(tf.boolean_mask(data,bool_mask,axis=0),0.),tf.multiply(tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0)),0.)])
    return output

#==================================================================================================================================================
#@tf.function(jit_compile=True)
def pinn_batch_sse_grad(model,x,y):
    # Physics gradient for Arrangement Type 1: 
    # DATA ARRANGEMENT FOR TYPE 1
    # Training Features: INPUTS: 
    # x is a list of inputs (numpy arrays or tensors) for the given batch_size
    # x[0] is the grid block x_coord in ft
    # x[1] is the grid block y_coord in ft
    # x[2] is the grid block z_coord in ft
    # x[3] is the time in day
    # x[4] is the grid block porosity 
    # x[5] is the grid block x-permeability in mD
    # x[6] is the grid block z-permeability in mD
    
    # x[7] is the segment vector in the x-direction in ft (used for approximating the outer boundary)
    # x[8] is the segment vector in the y-direction in ft (used for approximating the outer boundary)
    # x[9] is the segment vector in the z-direction in ft (used for approximating the outer boundary)        
    # x[10] is the grid block x-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[11] is the grid block y-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[12] is the grid block z-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[13] is the harmonic average capacity ratio (i.e., kxdz(i+1)/kxdz(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
    # x[14] is the harmonic average capacity ratio (i.e., kzdx(i+1)/kzdx(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
    # x[15] is the input label as float indicating whether DOM(0), DBC(1), NBC(2), IBC(3), IC(4) or Train Data(5)   Label x[0-DOM|1-DBC|2-NBC|3-IBC|4-IC|5-TD-Full|6-TD-Sample]

    # Training Label: OUTPUTS:
    # y[0] is the training label grid block pressure (psia)
    # y[1] is the training label block saturation--gas
    # y[2] is the training label block saturation--oil
    # y[3] is the training label gas Formation Volume Factor in bbl/MScf
    # y[4] is the training label oil Formation Volume Factor in bbl/STB
    # y[5] is the training label gas viscosity in cp
    # y[6] is the training label oil viscosity in cp
    # y[7] is the training label gas rate in Mscf/D
    # y[8] is the training label oil rate in STB/D
    
    # Model OUTPUTS:
    # out[0] is the predicted grid block pressure (psia)
    # out[1] is the predicted grid block gas saturation
    # out[2] is the predicted grid block oil saturation
    # out[3] is the predicted grid block gas Formation Volume Factor inverse (1/Bg)
    # out[4] is the predicted grid block oil Formation Volume Factor inverse (1/Bo)
    # out[5] is the predicted grid block gas viscosity inverse (1/ug)
    # out[6] is the predicted grid block oil viscosity inverse (1/uo)
    
    with tf.GradientTape(persistent=True) as tape3:
        #out = model(x, training=True) # Forward pass
         
        # Check the unweighted root MSE for threshold before computing the PINN losses

        #rmse=tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y[0]-tf.squeeze(ps[0]))))
                                                        
        #pinn_errors=tf.where(tf.math.less_equal(rmse,model.cfd_type['PINN_Activation_RMSE']), pinn_error(model,x,y),\
                           #tf.tile(tf.zeros_like(ps[0]),[1,4]))   
        
        # Activation function and re-normalization of the weights
        #nwt=tf.linalg.normalize(tf.nn.relu(model.nwt),ord=1)[0]
        '''x1=list(x)
        x2=list(x)'''
        pinn_errors,outs,checks,no_blks=model.loss_func['Physics_Error'](model,x,y) 
        dom_pinn=pinn_errors[0]
        dbc_pinn=pinn_errors[1]
        nbc_pinn=pinn_errors[2]
        ibc_pinn=pinn_errors[3]
        ic_pinn=pinn_errors[4]
        qrc_pinn=tf.stack(pinn_errors[5:])       # First Condition Rate Check -- MaxQ at MinBHP
                                                 # Rate Check (t=timestep) -- Init TimeStep
                                                 # Initial Rate check (t=0)
                                                 # First Condition Slack Variable Check (t=0) -- MaxQ at MinBHP,t=0       
        mbc_pinn=checks[0]                     # MBC: Tank Material Balance Check
        cmbc_pinn=checks[1]                      # CMBC: Cumulative Tank Material Balance Check
        #====================================Train Data (if any labelled)=======================================
        # Training data includes the pressure and gas saturation
        nT=model.nT#tf.shape(outs[0])[0]-4   
        
        y_label=tf.stack([model.loss_func['Reshape'](y[i]) for i in model.nT_list],axis=0)
        y_model=outs[0][0:nT]
        td=y_label-y_model     
        
        # Get the floor value (nearest integer lower than the index value). non integers are used the index values
        # Compute the PINN error by multiplying with the corresponding label index column vector
        
        # dom_lbl_idx=tf.reshape(x[model.lbl_idx][:,model.solu_idx['DOM']],model.cfd_type['Dimension']['Reshape'])
        # dbc_lbl_idx=tf.reshape(x[model.lbl_idx][:,model.solu_idx['DBC']],model.cfd_type['Dimension']['Reshape'])
        # nbc_lbl_idx=tf.reshape(x[model.lbl_idx][:,model.solu_idx['NBC']],model.cfd_type['Dimension']['Reshape'])
        # ibc_lbl_idx=tf.reshape(x[model.lbl_idx][:,model.solu_idx['IBC']],model.cfd_type['Dimension']['Reshape'])
        # ic_lbl_idx=tf.reshape(x[model.lbl_idx][:,model.solu_idx['IC']],model.cfd_type['Dimension']['Reshape'])  # Derived from DOM+DBC+NBC+IBC 
       
        # Calculate the (Euclidean norm)**2 of each solution term--i.e., the Error term
        '''lower_limit=-2**(model.cfd_type['Float_Exp']-1)
        upper_limit=2**(model.cfd_type['Float_Exp']-1)
        dom_pinn_se=tf.clip_by_value(tf.math.square(dom_pinn),lower_limit,upper_limit) '''

        dom_pinn_se=tf.math.square(dom_pinn)                 
        dbc_pinn_se=tf.math.square(dbc_pinn)
        nbc_pinn_se=tf.math.square(nbc_pinn) 
        ibc_pinn_se=tf.math.square(ibc_pinn)
        ic_pinn_se=tf.math.square(ic_pinn) 
        qrc_pinn_se=tf.math.square(qrc_pinn)
        mbc_pinn_se=tf.math.square(mbc_pinn)
        cmbc_pinn_se=tf.math.square(cmbc_pinn)
        
        # Compute the Sum of Squared Errors (SSE) for the PINN term
        dom_pinn_sse=tf.math.reduce_sum(dom_pinn_se)
        dbc_pinn_sse=tf.math.reduce_sum(dbc_pinn_se)
        nbc_pinn_sse=tf.math.reduce_sum(nbc_pinn_se)
        ibc_pinn_sse=tf.math.reduce_sum(ibc_pinn_se)
        ic_pinn_sse=tf.math.reduce_sum(ic_pinn_se)
        qrc_pinn_sse=tf.math.reduce_sum(qrc_pinn_se)#,axis=[1,2,3,4])
        mbc_pinn_sse=tf.math.reduce_sum(mbc_pinn_se)
        cmbc_pinn_sse=tf.math.reduce_sum(cmbc_pinn_se)
        # Calculate the (Euclidean norm)**2 of the training term
        td_se=tf.math.square(td)

        # Compute the Sum of Squared Errors (SSE) of the Training term
        td_sse=tf.math.reduce_sum(td_se,axis=[1,2,3,4])

        # Average the rate check losses with the nbc loss
        nbc_sum_pinn_sse=nbc_pinn_sse+tf.reduce_sum(qrc_pinn_sse) #Rate check is averaged with the NBC Loss
        nbc_count_nonzero=tf.math.count_nonzero(nbc_pinn_sse,dtype=nbc_sum_pinn_sse.dtype)+tf.math.count_nonzero(qrc_pinn_sse,dtype=nbc_sum_pinn_sse.dtype)
        nbc_avg_pinn_sse=tf.math.divide(nbc_sum_pinn_sse,nbc_count_nonzero)

        # Weight the regularization term 
        dom_wsse=model.nwt[0]*dom_pinn_sse
        dbc_wsse=model.nwt[1]*dbc_pinn_sse
        nbc_wsse=model.nwt[2]*nbc_sum_pinn_sse                # nbc_avg_pinn_sse
        ibc_wsse=model.nwt[3]*ibc_pinn_sse
        ic_wsse=model.nwt[4]*ic_pinn_sse                      # Exclusive to the PINN solution
        mbc_wsse=model.nwt[5]*mbc_pinn_sse
        cmbc_wsse=model.nwt[6]*cmbc_pinn_sse
        
        # Compute the weighted training loss for the batch
        td_wsse=model.nwt[7:(7+nT)]*td_sse
      
        batch_wsse = dom_wsse+dbc_wsse+nbc_wsse+ibc_wsse+ic_wsse+mbc_wsse+cmbc_wsse+tf.reduce_sum(td_wsse)
        
        # Count the unique appearance of each loss term that does not have a zero identifier
        
        '''dom_error_count=tf.cast(tf.math.reduce_prod(dom_pinn_se.shape),dtype=model.dtype)
        dbc_error_count=tf.cast(tf.math.reduce_prod(dbc_pinn_se.shape),dtype=model.dtype)
        nbc_error_count=tf.cast(tf.math.reduce_prod(nbc_pinn_se.shape),dtype=model.dtype)
        ibc_error_count=tf.cast(tf.math.reduce_prod(ibc_pinn_se.shape),dtype=model.dtype)
        ic_error_count=tf.cast(tf.math.reduce_prod(ic_pinn_se.shape),dtype=model.dtype)
        mbc_error_count=tf.cast(tf.math.reduce_prod(ic_pinn_se.shape),dtype=model.dtype)           # Tank Model 
        cmbc_error_count=tf.cast(tf.math.reduce_prod(ic_pinn_se.shape),dtype=model.dtype)  
        td_error_count=tf.cast(tf.math.reduce_prod(td_se[0].shape),dtype=model.dtype)'''

        dom_error_count=tf.math.reduce_sum(tf.ones_like(dom_pinn_se))
        dbc_error_count=tf.math.reduce_sum(tf.ones_like(dbc_pinn_se))
        nbc_error_count=tf.math.reduce_sum(tf.ones_like(nbc_pinn_se))
        ibc_error_count=tf.math.reduce_sum(tf.ones_like(ibc_pinn_se))
        ic_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))
        mbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))#*tf.cast(tf.math.reduce_prod(tf.shape(ic_pinn_se)[1:]),model.dtype)    #tf.math.reduce_sum(tf.reduce_mean(tf.ones_like(ic_pinn_se),axis=[1,2,3])) #tf.math.reduce_sum(tf.ones_like(ic_pinn_se))          # Tank Model 
        cmbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))  
        td_error_count=tf.math.reduce_sum(tf.ones_like(td_se[0]))
        #+tf.math.reduce_sum(x[model.lbl_idx][...,model.solu_idx['DBC']])+tf.math.reduce_sum(x[model.lbl_idx][...,model.solu_idx['IBC']])
                
        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        dom_wmse=dom_wsse/zeros_to_ones(dom_error_count)
        dbc_wmse=dbc_wsse/zeros_to_ones(dbc_error_count)
        nbc_wmse=nbc_wsse/zeros_to_ones(nbc_error_count)
        ibc_wmse=ibc_wsse/zeros_to_ones(ibc_error_count)
        ic_wmse=ic_wsse/zeros_to_ones(ic_error_count)
        mbc_wmse=mbc_wsse/zeros_to_ones(mbc_error_count)
        cmbc_wmse=cmbc_wsse/zeros_to_ones(cmbc_error_count)
        td_wmse=td_wsse/zeros_to_ones(td_error_count)
        
        #tf.print('DOM_WMSE\n',dom_wsse,'\nDBC_WMSE\n',dbc_wsse,'\nNBC_WMSE\n',nbc_wsse,'\nIBC_WMSE\n',ibc_wsse,'\nIC_WMSE\n',ic_wsse,'\nMBC_WMSE\n',mbc_wsse,'\nCMBC_WMSE\n',cmbc_wsse,'\nTD_WMSE\n',td_wmse,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
        batch_wmse = dom_wmse+dbc_wmse+nbc_wmse+ibc_wmse+ic_wmse+mbc_wmse+cmbc_wmse+tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix
             
    # Compute the gradients of each loss term
    dom_wsse_grad=tape3.gradient(dom_wsse, model.trainable_variables,unconnected_gradients='zero')
    dbc_wsse_grad=tape3.gradient(dbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    nbc_wsse_grad=tape3.gradient(nbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ibc_wsse_grad=tape3.gradient(ibc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ic_wsse_grad=tape3.gradient(ic_wsse, model.trainable_variables,unconnected_gradients='zero')
    mbc_wsse_grad=tape3.gradient(mbc_wsse, model.trainable_variables,unconnected_gradients='zero')   
    cmbc_wsse_grad=tape3.gradient(cmbc_wsse, model.trainable_variables,unconnected_gradients='zero') 
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.
    
    #Compute the gradient of the batch
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    
    del tape3
    
    _wsse=[batch_wsse,dom_wsse,dbc_wsse,nbc_wsse,ibc_wsse,ic_wsse,mbc_wsse,cmbc_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,dom_wsse_grad,dbc_wsse_grad,nbc_wsse_grad,ibc_wsse_grad,ic_wsse_grad,mbc_wsse_grad,cmbc_wsse_grad,td_wsse_grad]
    error_count=[1,dom_error_count,dbc_error_count,nbc_error_count,ibc_error_count,ic_error_count,mbc_error_count,cmbc_error_count,tf.reduce_sum(td_error_count)]

    _wmse=[batch_wmse,dom_wmse,dbc_wmse,nbc_wmse,ibc_wmse,ic_wmse,mbc_wmse,cmbc_wmse,td_wmse]  

    # Reshape the output for a flat data: out=tf.reshape(out[0][0:nT,...],(2,-1))    
    #breakpoint()
    return [_wsse,_wsse_grad,error_count,_wmse,model.loss_func['Squeeze_Out'](tf.reshape(outs[0][0:nT,...],(nT,-1,*model.cfd_type['Dimension']['Dim'])))]



@tf.function
def nopinn_batch_sse_grad_1(model,x,y):
    # Physics gradient for Arrangement Type 1: 
    # DATA ARRANGEMENT FOR TYPE 1
    # Training Features: INPUTS: 
    # x is a list of inputs (numpy arrays or tensors) for the given batch_size
    # x[0] is the grid block x_coord in ft
    # x[1] is the grid block y_coord in ft
    # x[2] is the grid block z_coord in ft
    # x[3] is the time in day
    # x[4] is the grid block porosity 
    # x[5] is the grid block x-permeability in mD
    # x[6] is the grid block z-permeability in mD
    
    # x[7] is the segment vector in the x-direction in ft (used for approximating the outer boundary)
    # x[8] is the segment vector in the y-direction in ft (used for approximating the outer boundary)
    # x[9] is the segment vector in the z-direction in ft (used for approximating the outer boundary)        
    # x[10] is the grid block x-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[11] is the grid block y-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[12] is the grid block z-dimension in ft (used for Inner Boundary Condition)--Average values can be used
    # x[13] is the harmonic average capacity ratio (i.e., kxdz(i+1)/kxdz(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
    # x[14] is the harmonic average capacity ratio (i.e., kzdx(i+1)/kzdx(i))  of two corresponding grid blocks (mainly for Outer Boundary Condition)
    # x[15] is the input label as float indicating whether DOM(0), DBC(1), NBC(2), IBC(3), IC(4) or Train Data(5)   Label x[0-DOM|1-DBC|2-NBC|3-IBC|4-IC|5-TD-Full|6-TD-Sample]

    # Training Label: OUTPUTS:
    # y[0] is the training label grid block pressure (psia)
    # y[1] is the training label block saturation--gas
    # y[2] is the training label block saturation--oil
    # y[3] is the training label gas Formation Volume Factor in bbl/MScf
    # y[4] is the training label oil Formation Volume Factor in bbl/STB
    # y[5] is the training label gas viscosity in cp
    # y[6] is the training label oil viscosity in cp
    # y[7] is the training label gas rate in Mscf/D
    # y[8] is the training label oil rate in STB/D
    
    # Model OUTPUTS:
    # out[0] is the predicted grid block pressure (psia)
    # out[1] is the predicted grid block gas saturation
    # out[2] is the predicted grid block oil saturation
    # out[3] is the predicted grid block gas Formation Volume Factor inverse (1/Bg)
    # out[4] is the predicted grid block oil Formation Volume Factor inverse (1/Bo)
    # out[5] is the predicted grid block gas viscosity inverse (1/ug)
    # out[6] is the predicted grid block oil viscosity inverse (1/uo)
    
    with tf.GradientTape(persistent=True) as tape3:
        out = model(x, training=True) # Forward pass
                         
        pinn_errors=zeros_like_pinn_error(model,x,y)
        dom=pinn_errors[:,0]
        dbc=pinn_errors[:,1]
        nbc=pinn_errors[:,2] 
        ibc=pinn_errors[:,3]
        ic=pinn_errors[:,4]
        #====================================Train Data (if any labelled)=======================================
        # Training data includes the porosity, constitutive relationship, k and Quantities of Interest (QoI) like pressure and saturation
        nT=tf.shape(out)[0]   
        y_label=tf.stack(y,axis=1)[:,0:nT]
        y_model=tf.squeeze(tf.stack(out,axis=1),[-1])
        td=y_label-y_model

        #td=tf.stack([y[0]-tf.squeeze(out[0]),y[1]-tf.squeeze(out[1]),y[2]-tf.squeeze(out[2]),y[3]-tf.squeeze(out[3]),\
        #             y[4]-tf.squeeze(out[4]),y[5]-tf.squeeze(out[5]),y[6]-tf.squeeze(out[6])],axis=1)
                

        # Get the floor value (nearest integer lower than the index value). non integers are used the index values
        # Compute the PINN error by multiplying with the corresponding label index column vector
        dom_pinn=tf.multiply(dom,tf.math.floor(x[model.lbl_idx][:,model.solu_idx['DOM']]))
        dbc_pinn=tf.multiply(dbc,tf.math.floor(x[model.lbl_idx][:,model.solu_idx['DBC']]))
        nbc_pinn=tf.multiply(nbc,tf.math.floor(x[model.lbl_idx][:,model.solu_idx['NBC']]))
        ibc_pinn=tf.multiply(ibc,tf.math.floor(x[model.lbl_idx][:,model.solu_idx['IBC']]))  
        ic_pinn=tf.multiply(ic,tf.math.floor(x[model.lbl_idx][:,model.solu_idx['IC']]))               # Derived from DOM+DBC+NBC+IBC 
       
        # Repeat label index colums based on the number of training variables. E.g., pressure, phi and k
        dom_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][:,model.solu_idx['DOM']],-1),nT,axis=1)
        #dbc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][:,model.solu_idx['DBC']],-1),nT,axis=1)
        #nbc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][:,model.solu_idx['NBC']],-1),nT,axis=1)
        #ibc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][:,model.solu_idx['IBC']],-1),nT,axis=1)
        
        # Compute the training data errors. Errors are computed according to the number of training variables
        dom_td=tf.multiply(td,dom_td_label_index) 
        #dbc_td=tf.multiply(td,dbc_td_label_index) 
        #nbc_td=tf.multiply(td,nbc_td_label_index) 
        #ibc_td=tf.multiply(td,ibc_td_label_index) 

        # Calculate the (Euclidean norm)**2 of each solution term--i.e., the Error term
        dom_pinn_se=tf.math.square(dom_pinn)                        
        dbc_pinn_se=tf.math.square(dbc_pinn)
        nbc_pinn_se=tf.math.square(nbc_pinn)
        ibc_pinn_se=tf.math.square(ibc_pinn)
        ic_pinn_se=tf.math.square(ic_pinn)
                
        # Compute the Sum of Squared Errors (SSE) for the PINN term
        dom_pinn_sse=tf.math.reduce_sum(dom_pinn_se)
        dbc_pinn_sse=tf.math.reduce_sum(dbc_pinn_se)
        nbc_pinn_sse=tf.math.reduce_sum(nbc_pinn_se)
        ibc_pinn_sse=tf.math.reduce_sum(ibc_pinn_se)
        ic_pinn_sse=tf.math.reduce_sum(ic_pinn_se)
        
        # Calculate the (Euclidean norm)**2 of the training term
        dom_td_se=tf.math.square(dom_td)
        #dbc_td_se=tf.math.square(dbc_td)
        #nbc_td_se=tf.math.square(nbc_td)
        #ibc_td_se=tf.math.square(ibc_td)
        
        # Compute the Sum of Squared Errors (SSE) of the Training term
        dom_td_sse=tf.math.reduce_sum(dom_td_se,axis=0)
        #dbc_td_sse=tf.math.reduce_sum(dbc_td_se,axis=0)
        #nbc_td_sse=tf.math.reduce_sum(nbc_td_se,axis=0)
        #ibc_td_sse=tf.math.reduce_sum(ibc_td_se,axis=0)
        
        # Weight the regularization term 
        dom_wsse=model.nwt[0]*dom_pinn_sse
        dbc_wsse=model.nwt[1]*dbc_pinn_sse
        nbc_wsse=model.nwt[2]*nbc_pinn_sse
        ibc_wsse=model.nwt[3]*ibc_pinn_sse
        ic_wsse=model.nwt[4]*ic_pinn_sse                      # Exclusive to the PINN solution

        # Compute the training loss for the batch
        td_sse=dom_td_sse  #dbc_td_sse+ibc_td_sse        
        td_wsse=model.nwt[5:(5+nT)]*td_sse
        #td_wsse_unstack=tf.unstack()        
        
        batch_wsse = dom_wsse+dbc_wsse+nbc_wsse+ibc_wsse+ic_wsse+tf.reduce_sum(td_wsse)
        
        # Count the unique appearance of each loss term that does not have a zero identifier
        dom_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][:,model.solu_idx['DOM']]))
        dbc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][:,model.solu_idx['DBC']]))
        nbc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][:,model.solu_idx['NBC']]))
        ibc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][:,model.solu_idx['IBC']]))
        ic_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][:,model.solu_idx['IC']]))  
    
        # Compute the training error count
        td_error_count=tf.math.reduce_sum(x[model.lbl_idx][:,model.solu_idx['DOM']],axis=0)
        #+tf.math.reduce_sum(x[model.lbl_idx][:,model.solu_idx['DBC']],axis=0)+tf.math.reduce_sum(x[model.lbl_idx][:,model.solu_idx['IBC']],axis=0)
                
        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        dom_wmse=dom_wsse/zeros_to_ones(dom_error_count)
        dbc_wmse=dbc_wsse/zeros_to_ones(dbc_error_count)
        nbc_wmse=nbc_wsse/zeros_to_ones(nbc_error_count)
        ibc_wmse=ibc_wsse/zeros_to_ones(ibc_error_count)
        ic_wmse=ic_wsse/zeros_to_ones(ic_error_count)
        td_wmse=td_wsse/zeros_to_ones(td_error_count)
        
        batch_wmse = dom_wmse+dbc_wmse+nbc_wmse+ibc_wmse+ic_wmse+tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix
            
    # Compute the gradients of each loss term
    dom_wsse_grad=tape3.gradient(dom_wsse, model.trainable_variables,unconnected_gradients='zero')
    dbc_wsse_grad=tape3.gradient(dbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    nbc_wsse_grad=tape3.gradient(nbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ibc_wsse_grad=tape3.gradient(ibc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ic_wsse_grad=tape3.gradient(ic_wsse, model.trainable_variables,unconnected_gradients='zero')
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.
    
    #Compute the gradient of the batch
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    del tape3
    
    _wsse=[batch_wsse,dom_wsse,dbc_wsse,nbc_wsse,ibc_wsse,ic_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,dom_wsse_grad,dbc_wsse_grad,nbc_wsse_grad,ibc_wsse_grad,ic_wsse_grad,td_wsse_grad]
    error_count=[1,dom_error_count,dbc_error_count,nbc_error_count,ibc_error_count,ic_error_count,tf.reduce_sum(td_error_count)]
    
    _wmse=[batch_wmse,dom_wmse,dbc_wmse,nbc_wmse,ibc_wmse,ic_wmse,td_wmse]   
    
    return [_wsse,_wsse_grad,error_count,_wmse,out]

@tf.function
def nopinn_batch_sse_grad_pvt(model,x,y):
    # Physics gradient for Arrangement Type 1: 
    # DATA ARRANGEMENT FOR TYPE 1
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
        #====================================Train Data (if any labelled)=======================================
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
        
        batch_wmse = tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix
            
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

@tf.function
def nopinn_batch_sse_grad_conv2d(model,x,y):
    # Non Physics gradient for 2D convolutional neural network: 
    # Training Model Features: INPUT CHANNELS: 
    # x[3] is the time channel in day
    # x[4] is the grid block porosity channel 
    # x[5] is the grid block x-permeability channel in mD

    # Training Label: OUTPUTS:
    # y[0] is the training label grid block pressure (psia) 
    # y[1] is the training label block saturation--gas
    # y[2] is the training label block saturation--oil
    
    # Model OUTPUT CHANNELS:
    # out[0] is the predicted grid block pressure (psia)
    # out[1] is the predicted grid block gas saturation
    # out[2] is the predicted grid block oil saturation
    
    with tf.GradientTape(persistent=True) as tape3:
        out = model(x, training=True) # Forward pass
        pinn_errors=zeros_like_pinn_error(model,x,y)
        dom=pinn_errors[:,0]
        dbc=pinn_errors[:,1]
        nbc=pinn_errors[:,2] 
        ibc=pinn_errors[:,3]
        ic=pinn_errors[:,4]
        #====================================Train Data (if any labelled)=======================================
        # Training data includes the porosity, constitutive relationship, k and Quantities of Interest (QoI) like pressure and saturation
        nT=tf.shape(out)[0]   
        y_label=tf.stack(y,axis=0)[0:nT]
        y_model=tf.stack(out,axis=0)
        td=y_label-y_model

        #td=tf.stack([y[0]-tf.squeeze(out[0]),y[1]-tf.squeeze(out[1]),y[2]-tf.squeeze(out[2]),y[3]-tf.squeeze(out[3]),\
        #             y[4]-tf.squeeze(out[4]),y[5]-tf.squeeze(out[5]),y[6]-tf.squeeze(out[6])],axis=1)
                
        # Get the floor value (nearest integer lower than the index value). non integers are used the index values
        # Compute the PINN error by multiplying with the corresponding label index column vector
        dom_pinn=tf.multiply(dom,tf.math.floor(x[model.lbl_idx][...,model.solu_idx['DOM']]))
        dbc_pinn=tf.multiply(dbc,tf.math.floor(x[model.lbl_idx][...,model.solu_idx['DBC']]))
        nbc_pinn=tf.multiply(nbc,tf.math.floor(x[model.lbl_idx][...,model.solu_idx['NBC']]))
        ibc_pinn=tf.multiply(ibc,tf.math.floor(x[model.lbl_idx][...,model.solu_idx['IBC']]))  
        ic_pinn=tf.multiply(ic,tf.math.floor(x[model.lbl_idx][...,model.solu_idx['IC']]))               # Derived from DOM+DBC+NBC+IBC 
       
        # Repeat label index colums based on the number of training variables. E.g., pressure, phi and k
        dom_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][...,model.solu_idx['DOM']],0),nT,axis=0)
        dbc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][...,model.solu_idx['DBC']],0),nT,axis=0)
        #nbc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][...,model.solu_idx['NBC'],0),nT,axis=0)
        ibc_td_label_index=tf.repeat(tf.expand_dims(x[model.lbl_idx][...,model.solu_idx['IBC']],0),nT,axis=0)

        # Compute the training data errors. Errors are computed according to the number of training variables
        dom_td=tf.multiply(td,dom_td_label_index) 
        dbc_td=tf.multiply(td,dbc_td_label_index) 
        #nbc_td=tf.multiply(td,nbc_td_label_index) 
        ibc_td=tf.multiply(td,ibc_td_label_index) 

        # Calculate the (Euclidean norm)**2 of each solution term--i.e., the Error term
        dom_pinn_se=tf.math.square(dom_pinn)                        
        dbc_pinn_se=tf.math.square(dbc_pinn)
        nbc_pinn_se=tf.math.square(nbc_pinn)
        ibc_pinn_se=tf.math.square(ibc_pinn)
        ic_pinn_se=tf.math.square(ic_pinn)
                
        # Compute the Sum of Squared Errors (SSE) for the PINN term
        dom_pinn_sse=tf.math.reduce_sum(dom_pinn_se)
        dbc_pinn_sse=tf.math.reduce_sum(dbc_pinn_se)
        nbc_pinn_sse=tf.math.reduce_sum(nbc_pinn_se)
        ibc_pinn_sse=tf.math.reduce_sum(ibc_pinn_se)
        ic_pinn_sse=tf.math.reduce_sum(ic_pinn_se)
        
        # Calculate the (Euclidean norm)**2 of the training term
        dom_td_se=tf.math.square(dom_td)
        dbc_td_se=tf.math.square(dbc_td)
        #nbc_td_se=tf.math.square(nbc_td)
        ibc_td_se=tf.math.square(ibc_td)

        # Compute the Sum of Squared Errors (SSE) of the Training term
        dom_td_sse=tf.math.reduce_sum(dom_td_se,axis=[1,2,3,4])
        dbc_td_sse=tf.math.reduce_sum(dbc_td_se,axis=[1,2,3,4])
        #nbc_td_sse=tf.math.reduce_sum(nbc_td_se,axis=[1,2,3,4])
        ibc_td_sse=tf.math.reduce_sum(ibc_td_se,axis=[1,2,3,4])
        
        # Weight the regularization term 
        dom_wsse=model.nwt[0]*dom_pinn_sse
        dbc_wsse=model.nwt[1]*dbc_pinn_sse
        nbc_wsse=model.nwt[2]*nbc_pinn_sse
        ibc_wsse=model.nwt[3]*ibc_pinn_sse
        ic_wsse=model.nwt[4]*ic_pinn_sse                      # Exclusive to the PINN solution

        # Compute the training loss for the batch
        td_sse=dom_td_sse+dbc_td_sse+ibc_td_sse        
        td_wsse=model.nwt[5:(5+nT)]*td_sse
      
        batch_wsse = dom_wsse+dbc_wsse+nbc_wsse+ibc_wsse+ic_wsse+tf.reduce_sum(td_wsse)
        
        # Count the unique appearance of each loss term that does not have a zero identifier
        dom_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][...,model.solu_idx['DOM']]))
        dbc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][...,model.solu_idx['DBC']]))
        nbc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][...,model.solu_idx['NBC']]))
        ibc_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][...,model.solu_idx['IBC']]))
        ic_error_count=tf.math.reduce_sum(tf.math.floor(x[model.lbl_idx][...,model.solu_idx['IC']]))  
    
        # Compute the training error count
        td_error_count=tf.math.reduce_sum(x[model.lbl_idx][...,model.solu_idx['DOM']])
        #+tf.math.reduce_sum(x[model.lbl_idx][...,model.solu_idx['DBC']])+tf.math.reduce_sum(x[model.lbl_idx][...,model.solu_idx['IBC']])
                
        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        dom_wmse=dom_wsse/zeros_to_ones(dom_error_count)
        dbc_wmse=dbc_wsse/zeros_to_ones(dbc_error_count)
        nbc_wmse=nbc_wsse/zeros_to_ones(nbc_error_count)
        ibc_wmse=ibc_wsse/zeros_to_ones(ibc_error_count)
        ic_wmse=ic_wsse/zeros_to_ones(ic_error_count)
        td_wmse=td_wsse/zeros_to_ones(td_error_count)
        
        batch_wmse = dom_wmse+dbc_wmse+nbc_wmse+ibc_wmse+ic_wmse+tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix
             
    # Compute the gradients of each loss term
    dom_wsse_grad=tape3.gradient(dom_wsse, model.trainable_variables,unconnected_gradients='zero')
    dbc_wsse_grad=tape3.gradient(dbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    nbc_wsse_grad=tape3.gradient(nbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ibc_wsse_grad=tape3.gradient(ibc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ic_wsse_grad=tape3.gradient(ic_wsse, model.trainable_variables,unconnected_gradients='zero')
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.
    
    #Compute the gradient of the batch
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    del tape3
    
    _wsse=[batch_wsse,dom_wsse,dbc_wsse,nbc_wsse,ibc_wsse,ic_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,dom_wsse_grad,dbc_wsse_grad,nbc_wsse_grad,ibc_wsse_grad,ic_wsse_grad,td_wsse_grad]
    error_count=[1,dom_error_count,dbc_error_count,nbc_error_count,ibc_error_count,ic_error_count,tf.reduce_sum(td_error_count)]

    _wmse=[batch_wmse,dom_wmse,dbc_wmse,nbc_wmse,ibc_wmse,ic_wmse,td_wmse]   
    return [_wsse,_wsse_grad,error_count,_wmse,out]
