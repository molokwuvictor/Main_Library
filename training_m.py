#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# © 2022 Victor Molokwu <vcm1@hw.ac.uk>
# Distributed under terms of the MIT license.
# A module of functions and classes for processing the data during batch training.

import os
import batch_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU: -1

import random
import tensorflow as tf
import numpy as np
import time
import tensorflow_addons as tfa
import math

# ================================================================================================================================
# A function that performs full dataset training training using a single or multi-optimizer.  
def train_ensembles(model=None,iteration={'Number':2, 'Seq_Ratio':0.7, 'Restarts':True},batch_xy=None,optimizer={'Name':['Adam'],'Multi':False},lrate={'Init_Rate':[0.0025,0.0003,0.0003],'Decay_Type':[None,None],'Decay_Weight':[0.25,0.25,0.25],\
                    'Decay_Ep':{'Start_Ep':[0.5,],'End_Ep':[0.8,],'Step_Fac':[20,],'Decay_Fac':[1,],'Stair_Case':[True,]},\
                    'Exp_Decay_Rate':{'Lr':[0.001,0.001,0.001],'Wt':[0.001,0.001,0.001]},\
                    'Exp_Decay_Rate_Base':{'Lr':[0.8,0.8,0.8],'Wt':[0.8,0.8,0.8]},'Cosine_Decay_Restarts_Par':{'Tmul':[1.5,1.5,1.5],'Mmul':[0.95,0.95,0.95],'Alpha':[0.01,0.01,0.01]},\
                    'Stair_Case':True,'Epoch_Fac':50,'Epochs':500,'Saved_Epochs':0.25,'Seq_Ratio':0.5},steps_per_epoch_=None, validation_data_=None, validation_steps_=None, verbose_=1, callbacks_=None):
    
    def scheduler(epoch, lr):
        if epoch < int(lrate['Decay_Start']*lrate['Epochs']):
            return lr
        else:
            return lr * tf.math.exp(-lrate['Exp_Decay_Rate'])
    
    # Piecewise Learning Rate Scheduler.
    def pwc_decay(boundaries = [194001,291000],values = [1e-0,0.7,0.3]):
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    
    # Exponential Learning Rate (Scheduler).
    def exp_decay(init_rate=None,decay_rate=0.8,epoch_fac=50,step_per_epoch=len(batch_xy),stair_case=True):
        return tf.keras.optimizers.schedules.ExponentialDecay(init_rate,decay_steps=int(tf.math.ceil(epoch_fac))*step_per_epoch,decay_rate=decay_rate,staircase=stair_case)
    
    # Constant-Exponential Learning Rate.
    def const_exp_decay(init_rate=None,decay_rate=0.1,decay_start=[0.5,],decay_end=[0.8,],epoch_fac=[20,],decay_fac=[1,],stair_case=[True,],steps_per_epoch=len(batch_xy),epochs=lrate['Epochs']):
        start_decay_step=[(i*steps_per_epoch*epochs) for i in decay_start]
        end_decay_step=[(i*steps_per_epoch*epochs) for i in decay_end]
        decay_steps=[int(i)*steps_per_epoch for i in epoch_fac]
        decay_rate=[(i*decay_rate) for i in decay_fac]
        
        return ConstantExponentialDecay(init_rate,decay_params={'Start':start_decay_step,'End':end_decay_step,'Steps':decay_steps,'Rate':decay_rate,'Stair_Case':stair_case})
    
    # Cosine Decay with Restarts (Scheduler)--Cosine Annealing.
    def cosine_decay_restarts(init_rate=None,epoch_fac=50,step_per_epoch=len(batch_xy),t_mul=2.0,m_mul=0.95,alpha=0.01):
        return CosineDecayRestarts(init_rate,first_decay_steps=int(tf.math.ceil(epoch_fac))*step_per_epoch,t_mul=t_mul,m_mul=m_mul,alpha=alpha)
        
    # Cyclical Learning Rate.
    def tricyc_decay(init_rate=None,max_rate=None,scale_fac=2.,epoch_fac=2,steps_per_epoch=len(batch_xy)):
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=init_rate,maximal_learning_rate=max_rate,\
            scale_fn=lambda x: 1/(scale_fac**(x-1)), step_size=int(tf.math.ceil(epoch_fac))*steps_per_epoch)
        return clr

    # Create a sequence list for the number of iterations.
    def create_seqlist(sum_seq=None,num_seq=None,seq_ratio=0.5):
        # Using Sum of terms of a GP.
        seq_list=[]; seq_start_idx=[]
        if seq_ratio<=1:
            # Find the first term.
            a=sum_seq*(1-seq_ratio)/(1-seq_ratio**(num_seq))
        else:
            a=sum_seq*(seq_ratio-1)/(seq_ratio**(num_seq)-1)
        for i in range(num_seq):
            seq_list.append(int(a*seq_ratio**(i)))
            if i==num_seq-1:        
                # Check and append remainder to last. 
                seq_list[-1]=seq_list[-1]+int(sum_seq-sum(seq_list))

            if i==0:
                seq_start_idx.append(int(0))
            else:
                seq_start_idx.append(sum(seq_list[:i]))
        return seq_list,seq_start_idx   

    # Create a sequence for the learning rates.
    lrate['Epochs'],epoch_start_idx=create_seqlist(sum_seq=lrate['Epochs'],num_seq=iteration['Number'],seq_ratio=iteration['Seq_Ratio'])

    # Update the learning and decay rates according to the number of iterations.
    lrate['Init_Rate']=[[lrate['Init_Rate'][i]]*iteration['Number'] for i in range(len(lrate['Init_Rate']))]
    lrate['Decay_Weight']=[[lrate['Init_Rate'][i][0]*lrate['Decay_Weight'][i]*0.001]*iteration['Number'] for i in range(len(lrate['Decay_Weight']))]

    # Create a list instance for the epoch factor if not list.
    if not isinstance(lrate['Epoch_Fac'], list):
        lrate['Epoch_Fac']=[lrate['Epoch_Fac']]*len(lrate['Init_Rate'])
    
    # Create a decay rate factor sequence for the weights -- used to multiply the decay rate for the weights after each iteration.
    drate_fac_wt=[1.*iteration['DRate']['Wt']**i for i in range(iteration['Number'])]

    drate_fac_ep=[1.*iteration['DRate']['Ep']**i for i in range(iteration['Number'])]
    
    # reinitialization method for the truncation error model (if any).
    def reinitialize_trn_err_model(model,no_realizations=100):
        if 'Truncation_Error_Model' in model.cfd_type:
            if model.cfd_type['Truncation_Error_Model']:
                reinitialize_model_weights(model.trn_model, no_realizations=no_realizations,global_seed=model.cfd_type['Seed'])
        return
    
    for i in range(0,iteration['Number']):
        if iteration['Restarts']['WBias']:
            # Re-Initialize the model's weights and biases as end of each training.
            reinitialize_model_weights(model, no_realizations=100,global_seed=model.cfd_type['Seed']) 
        else:
            if i==0:
                # Re-Initialize the model's weights and biases as end of each training.
                reinitialize_model_weights(model, no_realizations=100,global_seed=model.cfd_type['Seed']) 
            else:
                # Re-Initialize with model's best weights and biases (based on validation loss) of previous iteration.
                best_weight_bias_prev_iter=model.wblt_epoch_ens[i-1]['weight_bias']
        
        EP={'Frac':lrate['Saved_Epochs'],'Total':lrate['Epochs'][i]}                # Settings for the epochs fraction in which model's weights and biases are to be sampled
        wblt = callbacks_[0](epochs=EP)
        lr_decay=callbacks_[1](scheduler)
        tboard=callbacks_[2]                                                        # Tensorboard callback.
        if not optimizer['Multi']:
            if optimizer['Name'][0]=='Adam':
                opt_=tf.keras.optimizers.Adam(learning_rate=lrate['Init_Rate'][0],epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'])  # could as well used the Stochastic Gradient Descent (SGD|Adam)
            elif optimizer['Name'][0]=='Adamax':
                opt_=tf.keras.optimizers.Adamax(learning_rate=lrate['Init_Rate'][0],epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'])
            elif optimizer['Name'][0]=='SGD':
                opt_=tf.keras.optimizers.SGD(learning_rate=lrate['Init_Rate'][0],momentum=0.90,nesterov=True)
            elif optimizer['Name'][0]=='Nadam':
                opt_=tf.keras.optimizers.Nadam(learning_rate=lrate['Init_Rate'][0],epsilon=optimizer['Eps'])
            elif optimizer['Name'][0]=='LazyAdam':
                opt_=tfa.optimizers.LazyAdam(learning_rate=lrate['Init_Rate'][0],epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'])
            elif optimizer['Name'][0]=='RectAdam':
                opt_=tfa.optimizers.RectifiedAdam(learning_rate=lrate['Init_Rate'][0],epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'])
            callback_list=[wblt,lr_decay,tboard]
        else:
            step = tf.Variable(0, trainable=False)
            
            # Use a multioptimizer.
            optimizers_and_layers=[]
            layer_list=get_layer_list(model)
            
            # Sort the layer_list.
            var_list=['pre','psat','gsat','osat','inv','oth','rate','slack','GOR','OGR','trunc','time','bhp']
            dict_var_list={k:v for v,k in enumerate(var_list)}
            sorted_layer_list_keys=sorted(layer_list.keys(),key=dict_var_list.get)    # key is the index value for sorting .get values from a dictionary | .index for a list.
            layer_list = {key:layer_list[key] for key in sorted_layer_list_keys}

            print('Multi Optimizer Active...',' Variabes: ', layer_list.keys())
            if len(optimizer['Name'])==1:
                optimizer['Name']=len(lrate['Init_Rate'])*optimizer['Name']

            for key,val in layer_list.items():
                if len(val)!=0:
                    # lr_idx=list(layer_list.keys()).index(key)
                    if len(layer_list.keys())==1 or model.cfd_type['Data_Arr']==3:
                        lr_idx=j=0
                    else:
                        if key in ['pre','psat']:
                            lr_idx=j=0
                        elif key in ['gsat','osat']:
                            lr_idx=j=1
                        elif key in ['inv','oth','rate','GOR','OGR','bhp','trunc']:
                            lr_idx=j=-2
                        elif key in ['slack','time']:
                            lr_idx=j=-1

                    # Set the learning rates.
                    if lrate['Decay_Type'][j]['Type']=='piecewise':    # Options: 'piecewise' | 'exponential' | 'tricyclic' | 'expcyclic' | 'None'.
                        lr=lrate['Init_Rate'][lr_idx][0]*pwc_decay()(step)
                        if not iteration['Restarts']['LRate']:
                            lrate['Init_Rate'][lr_idx][i]=lr(epoch_start_idx[i]*len(batch_xy))
                            lr=lrate['Init_Rate'][lr_idx][i]*pwc_decay()(step)
                    elif lrate['Decay_Type'][j]['Type']=='exponential':
                        # Update the initial rate based on the end of iteration epoch number.
                        if not iteration['Restarts']['LRate']:
                            lr=exp_decay(init_rate=lrate['Init_Rate'][lr_idx][0],decay_rate=lrate['Exp_Decay_Rate_Base']['Lr'][j],epoch_fac=lrate['Epoch_Fac'][j]*drate_fac_ep[0],stair_case=lrate['Stair_Case'])
                            lrate['Init_Rate'][lr_idx][i]=lr(epoch_start_idx[i]*len(batch_xy))
                        lr=exp_decay(init_rate=lrate['Init_Rate'][lr_idx][i],decay_rate=lrate['Exp_Decay_Rate_Base']['Lr'][j],epoch_fac=lrate['Epoch_Fac'][j]*drate_fac_ep[i],stair_case=lrate['Stair_Case'])
                    elif lrate['Decay_Type'][j]['Type']=='constant-exponential':
                        if not iteration['Restarts']['LRate']:
                             lr=const_exp_decay(init_rate=lrate['Init_Rate'][lr_idx][0],decay_rate=lrate['Exp_Decay_Rate']['Lr'][j],decay_start=lrate['Decay_Ep']['Start_Ep'],decay_end=lrate['Decay_Ep']['End_Ep'],epoch_fac=lrate['Decay_Ep']['Step_Fac'],decay_fac=lrate['Decay_Ep']['Decay_Fac'],stair_case=lrate['Decay_Ep']['Stair_Case'],epochs=lrate['Epochs'][i])
                             lrate['Init_Rate'][lr_idx][i]=lr(epoch_start_idx[i]*len(batch_xy))
                        lr=const_exp_decay(init_rate=lrate['Init_Rate'][lr_idx][i],decay_rate=lrate['Exp_Decay_Rate']['Lr'][j],decay_start=lrate['Decay_Ep']['Start_Ep'],decay_end=lrate['Decay_Ep']['End_Ep'],epoch_fac=lrate['Decay_Ep']['Step_Fac'],decay_fac=lrate['Decay_Ep']['Decay_Fac'],stair_case=lrate['Decay_Ep']['Stair_Case'],epochs=lrate['Epochs'][i])
                    elif lrate['Decay_Type'][j]['Type']=='cosine-restarts':
                        if not iteration['Restarts']['LRate']:
                            lr=cosine_decay_restarts(init_rate=lrate['Init_Rate'][lr_idx][0],epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[0],t_mul=lrate['Cosine_Decay_Restarts_Par']['Tmul'][j],\
                                                 m_mul=lrate['Cosine_Decay_Restarts_Par']['Mmul'][j],alpha=lrate['Cosine_Decay_Restarts_Par']['Alpha'][j])
                            lrate['Init_Rate'][lr_idx][i]=lr(epoch_start_idx[i]*len(batch_xy))
                        lr=cosine_decay_restarts(init_rate=lrate['Init_Rate'][lr_idx][i],epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[i],t_mul=lrate['Cosine_Decay_Restarts_Par']['Tmul'][j],\
                                                 m_mul=lrate['Cosine_Decay_Restarts_Par']['Mmul'][j],alpha=lrate['Cosine_Decay_Restarts_Par']['Alpha'][j])
                    elif lrate['Decay_Type'][j]['Type']=='tricyclic':
                        if not iteration['Restarts']['LRate']:
                            lr=tricyc_decay(init_rate=(lrate['Init_Rate'][lr_idx][0]*1e-2),max_rate=lrate['Init_Rate'][lr_idx][0]*1e1,scale_fac=1.1,epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[0])
                            lrate['Init_Rate'][lr_idx][i]=lr(epoch_start_idx[i]*len(batch_xy))
                        lr=tricyc_decay(init_rate=(lrate['Init_Rate'][lr_idx][i]*1e-2),max_rate=lrate['Init_Rate'][lr_idx][i]*1e1,scale_fac=1.1,epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[i])
                    elif lrate['Decay_Type'][j]['Type']=='constant':
                        lr=lrate['Init_Rate'][lr_idx][0] 
                    else:
                        lr=lrate['Init_Rate'][lr_idx][0] 
                    
                    # Set the decay rates.
                    if lrate['Decay_Type'][j]['Type']=='piecewise':    # Options: 'piecewise' | 'exponential' | 'tricyclic' | 'expcyclic' | 'None'
                        wd=lrate['Decay_Weight'][lr_idx][0]*pwc_decay()(step)
                        if not iteration['Restarts']['LRate']:
                            lrate['Decay_Weight'][lr_idx][i]=wd(epoch_start_idx[i]*len(batch_xy))
                            wd=lrate['Decay_Weight'][lr_idx][i]*pwc_decay()(step)
                    elif lrate['Decay_Type'][j]['Type']=='exponential':
                        # Update the initial rate based on the end of iteration epoch number.
                        if not iteration['Restarts']['LRate']:
                            wd=exp_decay(init_rate=lrate['Decay_Weight'][lr_idx][0],decay_rate=lrate['Exp_Decay_Rate_Base']['Wt'][j],epoch_fac=lrate['Epoch_Fac'][j]*drate_fac_ep[0],stair_case=lrate['Stair_Case'])
                            lrate['Decay_Weight'][lr_idx][i]=wd(epoch_start_idx[i]*len(batch_xy))
                        wd=exp_decay(init_rate=lrate['Decay_Weight'][lr_idx][i],decay_rate=lrate['Exp_Decay_Rate_Base']['Wt'][j]*drate_fac_wt[i],epoch_fac=lrate['Epoch_Fac'][j]*drate_fac_ep[i],stair_case=lrate['Stair_Case'])
                    elif lrate['Decay_Type'][j]['Type']=='constant-exponential':
                        if not iteration['Restarts']['LRate']:
                             wd=const_exp_decay(init_rate=lrate['Decay_Weight'][lr_idx][0],decay_rate=lrate['Exp_Decay_Rate']['Wt'][j],decay_start=lrate['Decay_Ep']['Start_Ep'],decay_end=lrate['Decay_Ep']['End_Ep'],epoch_fac=lrate['Decay_Ep']['Step_Fac'],decay_fac=lrate['Decay_Ep']['Decay_Fac'],stair_case=lrate['Decay_Ep']['Stair_Case'],epochs=lrate['Epochs'][i])
                             lrate['Decay_Weight'][lr_idx][i]=wd(epoch_start_idx[i]*len(batch_xy))
                        wd=const_exp_decay(init_rate=lrate['Decay_Weight'][lr_idx][i],decay_rate=lrate['Exp_Decay_Rate']['Wt'][j]*drate_fac_wt[i],decay_start=lrate['Decay_Ep']['Start_Ep'],decay_end=lrate['Decay_Ep']['End_Ep'],epoch_fac=lrate['Decay_Ep']['Step_Fac'],decay_fac=lrate['Decay_Ep']['Decay_Fac'],stair_case=lrate['Decay_Ep']['Stair_Case'])
                    elif lrate['Decay_Type'][j]['Type']=='cosine-restarts':
                        if not iteration['Restarts']['LRate']:
                            wd=cosine_decay_restarts(init_rate=lrate['Decay_Weight'][lr_idx][0],epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[0],t_mul=lrate['Cosine_Decay_Restarts_Par']['Tmul'][j],\
                                                 m_mul=lrate['Cosine_Decay_Restarts_Par']['Mmul'][j],alpha=lrate['Cosine_Decay_Restarts_Par']['Alpha'][j])
                            lrate['Decay_Weight'][lr_idx][i]=wd(epoch_start_idx[i]*len(batch_xy))
                        wd=cosine_decay_restarts(init_rate=lrate['Decay_Weight'][lr_idx][i],epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[i],t_mul=lrate['Cosine_Decay_Restarts_Par']['Tmul'][j],\
                                                 m_mul=lrate['Cosine_Decay_Restarts_Par']['Mmul'][j],alpha=lrate['Cosine_Decay_Restarts_Par']['Alpha'][j])
                    elif lrate['Decay_Type'][j]['Type']=='tricyclic':
                        if not iteration['Restarts']['LRate']:
                            wd=tricyc_decay(init_rate=(lrate['Decay_Weight'][lr_idx][0]*1e-2),max_rate=lrate['Decay_Weight'][lr_idx][0]*1e1,scale_fac=1.1,epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[0])
                            lrate['Decay_Weight'][lr_idx][i]=wd(epoch_start_idx[i]*len(batch_xy))
                        wd=tricyc_decay(init_rate=(lrate['Decay_Weight'][lr_idx][i]),max_rate=lrate['Decay_Weight'][lr_idx][i]*1e1,scale_fac=1.1,epoch_fac=lrate['Epoch_Fac']*drate_fac_ep[i])
                    elif lrate['Decay_Type'][j]['Type']=='constant':
                        wd=lrate['Decay_Weight'][lr_idx][i]
                    else:
                        wd=lrate['Decay_Weight'][lr_idx][i]
                       
                    # Can set clip value for gradient of small-valued outupts.
                    if key not in ['gsat','osat']:
                        _clipvalue=None;_global_clipnorm=None
                    else:
                        _clipvalue=None;_global_clipnorm=None
                    
                    # Set the Optimizer.
                    if optimizer['Name'][lr_idx]=='AdamW':
                        optimizers_and_layers.append((tfa.optimizers.AdamW(learning_rate=lr,weight_decay=wd,epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad']),val))
                    elif optimizer['Name'][lr_idx]=='AdaBelief':
                        optimizers_and_layers.append((tfa.optimizers.AdaBelief(learning_rate=lr,weight_decay=wd,epsilon=optimizer['Eps']**2, amsgrad=optimizer['AmsGrad'],rectify=optimizer['Rectify']),val))
                    elif optimizer['Name'][lr_idx]=='SGDW':
                        optimizers_and_layers.append((tfa.optimizers.SGDW(learning_rate=lr,weight_decay=wd, momentum=0.),val))
                    elif optimizer['Name'][lr_idx]=='RectAdam':
                        optimizers_and_layers.append((tfa.optimizers.RectifiedAdam(learning_rate=lr,weight_decay=wd,epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad']),val))
                    elif optimizer['Name'][lr_idx]=='Adam':
                        optimizers_and_layers.append((tf.keras.optimizers.Adam(learning_rate=lr,epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'],clipvalue=_clipvalue,global_clipnorm=_global_clipnorm),val))
                    else:
                        optimizers_and_layers.append((tf.keras.optimizers.Adam(learning_rate=lr,epsilon=optimizer['Eps'], amsgrad=optimizer['AmsGrad'],clipvalue=_clipvalue,global_clipnorm=_global_clipnorm),val))

            # Implements the LookAhead optimizer.
            if optimizer['LookAhead']:
                optimizers_and_layers=[(tfa.optimizers.Lookahead(optimizers_and_layers[i][0],sync_period=optimizer['Sync_Period'],slow_step_size=optimizer['LookAhead_Step']),optimizers_and_layers[i][1]) for i in range(len(optimizers_and_layers))]

            opt_= tfa.optimizers.MultiOptimizer(optimizers_and_layers)
            callback_list=[wblt,tboard]

        # opt_adam = runai.ga.keras.optimizers.Adam(steps=len(batch_xy),learning_rate=learning_rate_)
        model.compile(optimizer=opt_, run_eagerly=False) 
        model.fit(batch_xy, epochs=lrate['Epochs'][i], steps_per_epoch=steps_per_epoch_,validation_data=validation_data_, validation_steps=validation_steps_,verbose=verbose_, callbacks=callback_list)    #validation_split can be used when the split fraction is known
         
    # Get the index of the minimum val loss from the ensembles.
    loss_list=[model.wblt_epoch_ens[i]['val_loss'] for i in range(len(model.wblt_epoch_ens))]
    if sum(loss_list)==0.:
        # Get the index of the minimum train loss from the ensembles.
        loss_list=[model.wblt_epoch_ens[i]['loss'] for i in range(len(model.wblt_epoch_ens))]
    
    min_loss_idx_ens=loss_list.index(min(loss_list))
    # Save the weights of the model with lowest end loss.
    
    # Get the weights and biases with lowest validation loss in the ensembles.
    best_weight_bias_ens=model.wblt_epoch_ens[min_loss_idx_ens]['weight_bias']
    
    # Update the model with the best weights and biases.
    model.set_weights(best_weight_bias_ens)
    model.best_ens=min_loss_idx_ens
    return 

# A function that reinitializes the weights and biases of an artificial neural network model.  
def reinitialize_model_weights(model, no_realizations=1,skip_layers=['inv',],global_seed=None):
    if hasattr(model,'cfd_type') and global_seed==None:
        global_seed=model.cfd_type['Seed']
        
    weights_norm_realz=[]
    
    # Generate the seed list for the initializer.
    def custom_layer_reinitialize(_layer):
        kernel_initializer = _layer._kernel_initializer
        spectral_norm_kernel_initializer = tf.initializers.TruncatedNormal(stddev=0.02)                 # Kernel initializer for any spectral normalization layer. 
        random_fourier_bias_initializer = tf.initializers.RandomUniform(minval=0.0, maxval=2 * np.pi)   # Bias initializer for any random Fourier features.  
        old_weights_biases = _layer.get_weights()                                                       # tf.variables returns the class values and attributes.

        new_weights_biases=[]        
        for iowb in range(len(old_weights_biases)):
            if tf.reduce_sum(old_weights_biases[iowb])==0 or tf.reduce_prod(old_weights_biases[iowb].shape)==1:
                new_weights_biases.append(old_weights_biases[iowb])
            else:
                if tf.reduce_sum(old_weights_biases[iowb])>1.0 and 'bias' in _layer.variables[iowb].name.lower():
                    # Bias initializer for any random Fourier features.
                    new_weights_biases.append(random_fourier_bias_initializer(shape=old_weights_biases[iowb].shape))
                else:
                    if 'spectral_norm' in _layer.variables[iowb].name.lower():
                        # Spectral norm is usually initialzed from a truncated normal distribution.
                        new_weights_biases.append(spectral_norm_kernel_initializer(shape=old_weights_biases[iowb].shape))
                    else:
                        new_weights_biases.append(kernel_initializer(shape=old_weights_biases[iowb].shape))
                
        _layer.set_weights(new_weights_biases)
        return     
    random.seed(global_seed)
    seed=[(random.randint(0,2**(8-1))) for i in range(no_realizations)]

    for k in range(no_realizations):
        tf.random.set_seed(seed[k])
        for ix, layer in enumerate(model.layers):
            if hasattr(model,'cfd_type'):
                if model.cfd_type['Type']=='PINN' and bool([skip_layer for skip_layer in skip_layers if skip_layer.lower() in layer.name]):
                    continue
            if (hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer')):
                # tf.random.set_seed(seed[k])
                # get the kernel initializer and update the seed, bias is set to zero.
                kernel_initializer = model.layers[ix].kernel_initializer
                bias_initializer = model.layers[ix].bias_initializer

                #bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-6)
                # Initializer for the recurrent_kernel weights matrix if any (LSTM)
                if hasattr(model.layers[ix],'recurrent_initializer'):
                    tf.keras.initializers.get(model.layers[ix].recurrent_initializer).seed=tf.keras.initializers.get(model.layers[ix].kernel_initializer).seed
                    recurrent_kernel_initializer = model.layers[ix].recurrent_initializer
                    old_weights, old_recurrent_weights,old_biases = model.layers[ix].get_weights()
                    model.layers[ix].set_weights([kernel_initializer(shape=old_weights.shape,dtype=tf.float64),recurrent_kernel_initializer(old_recurrent_weights.shape),bias_initializer(shape=len(old_biases))])
                elif hasattr(model.layers[ix],'query_kernel'):
                    # Query kernel, key kernel, value kernel, projection kernel, projection bias
                    new_weights=[kernel_initializer(shape=model.layers[ix].get_weights()[i].shape) for i in range(len(model.layers[ix].get_weights())-1)]
                    new_bias=[bias_initializer(shape=len(model.layers[ix].get_weights()[-1]))]
                    model.layers[ix].set_weights(new_weights+new_bias)
                else:
                    old_weights,old_biases = model.layers[ix].get_weights()
                    model.layers[ix].set_weights([kernel_initializer(shape=old_weights.shape),bias_initializer(shape=len(old_biases))])
            elif hasattr(model.layers[ix], '_kernel_initializer'):
                 # tf.random.set_seed(seed[k])
                 custom_layer_reinitialize(model.layers[ix])
            elif (hasattr(model.layers[ix],'layers')):
                # has sublayers
                for ix_i,layer_i in enumerate(model.layers[ix].layers):
                    if (hasattr(model.layers[ix].layers[ix_i], 'kernel_initializer') and \
                        hasattr(model.layers[ix].layers[ix_i], 'bias_initializer')):
                        #tf.random.set_seed(seed[k])
                        kernel_initializer = model.layers[ix].layers[ix_i].kernel_initializer
                        bias_initializer = model.layers[ix].layers[ix_i].bias_initializer
                        old_weights,old_biases = model.layers[ix].layers[ix_i].get_weights()
                        model.layers[ix].layers[ix_i].set_weights([kernel_initializer(shape=old_weights.shape),bias_initializer(shape=len(old_biases))])
                    elif (hasattr(model.layers[ix].layers[ix_i],'_kernel_initializer')):
                        # Custom layer.
                        # tf.random.set_seed(seed[k])
                        custom_layer_reinitialize(model.layers[ix].layers[ix_i])
                        
        norm=(tf.math.divide(tf.linalg.global_norm(model.trainable_variables),tf.math.sqrt(tf.cast(len(model.trainable_variables),dtype=model.dtype)))) 
        weights_norm_realz.append([model.get_weights(),norm])
        
    # Get the minimum norm or close to zero

    # min_val_norm=min([weights_norm_realz[k][1] for k in range(len(weights_norm_realz))])
    # min_idx_norm=[weights_norm_realz[k][1] for k in range(len(weights_norm_realz))].index(min_val_norm)
    
    init_arr=[(weights_norm_realz[k][1]) for k in range(len(weights_norm_realz))]
    median_val_norm=np.median(init_arr)
    print('median_initialization_norm:',median_val_norm)
    min_idx_norm=(np.abs(init_arr-median_val_norm)).argmin()
    
    # Set initial weights with median or minimum norm parameters.

    # avg_weights_realz=[tf.reduce_mean(i,axis=0) for i in zip(*[weights_norm_realz[k][0] for k in range(len(weights_norm_realz))])] 
    model.set_weights(weights_norm_realz[min_idx_norm][0])
    #model.set_weights(avg_weights_realz)

# A function that return a list of layers.
def get_layer_list(model=None):
    layer_keys=['pre','gsat','osat','inv','oth','rate','slack','GOR','OGR','trunc','time','bhp','psat']
    if model.cfd_type['Fluid_Type']=='dry-gas':
        layer_keys.remove('osat')
        
    if model.cfd_type['Type']!='PINN':
       layer_keys=list(set(layer_keys)-set(['rate','slack']))

    layer_list={}

    def update_layer_list(_layer,_layer_keys,_layer_list):
        if (hasattr(_layer, 'kernel_initializer') and (hasattr(_layer, 'bias_initializer')) or \
            hasattr(_layer,'recurrent_initializer')) or hasattr(_layer, '_kernel_initializer') or hasattr(_layer,'trainable_kernel'):
            for key in _layer_keys[:]:
                if key.lower() in _layer.name.lower():
                    # Check if key exist in dictionary to create or append
                    if key not in _layer_list:
                        _layer_list[key]=[layer]
                    elif layer not in _layer_list[key]:
                        _layer_list[key].append(layer)
                    break
                continue
            else:
                if 'oth' not in _layer_list.keys():
                    _layer_list['oth']=[]
                _layer_list['oth'].append(layer)
        return       
    for ix, layer in enumerate(model.layers):
        if not (hasattr(layer,'layers')):
            update_layer_list(layer,layer_keys,layer_list)
        else:
            for ix_i,layer_i in enumerate(layer.layers):
                update_layer_list(layer_i,layer_keys,layer_list)
    return layer_list
                
# A function that freezes a layer weights and biases. 
def freeze_layer(model=None, layer_prefix=['inv','oth']):
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and (hasattr(model.layers[ix], 'bias_initializer') or hasattr(model.layers[ix],'recurrent_initializer')) :
            for key in layer_prefix:
                if key in layer.name.lower():
                    layer=model.get_layer(index=ix)
                    layer.trainable=False                    
    return 

# A function that returns the rank index of a list of training losses. 
def rank_losses(losses=[],weights=[],normalization={'Type':'linear-scaling','Limits':[0,1]}):
    # Weights is a two-item list of [train_weights:['DOM','DBC','NBC','IBC','IC'], validation_weight]
    nrows=len(losses[0])
    ncols=len(losses)
    wt=weights[:ncols]
    nwt=wt/np.linalg.norm(wt,ord=1)
    
    if len(losses)!=nwt.shape[-1]:
        return print('Error: Check loss outer dimension is not equal to the inner dimension of the weights')
        
    # Define a normalization function
    def norm_func(values_list=None,normalization={'Type':'linear-scaling','Limits':[0,1]} ):
        # Compute statistics
        if normalization['Type']=='linear-scaling':
            a=normalization['Limits'][0]
            b=normalization['Limits'][1]
            values_min=np.min(values_list,axis=1)
            values_max=np.max(values_list,axis=1)
            ((np.transpose(values_list)-values_min)/(values_max-values_min))*(b-a)+a
            return ((np.transpose(values_list)-values_min)/(values_max-values_min))*(b-a)+a
        else:
            values_mean=np.mean(values_list,axis=1)
            values_std=np.std(values_list,axis=1)
            return (np.transpose(values_list)-values_mean)/values_std 
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]
    
    norm_losses=norm_func(values_list=losses,normalization=normalization)
    norm_losses=np.nan_to_num(norm_losses, copy=True, nan=0.0, posinf=None, neginf=None)
    l2_norm=np.linalg.norm((norm_losses*nwt),ord=2,axis=1)
    l2_norm_min=np.min(l2_norm)
    l2_norm_mean=np.mean(l2_norm)
    minl2_norm_idx=np.where(l2_norm==l2_norm_min)[0][0]
    return minl2_norm_idx,np.transpose(losses)[minl2_norm_idx]

# A function that returns a randomly shuffled list. 
def shuffle_list(nlist=None,npartitions=2,seed=50,shuffle_partition=False):
    # Determine the length of the list.
    list_len=len(nlist)

    partition_size=int(list_len/npartitions)
    partition_size_rem=int(list_len%npartitions)
    
    partitioned_list=np.reshape(nlist[:(list_len-partition_size_rem)],(npartitions,partition_size)).tolist()
      
    # Add the remainder to the list.
    if partition_size_rem==0:
        partitioned_list.append([])
    else:
        partitioned_list.append(nlist[-partition_size_rem:])
    
    # Shuffle each sublist.
    for sub_list in partitioned_list:
        random.Random(seed).shuffle(sub_list)

    shuffled_partitioned_list=[item for sublist in partitioned_list for item in sublist]
    if shuffle_partition:
        shuffled_partitioned_list=[]
        for i in range(max(len(partitioned_list[0]),partition_size_rem)):
            for sub_list in partitioned_list:
                if sub_list!=partitioned_list[-1]:
                    if i<len(partitioned_list[0]):
                        shuffled_partitioned_list.append(sub_list[i])
                else:
                    if i<partition_size_rem:
                        shuffled_partitioned_list.append(sub_list[i])
                        # Check if sub_list is the last index to add the remainder.            
    return shuffled_partitioned_list

# A class which generates a batched dataset during training. The batch generator class inherits the keras utility sequence.
class batch_generator(tf.keras.utils.Sequence):
    def __init__(self, xdataset, ydataset, batch_size=[16,16,16,16,16,16],shape=None,reshape=(29,29,1),shred_end_len=None,data_type=None,reshuffle_data=False,\
                 no_perturbations=6,seed=50,square_sampling=False,steps_per_epoch=None,duplicate_time=[True,3]):
        # Check the batch size.
        if len(batch_size)<len(xdataset):
            diff=abs(len(xdataset)-len(batch_size))
            batch_size=batch_size+[batch_size[-1]]*diff
        
        # Duplicates the time axis. 
        if duplicate_time[0]:
            for i in range(len(xdataset)):
                xdataset[i][-1][:,0]=xdataset[0][duplicate_time[1]]
                    
        # Resample the dataset.
        if reshuffle_data:
            # Applied to a flattened data
            import random
            if np.prod(shape)==1:
                end_idx=[int(np.prod(reshape[0:])) for i in range(len(xdataset))]
            else:
                end_idx=[np.prod(shape[i]) for i in range(len(xdataset))]
            
            # All features of the dataset are column arrays except the last feature which is a matrix array.
            # Last feature inner dimension is 7 columns or obtained from the dataset.
            end_idx_in=[xdataset[i][-1].shape[-1] for i in range(len(xdataset))]

            # Check if the last index of the list of datasets is a list.            
            # if isinstance(xdataset[0][-1],list):
            xdtset=[[np.reshape(xdataset[i][j],(no_perturbations,-1,end_idx[i])) for j in range(len(xdataset[i])-1)]+
                    [np.reshape(xdataset[i][-1],(no_perturbations,-1,end_idx[i],end_idx_in[i]))]for i in range(len(xdataset))] 
                
            ydtset=[[np.reshape(ydataset[i][j],(no_perturbations,-1,end_idx[i])) for j in range(len(ydataset[i]))] for i in range(len(ydataset))]            

            # Get the shape of the new dataset.
            new_shape=[xdtset[i][0].shape for i in range(len(xdtset))]

            # Create a list of the outer dimensions based on the new shape.
            a_list=[list(range(new_shape[i][0])) for i in range(len(new_shape))]            # No perturbations
            b_list=[list(range(new_shape[i][1])) for i in range(len(new_shape))]            # Timesteps/Perturbation
            c=[new_shape[i][2] for i in range(len(new_shape))] 
            d=[[[a,b] for b in b_list[i] for a in a_list[i]] for i in range(len(new_shape))] 

            # Shuffle the list randomly
            #for j in range(50):
                #[random.shuffle(a_list[i]) for i in range(len(a_list))]
                #[random.shuffle(b_list[i]) for i in range(len(b_list))]
            
            #a_list=[shuffle_list(nlist=a_list[i],npartitions=2,seed=50,shuffle_partition=True) for i in range(len(a_list))]
            #b_list=[shuffle_list(nlist=b_list[i],npartitions=2,seed=50,shuffle_partition=True) for i in range(len(b_list))]
            dtset_idx_list=[a_list,b_list,c,d]
            def dataset_resampler(dtset_idx_list,dtset=None,square_sampling=False,seed=50,use_seed=True):
                # The dataset index is passed by object reference, which allows direct update (random.shuffle) to the values. 
                no_features=[len(dtset[i]) for i in range(len(dtset))] 
                dtset_new=[]
                c=dtset_idx_list[2]
                for i in range(len(dtset)):
                    dtset_features=[]
                    if not square_sampling:
                        if use_seed:
                            # random.seed(seed)
                            random.Random(seed).shuffle(dtset_idx_list[0][i])
                            random.Random(seed).shuffle(dtset_idx_list[1][i])
                        for j in range(no_features[i]):
                            dtset_samples=[]
                            for b in dtset_idx_list[1][i]:
                                for a in dtset_idx_list[0][i]:
                                    if dtset_idx_list[0][i].index(a)==0 and dtset_idx_list[1][i].index(b)==0:
                                        dtset_samples=dtset[i][j][a,b,0:c[i],...]
                                    else:
                                        dtset_samples=np.concatenate([dtset_samples,dtset[i][j][a,b,0:c[i],...]],axis=0)
                            dtset_features.append(dtset_samples)
                    else:
                        if use_seed:
                            random.Random(seed).shuffle(dtset_idx_list[3][i])
                        for j in range(no_features[i]):
                            for sp in dtset_idx_list[3][i]:
                                if dtset_idx_list[3][i].index(sp)==0:
                                    dtset_samples=dtset[i][j][sp[0],sp[1],0:c[i],...]
                                else:
                                    dtset_samples=np.concatenate([dtset_samples,dtset[i][j][sp[0],sp[1],0:c[i],...]],axis=0)
                            dtset_features.append(dtset_samples)                               
                    dtset_new.append(dtset_features)
                return dtset_new
            
            xdataset=dataset_resampler(dtset_idx_list,dtset=xdtset,square_sampling=square_sampling,seed=seed,)
            ydataset=dataset_resampler(dtset_idx_list,dtset=ydtset,square_sampling=square_sampling,seed=seed,use_seed=False)
        if shred_end_len==None:
            self.X = xdataset
            self.y = ydataset
        else:
            def regroup(x):
                b=[]
                for i in range(len(x)):
                    a=[]
                    for j in range(len(x[i])):
                        a.append(x[i][j][0:len(x[i][j])-shred_end_len])
                    b.append(a)
                else:
                    return b
            self.X=regroup(xdataset)
            self.y=regroup(ydataset)
        self.batch_size = batch_size
        self.id=[]                                                                  # Unique ID for indexing the batch data.
        self.tstep_shape=shape
        self.data_type=data_type
        
    def __len__(self):
        l=0
        self.id=[]
        for n in range(0, len(self.X)):
            # Check if the dataset item is empty.
            if self.X[n][0].shape[0]==0:
                continue
            # Determine the length of each item i.e., DOM, DBC, NBC, etc. --should correspond to the batch_size index.
            batch_size_flat=self.batch_size[n]*np.prod(self.tstep_shape[n])
            if batch_size_flat>=len(self.X[n][0]):
                batch_size_flat=len(self.X[n][0])                    
                
            nn=tf.cast(tf.math.ceil(self.X[n][0].shape[0]/batch_size_flat),dtype=tf.int32)  # Ceil value to ensure any remainder is added as a separate batch
            l=l+nn
            for i in range(0, nn):
                # Determine starting and ending slice indexes for the current.
                start = i * batch_size_flat
                end = start + batch_size_flat
                if i==nn-1:                                                         # Last index of the loop.
                    batch_rem=self.X[n][0].shape[0]%batch_size_flat
                    if batch_rem==0:
                        end = start+batch_size_flat+batch_rem               
                    else:
                        end=start+batch_rem
                st_end=[n,start,end]
                self.id.append(st_end)
        return l

    def __getitem__(self, index):
        if len(self.X)!=len(self.y):
            return print('Check data set arrangment!')
        l=0
        for n in range(0, len(self.X)):
            # Check if the dataset item is empty. 
            if self.X[n][0].shape[0]==0:
                continue
            # Determine the length of each item i.e., DOM, DBC, NBC, etc. --should correspond to the batch_size index. 
            batch_size_flat=self.batch_size[n]*np.prod(self.tstep_shape[n])
            if batch_size_flat>=len(self.X[n][0]):
                batch_size_flat=len(self.X[n][0])                    
                            
            nn=tf.cast(tf.math.ceil(self.X[n][0].shape[0]/batch_size_flat),dtype=tf.int32)
            for i in range(0, nn):
                # Determine starting and ending slice indexes for the current
                start = i * batch_size_flat
                end = start + batch_size_flat
                if i==nn-1:                                                         # Last index of the loop.
                    batch_rem=self.X[n][0].shape[0]%batch_size_flat
                    if batch_rem==0:
                        end = start+batch_size_flat+batch_rem                       # Remainder as a separate batch.
                    else:
                        end=start+batch_rem
                if l==index:
                    # Create a slice of the list along the batch.
                    if self.tstep_shape==None or len(self.tstep_shape[n])==1:
                        xi=[self.X[n][j][start:end] for j in range(0,len(self.X[n]))]
                        yi=[self.y[n][j][start:end] for j in range(0,len(self.y[n]))] 
                    else:
                        new_shape=(-1,*self.tstep_shape[n])
                        new_shape_1=(-1,*self.tstep_shape[n],self.X[n][-1].shape[1])
                        xi=[np.reshape(self.X[n][j][start:end],new_shape) for j in range(0,len(self.X[n])-1)]+[np.reshape(self.X[n][len(self.X[n])-1][start:end],new_shape_1)]
                        yi=[np.reshape(self.y[n][j][start:end],new_shape) for j in range(0,len(self.y[n]))]                         
                    return xi,yi
                l=l+1
    
# A callback class that gets the total time spent during training. 
class adaptiveRegularizer(tf.keras.callbacks.Callback):
    def __init__(self,epochs={'Frac':0.8,'Total':500},*args,**kwargs):
        # epoch['Frac'] is the % of the ending epoch
        super().__init__( *args, **kwargs)
        self.epochs=epochs
        self.epoch_limit=tf.cast(tf.math.floor(self.epochs['Total']*(1-self.epochs['Frac'])),dtype=tf.int32)
        self.wbl_epoch={'epoch':[],'loss':[],'val_loss':[],'weight_bias':[],'train_time':[]}
        
        self.epoch_time=[]
        self.epoch_start_time=0.
        self.wt_final=1e-16

        self.decay_step=100
    def on_train_begin(self,logs={}):
        self.wbl_epoch={'epoch':[],'loss':[],'val_loss':[],'weight_bias':[],'train_time':[]}
        self.epoch_time=[]      

        wt_init=self.model.nwt[5]
        self.decay_rate=(1-(self.wt_final/wt_init)**(1/max(self.epochs['Total']//self.decay_step,1)))
    
    def on_epoch_begin(self, epoch, logs={}):
        super().on_epoch_begin(epoch, logs)
        self.epoch_start_time = time.time()

        # Resets the cumulative variables, if defined. 
        [self.model.cum[key].assign(0) for key in ['Gas_Pred','Gas_Obs','Cum_N']]

        self.model.batch_seed_no['numpy']=0
        self.model.epoch_idx=epoch 

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        if epoch>=self.epoch_limit:
            self.wbl_epoch['epoch'].append(epoch)
            self.wbl_epoch['loss'].append(logs.get('loss'))
            self.wbl_epoch['val_loss'].append((logs.get('val_loss')))
            self.wbl_epoch['weight_bias'].append(self.model.get_weights())
            self.wbl_epoch['train_time'].append(time.time() - self.epoch_start_time)
        self.epoch_time.append(time.time() - self.epoch_start_time)
            
    def on_train_end(self,logs={}):
        # Get the index of the minimum training and validation loss.
        min_train_loss=min(self.wbl_epoch['loss'])
        if all(x is None for x in self.wbl_epoch['val_loss']):
            # Val loss not computed.
            min_val_loss=0.
            min_loss_idx=self.wbl_epoch['loss'].index(min_train_loss)
        else:
            if self.model.cfd_type['Type']!='PINN':
                min_val_loss=min(self.wbl_epoch['val_loss'])
                min_loss_idx=self.wbl_epoch['val_loss'].index(min_val_loss)
            else:
                # Use a weighted training and validation loss.
                min_loss_idx,_=rank_losses(losses=[self.wbl_epoch['loss'],self.wbl_epoch['val_loss']],weights=[1.0,0.],normalization={'Type':'linear-scaling','Limits':[0,1]})
                min_val_loss=self.wbl_epoch['val_loss'][min_loss_idx]
        
        best_epoch=self.wbl_epoch['epoch'][min_loss_idx]                            # 1 is added during reporting epoch (e.g. plots) is not zero-based.
        best_weight_bias=self.wbl_epoch['weight_bias'][min_loss_idx]
        train_time=sum(self.epoch_time)
        
        # Update the model with the best weights and biases.
        # self.model.set_weights(best_weight_bias)
        
        # Update the model's weights and biases ensemble list
        self.model.wblt_epoch_ens.append({'epoch':best_epoch,'loss':min_train_loss,'val_loss':min_val_loss,'weight_bias':best_weight_bias,'train_time':train_time})
        
        self.model.history_ens.append(self.model.history.history)
        
        # Add all watched model parameters for manual tuning.
        self.model.wbl_epoch.append(self.wbl_epoch)
        
        # Clear the parameters.
        self.wbl_epoch={'epoch':[],'loss':[],'val_loss':[],'weight_bias':[],'train_time':[]}
        self.epoch_time=[]
        
    def remove_none(number):
        if number==None:
             number=0.
        return number

# ================================================== Learning Rate Scheduler =====================================================    
# A class that performe Cosine Annealing with Warm Restarts.
class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule with restarts.
  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.
  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function with
  restarts to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.
  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times smaller initial learning rate.
  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
  ```
  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    """Applies cosine decay with restarts to the learning rate.
    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    """
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "SGDRDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      first_decay_steps = tf.cast(self.first_decay_steps, dtype)
      alpha = tf.cast(self.alpha, dtype)
      t_mul = tf.cast(self._t_mul, dtype)
      m_mul = tf.cast(self._m_mul, dtype)

      global_step_recomp = tf.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        """Helper for `cond` operation."""
        if geometric:
          i_restart = tf.floor(
              tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              tf.math.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = tf.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = tf.cond(
          tf.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
          tf.constant(math.pi) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return tf.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }

# A class that performs the constant-exponential decay. 
class ConstantExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,initial_learning_rate,decay_params={'Start':[],'End':[],'Steps':[],'Rate':[],'Stair_Case':[True,True]},name=None):
        super(ConstantExponentialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_params=decay_params
        self.name=name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            start_decay_step = tf.cast(self.decay_params['Start'], dtype)
            end_decay_step = tf.cast(self.decay_params['End'], dtype)            
            decay_steps=tf.cast(self.decay_params['Steps'], dtype)
            decay_rate=tf.cast(self.decay_params['Rate'],dtype)
            global_step=tf.cast(step,dtype)

            x_0=(tf.math.minimum(global_step,end_decay_step[0])-start_decay_step[0])/decay_steps[0]
            x_0=tf.cond(tf.math.equal(self.decay_params['Stair_Case'][0],True),lambda:tf.math.floor(x_0),lambda:x_0)
            x_0 = tf.cast(x_0 > 0, x_0.dtype) * x_0
            lr_0=initial_learning_rate * tf.math.exp(-decay_rate[0]*x_0)
            
            x_1=(tf.math.minimum(global_step,end_decay_step[1])-start_decay_step[1])/decay_steps[1]
            x_1=tf.cond(tf.math.equal(self.decay_params['Stair_Case'][1],True),lambda:tf.math.floor(x_1),lambda:x_1)
            x_1 = tf.cast(x_1 > 0, x_1.dtype) * x_1
            lr_1=lr_0 * tf.math.exp(-decay_rate[1]*x_1)
            
            y_1=(global_step-start_decay_step[1])
            lr=tf.cast(y_1 < 0, y_1.dtype)*lr_0+tf.cast(y_1 >= 0, y_1.dtype)*lr_1
            return lr

    def get_config(self):
        return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_params":self.decay_params,
        "name":self.name}

