#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# © 2022 Victor Molokwu <vcm1@hw.ac.uk>
# Distributed under terms of the MIT license.
# A module of functions used for visualization of the model outputs.

import os
import batch_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU: -1

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import font_manager
import datetime
import time

plt.rcParams["font.family"] =["Times New Roman"]
plt.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams['figure.dpi'] = 600   
#================================================================================================================================
# A function that returns a plot of the MSE losses vs. epoch during training
def plot_history(model,func=None,dname=None,tr_time=None,dump_excel=True,folder_path=None):
    if model.cfd_type['Optimizer']=='first-order':
        tr_loss=model.history_ens[model.best_ens]
        dic_keys_list=list(tr_loss)
        
        # Create Excel dump file if any
        if dump_excel and folder_path is not None:
            excel_file_path=f'{folder_path}/MSE_losses.xlsx'
            excel_data={}           
        for i in range(0,len(dic_keys_list)):
            if dic_keys_list[i]=='val_loss':
                plt.plot(tr_loss[dic_keys_list[i]], label='MSE: '+dic_keys_list[i],linewidth=2.0)
                continue
            plt.plot(tr_loss[dic_keys_list[i]], label='MSE: '+dic_keys_list[i])
            if dump_excel and folder_path is not None:
                excel_data[dic_keys_list[i]]=list(tr_loss[dic_keys_list[i]])
            
        # Prints the best epoch using the saved best model parameters
        best_epoch=model.wblt_epoch_ens[model.best_ens]['epoch']
        total_ens_train_time=sum([model.wblt_epoch_ens[j]['train_time'] for j in range(len(model.wblt_epoch_ens))])/60.
        best_ens_train_time=model.wblt_epoch_ens[model.best_ens]['train_time']/60.
        
        str1=model.cfd_type['DNN_Type'][0:int((3/5)*len(model.cfd_type['DNN_Type']))]
        str2=model.cfd_type['DNN_Type'][int((3/5)*len(model.cfd_type['DNN_Type'])):]
        if tr_time==None:
            train_time_text='Total ens. training time: {:.2f} mins \nBest ens. training time: {:.2f} mins \nBest epoch: {}  Best ens.: {} \nHLayer config.: {}\n{}'.\
            format(total_ens_train_time,best_ens_train_time,best_epoch+1,model.best_ens+1,str(dname)+'-'+str1,str2)
        else:
            train_time_text='Total ens. training time: {:.2f} mins \nBest ens. training time: {:.2f} mins \nBest epoch: {}  Best ens.: {} \nHLayer config.: {}\n{}'.\
            format(tr_time,tr_time,best_epoch+1,model.best_ens+1,str(dname)+'-'+str1,str2)

    else:
        labels_list=func.hist_loss['loss_keys']
        tr_loss=func.hist_loss['loss_values']
        for i in range(0,len(tr_loss[0])):
        # Plot history: MSE
            if labels_list[i]=='loss':
                plt.plot([tr_loss[j][i] for j in range(len(tr_loss))], label='MSE: '+labels_list[i])      # linewidth=2.0, color='black', zorder=len(tr_loss)-1
                continue       
            plt.plot([tr_loss[j][i] for j in range(len(tr_loss))], label='MSE: '+labels_list[i])
        # Print the total train time on plot
        train_time_text='Total ens. training time: {:.2f} mins'.\
        format(tr_time)
    axis=plt.gca()
    plt.text(0.525, 0.90,train_time_text, size=7.0, ha='center', va='center', transform=axis.transAxes, zorder=100)
    try:
        fldr_name=os.path.basename(__file__)[0:4]
    except:
        fldr_name=''
    plt.title(fldr_name+'_MSE for Non Physics-Informed Training')
    plt.ylabel('MSE value')
    plt.xlabel('No. Iterations/Epoch')
    plt.yscale('log')
    plt.legend(loc="upper left",  prop={'size': 8},ncol=1,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(1, 1))  
    plt.rc('grid', linestyle="--", color='grey',linewidth=0.5)
    plt.grid(True)
   
    plt.show()
    if dump_excel and folder_path is not None:
        df=pd.DataFrame(excel_data).transpose()
        df.to_excel(excel_writer=excel_file_path,sheet_name='Sheet_'+str(datetime.datetime.now())[:20].replace(':','_'))

    return

# A function that returns a plot of the average grid predictions vs. time for the different quantities of interest (e.g., pressure, gas saturation and condensate saturation).
def predict_plot_avgout(model=None, obs_fdata=None,obs_ldata=None,shape=(29,29),bound_int=50,output_keys=['Pressure','Gas Saturation','Condensate Saturation'],\
                        output_colours=['lightcoral','limegreen','orange'],output_linestyles=['dashed','dotted','dashdot'],no_perturbations=6,plot_range=[0,1,2,3,4,5],\
                        dump_excel=True,folder_path=None):
    def abs_error(x_pred,x_obs):
        return tf.math.abs((x_pred-x_obs)/x_obs)
    full_shape=(no_perturbations,-1,*shape)
    start_time=time.time()
    pred_ldata=model.predict(obs_fdata)
    end_time=time.time()
    tr_time=print((end_time-start_time))
    # Un normalize if outputs are normalized prior to training
    if model.cfd_type['Output_Normalization']!=None:
        a=model.cfd_type['Norm_Limits'][0]
        b=model.cfd_type['Norm_Limits'][1]
        ts=pd.DataFrame(model.ts.numpy(),index=model.ts_idx_keys[0],columns=model.ts_idx_keys[1])
        for key in output_keys:
            if model.cfd_type['Output_Normalization']=='linear-scaling':
                 nonorm_pred=((pred_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
                 nonorm_obs=((obs_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
            else:
                nonorm_pred=pred_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
                nonorm_obs=obs_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
            pred_ldata[output_keys.index(key)]=nonorm_pred
            obs_ldata[output_keys.index(key)]=nonorm_obs
    
    nonflat_pred_ldata=[np.reshape(pred_ldata[i],full_shape) for i in range(len(output_keys))]
    nonflat_obs_ldata=[np.reshape(obs_ldata[i],full_shape) for i in range(len(output_keys))]
    
    avg_pred_ldata=[]
    avg_obs_ldata=[]
    mae=[]
    # Using the weighted volume to determine the average pressure, gas and condensate saturations
    # Porosity is index index 4
    nonflat_time=np.reshape(obs_fdata[3],full_shape)
    nonflat_phi=np.reshape(obs_fdata[4],full_shape)
    nonflat_permx=np.reshape(obs_fdata[5],full_shape)

    for i in range(len(nonflat_pred_ldata)):
        nonflat_pred_ldata[i],nonflat_obs_ldata[i],nonflat_time[i],_,nonflat_permx[i]=unnormalize_2D(model,nonflat_pred_ldata[i],nonflat_obs_ldata[i],inputs={'time':nonflat_time[i],'phi':nonflat_phi[i],'permx':nonflat_permx[i]},output_keys=output_keys)                

    for i in range(nonflat_pred_ldata[0].shape[0]):
        avg_pred_ldata_per_pert={key:[] for key in output_keys}
        avg_obs_ldata_per_pert={key:[] for key in output_keys}
        for j in range(nonflat_pred_ldata[0].shape[1]):
            for k in range(len(output_keys)):
                # Volumetric weighted average
                avg_pred_ldata_per_pert[output_keys[k]].append(np.sum(nonflat_pred_ldata[k][i,j,:,:]*nonflat_phi[i,j,:,:])/np.sum(nonflat_phi[i,j,:,:]))
                avg_obs_ldata_per_pert[output_keys[k]].append(np.sum(nonflat_obs_ldata[k][i,j,:,:]*nonflat_phi[i,j,:,:])/np.sum(nonflat_phi[i,j,:,:]))
        avg_pred_ldata.append(avg_pred_ldata_per_pert)  
        avg_obs_ldata.append(avg_obs_ldata_per_pert) 
    
    # Print output on screen
    print('Pred_Avg {}  Obs_Avg {}  ==  Pred_Avg {}  Obs_Avg {}\n'\
          .format(output_keys[0],output_keys[0],output_keys[1],output_keys[1]))
   
    for i in range(len(avg_pred_ldata)):
        mae_per_pert={key:[] for key in output_keys}
        for j in range(0,len(avg_pred_ldata[i][output_keys[0]]),int(np.ceil(len(avg_pred_ldata[i][output_keys[0]])*0.1))):
            for k in output_keys:
                mae_per_pert[k].append(abs_error(avg_pred_ldata[i][k][j],avg_obs_ldata[i][k][j]))
            print(avg_pred_ldata[i][output_keys[0]][j], ' ', avg_obs_ldata[i][output_keys[0]][j],'==', avg_pred_ldata[i][output_keys[1]][j], ' ', avg_obs_ldata[i][output_keys[1]][j] )
        mae.append(mae_per_pert)
    
    for i in range(len(mae)):
        print(f'MAE {output_keys[0]}: {tf.math.reduce_mean(mae[i][output_keys[0]])}\nMAE {output_keys[1]}: {tf.math.reduce_mean(mae[i][output_keys[1]])})')

    # Create Excel dump file if any
    if dump_excel and folder_path is not None:
        excel_file_path=f'{folder_path}/avgkx_{(np.mean(nonflat_permx)):.2f}mD_{no_perturbations}test_realiz.xlsx'
        excel_data={}

    # Plot the predicted and observed averages for all the metrics
        # Create the subplots
    plt_height=(len(avg_pred_ldata))
    plt_width=len(output_keys)
    fig, ax = plt.subplots(plt_height,plt_width,dpi=1200)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.95, wspace=0.4)
    fig.suptitle('Average '+', '.join(output_keys).lower(),fontsize=9,y=1.01,weight='bold')
    
    for i in range(len(plot_range)):
        for j in range(len(output_keys)):
            ax[i,j].plot(avg_pred_ldata[plot_range[i]][output_keys[j]], label='Predicted Average '+output_keys[j], color='#0343DF', linestyle=output_linestyles[j],linewidth=1.5,zorder=100)      # Color azure
            ax[i,j].plot(avg_obs_ldata[plot_range[i]][output_keys[j]], label='Observed Average '+output_keys[j], linestyle='', marker='s', markerfacecolor=output_colours[j],markeredgecolor='#A9561E',markersize=4,markeredgewidth=0.75) #CB9978 #A9561E
            
            if i==0:
                ax[i,j].set_title(f'Avg. {output_keys[j][0:5]}, Realiz.({i+1})',fontsize=7.5, y=0.95)
                if j==2:
                    fig.legend(loc="upper left",  prop={'size': 6},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.01))  
            else:
                ax[i,j].set_title(f'Realiz. ({i+1})',fontsize=7.5, y=0.95)
            
            if output_keys[j].lower()=='pressure': 
                units='(psia)'
            else: 
                units=''           
            
            ax[i,j].set_ylabel(f'{output_keys[j][0:3]} {units}',fontsize=6)
            if i==len(plot_range)-1:
                ax[i,j].set_xlabel('Timesteps',fontsize=6)
            
            ax[i,j].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)
            ax[i,j].set_xticks(np.linspace(0,len(avg_pred_ldata[plot_range[i]][output_keys[j]]),6))
            if output_keys[j].lower()[0:3]=='pre':
                round_fac=100
            elif output_keys[j].lower()[0:3]=='gas':
                round_fac=0.05
            else:
                round_fac=0.025
                #ax[i,j].set_yticks(np.round_(np.linspace(0.5,1,7),2))

            # Get the min and max values for the plot. The min and max is scaled by a factor 0f 0.9 and 1.10 respectively, then rounded to the nearest 50
            y_min=np.floor((0.9*np.min([np.min(avg_pred_ldata[plot_range[i]][output_keys[j]]),np.min(avg_obs_ldata[plot_range[i]][output_keys[j]])]))/round_fac)*round_fac
            y_max=np.ceil((1.1*np.max([np.max(avg_pred_ldata[plot_range[i]][output_keys[j]]),np.max(avg_obs_ldata[plot_range[i]][output_keys[j]])]))/round_fac)*round_fac
            n_ticks=5
        
            ax[i,j].set_yticks(np.linspace(y_min,y_max,n_ticks))
            
            if j==0:
                ax[i,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
            elif j==1:
                ax[i,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
            else:
                ax[i,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
            
            ax[i,j].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
            ax[i,j].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
    
            if dump_excel and folder_path is not None:
                if 'Time' not in excel_data.keys():
                    excel_data['Time']=list(nonflat_time[plot_range[i]])   #Use the zero index

                excel_header_name_obs=f'{output_keys[j][0:3]}_{np.mean(nonflat_permx[plot_range[i]][...]):.2f}mD_Avg_Obs'
                excel_header_name_pred=f'{output_keys[j][0:3]}_{np.mean(nonflat_permx[plot_range[i]][...]):.2f}mD_Avg_Pred'
                excel_data[excel_header_name_pred]=list(avg_pred_ldata[plot_range[i]][output_keys[j]])
                excel_data[excel_header_name_obs]=list(avg_obs_ldata[plot_range[i]][output_keys[j]])

            
            #plt.text(0.800, 0.95,f'Number of Perturbations: {no_perturbations}', size=7.0, ha='center', va='center', transform=axes1.transAxes, zorder=100)
    #[ax[len(plot_range)-1,k].legend(loc="lower left",  prop={'size': 8},ncol=1,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -2.25)) for k in range(len(output_keys))] 
    #from matplotlib.legend import _get_legend_handles_labels
    plt.show()
    if dump_excel and folder_path is not None:
        df=pd.DataFrame(excel_data).transpose()
        df.to_excel(excel_writer=excel_file_path)
    return avg_pred_ldata,avg_obs_ldata

# A function that returns a plot of the wells' grid block predictions vs. time for the different quantities of interest (e.g., pressure, gas saturation and condensate saturation).
def predict_plot_wells(model=None, obs_fdata=None,obs_ldata=None,wells_idx=[(14,14)],shape=(29,29),bound_int=50,output_keys=['Pressure','Gas Saturation','Condensate Saturation'],output_colours=['lightcoral','limegreen','orange'],output_linestyles=['dashed','dotted','dashdot'],no_perturbations=6,plot_range=[0,1,2,3,4,5]):
    def abs_error(x_pred,x_obs):
        return tf.math.abs((x_pred-x_obs)/x_obs)
    full_shape=(no_perturbations,-1,*shape)
    pred_ldata=model.predict(obs_fdata)
    # Un normalize if outputs are normalized prior to training
    if model.cfd_type['Output_Normalization']!=None:
        a=model.cfd_type['Norm_Limits'][0]
        b=model.cfd_type['Norm_Limits'][1]
        ts=pd.DataFrame(model.ts.numpy(),index=model.ts_idx_keys[0],columns=model.ts_idx_keys[1])
        for key in output_keys:
            if model.cfd_type['Output_Normalization']=='linear-scaling':
                 nonorm_pred=((pred_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
                 nonorm_obs=((obs_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
            else:
                nonorm_pred=pred_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
                nonorm_obs=obs_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
            pred_ldata[output_keys.index(key)]=nonorm_pred
            obs_ldata[output_keys.index(key)]=nonorm_obs
    
    nonflat_pred_ldata=np.stack([np.reshape(pred_ldata[i],full_shape) for i in range(len(output_keys))],axis=-1)
    nonflat_obs_ldata=np.stack([np.reshape(obs_ldata[i],full_shape) for i in range(len(output_keys))],axis=-1)
    mae=[]
    # Print output on screen
    print('Pred_Well {}  Obs_Well {}  ==  Pred_Well {}  Obs_Well {}\n'\
          .format(output_keys[0],output_keys[0],output_keys[1],output_keys[1]))
   
    for i in range(np.shape(nonflat_pred_ldata)[0]):
        mae_per_pert={key:[] for key in output_keys}
        for j in range(0,np.shape(nonflat_pred_ldata)[1],int(np.ceil(np.shape(nonflat_pred_ldata)[1]*0.1))):
            for k in wells_idx:
                for kk in range(len(output_keys)):
                    mae_per_pert[output_keys[kk]].append(abs_error(nonflat_pred_ldata[i,j,k[0],k[1],kk],nonflat_obs_ldata[i,j,k[0],k[1],kk]))
                    if kk!=0:
                        print('==',end='')
                    print(nonflat_pred_ldata[i,j,k[0],k[1],kk], ' ', nonflat_obs_ldata[i,j,k[0],k[1],kk], end='')
                print('\r')
        mae.append(mae_per_pert)
        
    for i in range(len(mae)):
        for j in range(len(output_keys)):
            print(f'MAE {output_keys[j]}: {tf.math.reduce_mean(mae[i][output_keys[j]])}\n',end='')
        

    # Plot the predicted and observed averages for all the metrics
        # Create the subplots
    fig, ax = plt.subplots(len(plot_range),len(output_keys),dpi=1200)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.95, wspace=0.4)
    fig.suptitle('Wells '+', '.join(output_keys).lower(),fontsize=9,y=1.02,weight='bold')
    
    for i in range(len(plot_range)):
        for k in wells_idx:
            for kk in range(len(output_keys)):
                ax[i,kk].plot(list(nonflat_pred_ldata[i,:,k[0],k[1],kk]), label='Predicted Well '+output_keys[kk], color='#0343DF', linestyle=output_linestyles[kk],linewidth=1.5,zorder=100)      # Color azure
                ax[i,kk].plot(list(nonflat_obs_ldata[i,:,k[0],k[1],kk]), label='Observed Well '+output_keys[kk], linestyle='', marker='s', markerfacecolor=output_colours[kk],markeredgecolor='#A9561E',markersize=4,markeredgewidth=0.75) #CB9978 #A9561E
                
                if i==0:
                    ax[i,kk].set_title(f'Well. {output_keys[kk][0:5]}, Realiz.({i+1})',fontsize=7.5, y=0.95)
                    if j==2:
                        fig.legend(loc="upper left",  prop={'size': 6},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.01))  
                else:
                    ax[i,kk].set_title(f'Realiz. ({i+1})',fontsize=7.5, y=0.95)
                
                if output_keys[kk].lower()=='pressure': 
                    units='(psia)'
                else: 
                    units=''           
                
                ax[i,kk].set_ylabel(f'{output_keys[kk][0:3]} {units}',fontsize=6)
                if i==len(plot_range)-1:
                    ax[i,kk].set_xlabel('Timesteps',fontsize=6)
                
                ax[i,kk].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)
                ax[i,kk].set_xticks(np.linspace(0,nonflat_obs_ldata[i,:,k[0],k[1],kk].shape[0],6))
                if output_keys[kk].lower()[0:3]=='pre':
                    round_fac=100
                elif output_keys[kk].lower()[0:3]=='gas':
                    round_fac=0.05
                else:
                    round_fac=0.025
                    #ax[i,kk].set_yticks(np.round_(np.linspace(0.5,1,7),2))
    
                # Get the min and max values for the plot. The min and max is scaled by a factor 0f 0.9 and 1.10 respectively, then rounded to the nearest 50
                y_min=np.floor((0.9*np.min([np.min(nonflat_pred_ldata[i,:,k[0],k[1],kk]),np.min(nonflat_obs_ldata[i,...,k[0],k[1],kk])]))/round_fac)*round_fac
                y_max=np.ceil((1.1*np.max([np.max(nonflat_pred_ldata[i,:,k[0],k[1],kk]),np.max(nonflat_obs_ldata[i,...,k[0],k[1],kk])]))/round_fac)*round_fac
                n_ticks=5
            
                ax[i,kk].set_yticks(np.linspace(y_min,y_max,n_ticks))
                
                if j==0:
                    ax[i,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                elif j==1:
                    ax[i,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
                else:
                    ax[i,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
                
                ax[i,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
                ax[i,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
    plt.show()
    
    return nonflat_pred_ldata,nonflat_obs_ldata    
#pred_accuracy = tf.keras.metrics.Accuracy()

# A function that returns an image plot of the grid, at selected time point indexes, for the different quantities of interest (e.g., pressure, gas saturation and condensate saturation).
def img_plot(model,obs_fdata=None, obs_ldata=None, tstep_idx=[0,10,20,40,80],shape=(29,29),bound_int=50,plot_title=['Pressure','Gas Saturation','Condensate Saturation'],cmap=[plt.cm.Reds,plt.cm.Greens,plt.cm.Oranges,plt.cm.Blues],no_perturbations=6,timestep=10):
    full_shape=(no_perturbations,-1,*shape)
    max_tstep_idx=int(obs_fdata[0].shape[0]/(shape[0]*shape[1]*no_perturbations))
    
    for val in tstep_idx:
        if int(val) > 100:
            tstep_idx.remove(val)
    tstep_idx=[int(tstep_idx[i]*max_tstep_idx/100.) for i in range(len(tstep_idx))]
    x_mgrid=np.reshape(obs_fdata[0],full_shape)
    y_mgrid=np.reshape(obs_fdata[1],full_shape) 
    extent_ =[[x_mgrid[i].min(), x_mgrid[i].max(), y_mgrid[i].min(), y_mgrid[i].max()] for i in range(x_mgrid.shape[0])]                 
    
    pred_ldata=model.predict(obs_fdata)
    if model.cfd_type['Output_Normalization']!=None:
        a=model.cfd_type['Norm_Limits'][0]
        b=model.cfd_type['Norm_Limits'][1]
        ts=pd.DataFrame(model.ts.numpy(),index=model.ts_idx_keys[0],columns=model.ts_idx_keys[1])
        output_keys=['pressure','sgas']
        for key in output_keys:
            if model.cfd_type['Output_Normalization']=='linear-scaling':
                nonorm_obs=((obs_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
                nonorm_pred=((pred_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
            else:
                nonorm_obs=obs_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
                nonorm_pred=pred_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
            pred_ldata[output_keys.index(key)]=nonorm_pred
            obs_ldata[output_keys.index(key)]=nonorm_obs
    
    pred_ldata_mgrid=[np.reshape(pred_ldata[i],full_shape) for i in range(len(plot_title))]  
    obs_ldata_mgrid=[np.reshape(obs_ldata[i],full_shape) for i in range(len(plot_title))]    
    res_mgrid=[(tf.math.abs((obs_ldata_mgrid[i]-pred_ldata_mgrid[i]))/obs_ldata_mgrid[i])*100 for i in range(len(plot_title))]

    res_min=[tf.constant(0.,dtype=res_mgrid[i].dtype) for i in range(len(plot_title))]
    res_max=[tf.constant(10.,dtype=res_mgrid[i].dtype) for i in range(len(plot_title))]

    import matplotlib.gridspec as gridspec 
    for i in range(no_perturbations):
        
        # z=+/-5% observed value
        #z_min = [1.00*np.min(obs_ldata_mgrid[l][i,0:np.max(tstep_idx),:,:]) for l in range(len(plot_title))]    
        #z_max = [1.00*np.max(obs_ldata_mgrid[l][i,0:np.max(tstep_idx),:,:]) for l in range(len(plot_title))]
        
        for k in range(len(plot_title)):
            height_ratios=[1]*len(tstep_idx); width_ratios=[1]*3
            fig, ax = plt.subplots(len(tstep_idx),3,figsize=(5,7),dpi=1200)                  #gridspec_kw={'height_ratios':height_ratios,'width_ratios':width_ratios,'wspace':0.4}
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.4, wspace=0.4)
            fig.suptitle(f'Predicted and Observed {plot_title[k]}; Residual Maps - Realization ({i+1})',fontsize=7,y=1.01,weight='bold')

            for j in range(len(tstep_idx)):
                #res_min=[1.00*np.min(res_mgrid[l][i,tstep_idx[j],:,:]) for l in range(len(plot_title))]
                #res_max=[1.00*np.max(res_mgrid[l][i,tstep_idx[j],:,:]) for l in range(len(plot_title))]

                z_min = [1.00*np.min(obs_ldata_mgrid[l][i,tstep_idx[j],:,:]) for l in range(len(plot_title))]    
                z_max = [1.00*np.max(obs_ldata_mgrid[l][i,tstep_idx[j],:,:]) for l in range(len(plot_title))]
                
                img1=ax[j,0].imshow(pred_ldata_mgrid[k][i,tstep_idx[j],:,:],cmap =cmap[k], vmin = z_min[k], vmax = z_max[k],interpolation='nearest',origin='lower')
                img2=ax[j,1].imshow(obs_ldata_mgrid[k][i,tstep_idx[j],:,:],cmap =cmap[k], vmin = z_min[k], vmax = z_max[k],interpolation='nearest',origin='lower')
                img3=ax[j,2].imshow(res_mgrid[k][i,tstep_idx[j],:,:],cmap =cmap[-1], vmin = res_min[k], vmax = res_max[k],interpolation='nearest',origin='lower')
                
                if j==0:
                    ax[j,0].set_title(f'Pred. {plot_title[k][0:5]} Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                    ax[j,1].set_title(f'Obs. {plot_title[k][0:5]} Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                    ax[j,2].set_title(f'%Residual. {plot_title[k][0:5]} Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                else:
                    ax[j,0].set_title(f'Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                    ax[j,1].set_title(f'Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                    ax[j,2].set_title(f'Time={(tstep_idx[j]+1)*timestep} D',fontsize=7,y=0.95)
                
                ax[j,0].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                ax[j,1].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                ax[j,2].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                
                divider=[make_axes_locatable(ax[j,l]) for l in range(3)]
                cax = [divider[l].append_axes("right", size="8%", pad=0.05) for l in range(3)]
                
                cax[0].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
                cax[1].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
                cax[2].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
    
                if k==1:
                    cbar_fmt=mpl.ticker.FormatStrFormatter('%.3f')
                else:
                    cbar_fmt=None
                    
                fig.colorbar(img1,cax=cax[0],format=cbar_fmt)
                fig.colorbar(img2,cax=cax[1],format=cbar_fmt)
                fig.colorbar(img3,cax=cax[2],format=cbar_fmt)
            
            plt.show()
    return

# A function that returns a plot of the averaged grid quantity vs. time for different quantities of interest (e.g., pressure, gas saturation and condensate saturation).
# This is used with a 2D model output (e.g., convolutional-based encoder-decoder network).
def predict_plot_avgout_2D(model=None, BG_test=None,output_idx=[],output_keys=['Pressure','Gas Saturation','Condensate Saturation'],output_colours=['lightcoral','limegreen','orange'],output_linestyles=['dashed','dotted','dashdot'],no_realizations=None,perm=[],\
                           legend_interval=2,xy_label_size=7.5,no_subplots=3,xy_limits={'y':None},group_plots=False,plot_all_series=False, box_plots=False):
    def abs_error(x_pred,x_obs):
        return tf.math.abs((x_pred-x_obs)/x_obs)

    # Get string index.
    if output_idx in [[], None]:
        output_idx=list(range(len(output_keys)))
        for idx, value in enumerate(output_keys):
            if 'pre' in value.lower():
                output_idx[idx]=0
            elif 'gas' in value.lower():
                output_idx[idx]=1
            elif 'cond' in value.lower():
                output_idx[idx]=2
            elif 'timestep' in value.lower():
                output_idx[idx]=len(model.outputs)-4
            
    nonflat_pred_ldata=[model(BG_test[i][0]) for i in range(len(BG_test))]                      
    nonflat_obs_ldata=[[BG_test[i][1][j] for j in range(np.shape(BG_test[0][1])[0])]+[np.zeros_like(BG_test[i][1][0]) for k in range(max((len(model.outputs)-np.shape(BG_test[0][1])[0]),0))] for i in range(len(BG_test))]
    
   
    # Using the weighted volume to determine the average pressure, gas and condensate saturations.
    # Porosity is index index 4.
    nonflat_time=[BG_test[i][0][3] for i in range(len(BG_test))]
    nonflat_phi=[BG_test[i][0][4] for i in range(len(BG_test))]
    nonflat_permx=[BG_test[i][0][5] for i in range(len(BG_test))]

    # Unnormalize if outputs are normalized prior to training.   
    for i in range(len(nonflat_pred_ldata)):
        nonflat_pred_ldata[i],nonflat_obs_ldata[i],nonflat_time[i],nonflat_phi[i],nonflat_permx[i]=unnormalize_2D(model,nonflat_pred_ldata[i],nonflat_obs_ldata[i],inputs={'time':nonflat_time[i],'phi':nonflat_phi[i],'permx':nonflat_permx[i]},)                
    
    avg_pred_ldata=[]
    avg_obs_ldata=[]
    mae=[]

    for i in range(len(nonflat_pred_ldata)):
        avg_pred_ldata_per_key=[]
        avg_obs_ldata_per_key=[]
        mae_per_key=[]
        for j in output_idx:
            avg_pred_ldata_per_tstep=[]
            avg_obs_ldata_per_tstep=[]
            mae_per_tstep=[]
            for k in range(nonflat_pred_ldata[0][0].shape[0]):
                # Volumetric weighted average.
                avg_pred=np.sum(nonflat_pred_ldata[i][j][k]*nonflat_phi[i][k])/np.sum(nonflat_phi[i][k])
                avg_obs=np.sum(nonflat_obs_ldata[i][j][k]*nonflat_phi[i][k])/np.sum(nonflat_phi[i][k])
                avg_pred_ldata_per_tstep.append(avg_pred)
                avg_obs_ldata_per_tstep.append(avg_obs)
                mae_per_tstep.append(abs_error(avg_pred,avg_obs))
            avg_pred_ldata_per_key.append(avg_pred_ldata_per_tstep)  
            avg_obs_ldata_per_key.append(avg_obs_ldata_per_tstep)
            mae_per_key.append(mae_per_tstep)
        avg_pred_ldata.append(avg_pred_ldata_per_key)
        avg_obs_ldata.append(avg_obs_ldata_per_key)
        mae.append(mae_per_key)
        
    # Print output on screen
    for i in range(len(output_keys)):
        if i<len(output_keys)-1:
            print('Pred_Avg {}  == Obs_Avg {} ||'.format(output_keys[i],output_keys[i]),end=' ')
        else:
            print('Pred_Avg {}  == Obs_Avg {}'.format(output_keys[i],output_keys[i]))

    for i in range(len(avg_pred_ldata)):
        for j in range(0,len(avg_pred_ldata[i][0]),5):
            for k in range(len(output_keys)):
                if k<len(output_keys)-1:
                    print('{}  {} == '.format(avg_pred_ldata[i][k][j],avg_obs_ldata[i][k][j]),end=' ' )   
                else:
                    print('{}  {}'.format(avg_pred_ldata[i][k][j],avg_obs_ldata[i][k][j]),end='\n' )
    
    for i in range(len(output_keys)):
        print('MAE {}  == {} '.format(output_keys[i],tf.math.reduce_mean(mae,axis=[0,2])[i]),end='\n\n')

    nonflat_time=np.mean(nonflat_time,axis=(2,3,4))
    
    _BG_test=_BG_test_plot=list(range(len(BG_test)))
    if no_realizations is not None:
        _BG_test=list(range(0,len(BG_test),len(BG_test)//no_realizations)) 
        _BG_test_plot=list(range(0,len(BG_test),no_realizations)) 
        
    show_legend=[]
    for l in (_BG_test_plot):
        if _BG_test_plot.index(l)%legend_interval==0:
            show_legend.append(True)
        else:
            show_legend.append(False)
        
    # Split the realizations in blocks subplots of five to allow viewing to scale.
    rdiv=no_subplots
    rs=[_BG_test[i:i+rdiv] for i in range(0, len(_BG_test), rdiv)]

    # Plot the predicted and observed averages for all the metrics.
    sep_plots=True
    if group_plots:
        sep_plots=False

    if sep_plots:
        for r in rs:
            fig, ax = plt.subplots(np.shape(r)[0],len(output_keys),dpi=1200)
            if len(output_keys)==1 or np.shape(r)[0]==1:
                # Reshape the axis to a two dimensional array
                ax=np.reshape(ax,(np.shape(r)[0],len(output_keys)))
            fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.3)
            fig.suptitle('Average '+' '.join(output_keys).lower(),fontsize=9,y=1.01,weight='bold')
            
            for i in r:   # Realizations
                ri=r.index(i)
                for j in range(len(output_keys)):
                    ax[ri,j].plot(list(nonflat_time[i]),avg_pred_ldata[i][j], label='Average predicted '+output_keys[j], color='#0343DF', linestyle=output_linestyles[j],linewidth=1.0,zorder=100)      # Color azure
                    ax[ri,j].plot(list(nonflat_time[i]),avg_obs_ldata[i][j], label='Average observed '+output_keys[j], linestyle='', marker='s', markerfacecolor=output_colours[j],markeredgecolor='#A9561E',markersize=4) 
                    
                    if i==0:
                        ax[ri,j].set_title(f'Avg. {output_keys[j][0:5]}, Realiz.({i+1}), kavg={np.mean(nonflat_permx[i][...]):.2f} mD',fontsize=8.0, y=0.95)
                        if j==len(output_keys)-1:
                            fig.legend(loc="upper left",  prop={'size': xy_label_size},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.015))  
                    else:
                        ax[ri,j].set_title(f'Realiz. ({i+1}), kavg={np.mean(nonflat_permx[i][...]):.2f} mD',fontsize=8.0, y=0.95)
                    
                    if output_keys[j].lower()=='pressure': 
                        units='(psia)'
                    else: 
                        units=''           
                    
                    ax[ri,j].set_ylabel(f'{output_keys[j][0:3]} {units}',fontsize=xy_label_size)
                    if i==len(avg_pred_ldata)-1:
                        ax[ri,j].set_xlabel('Time (Days)',fontsize=xy_label_size)
                    
                    ax[ri,j].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)
               
                    y_obs_range=np.compress(np.array(avg_obs_ldata)[...,j,:].flatten()>0.,np.array(avg_obs_ldata)[...,j,:].flatten())
                    if output_keys[j].lower()[0:3]=='pre':
                        round_fac=100
                        n_ticks=5
                        if xy_limits['y'] is not None:
                            #ax[ri,j].set_yticks(np.round_(np.linspace(xy_limits['y'],model.cfd_type['Pi'],5),))
                            y_min=xy_limits['y']
                        else:
                            #ax[ri,j].set_yticks(np.round_(np.linspace(int(min(avg_obs_ldata[i][j])/np.min(model.cfd_type['Min_BHP']))*np.min(model.cfd_type['Min_BHP']),model.cfd_type['Pi'],5),))
                            y_min=np.floor((1.0*np.min([np.min(np.array(avg_obs_ldata)[...,j,:]),np.min(y_obs_range)]))/round_fac)*round_fac 
                        y_max=np.ceil((1.0*np.max([np.max(np.array(avg_obs_ldata)[...,j]),np.max(y_obs_range)]))/round_fac)*round_fac   
                        ax[ri,j].set_yticks(np.linspace(y_min,y_max,n_ticks))
                    if output_keys[j].lower()[0:3] in ['gsat','gas']:
                        round_fac=0.05
                        ax[ri,j].set_yticks(np.round_(np.linspace(0.5,1,7),2))
                    if j==2:
                        ax[ri,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
                    
                    ax[ri,j].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
                    ax[ri,j].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
                    
                    #plt.text(0.800, 0.95,f'Number of Perturbations: {no_perturbations}', size=7.0, ha='center', va='center', transform=axes1.transAxes, zorder=100)
            #from matplotlib.legend import _get_legend_handles_labels        
            plt.show()

    if group_plots:
        # Create the group subplots.
        fig1, ax1 = plt.subplots(1,len(output_keys),dpi=1200)   
        if len(output_keys)==1:#ax.ndim==1:
            # Reshape the axis to a two dimensional array.
            ax1=np.reshape(ax1,(1,len(output_keys)))
            fig1.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.4)
            fig1.suptitle(f'Average '+', '.join(output_keys).lower(),fontsize=9,y=1.02,weight='regular')   
        
        
        np.random.seed(5)
        for i in range(len(BG_test)):
            for j in range(len(output_keys)):
                j_idx=output_idx[j]
                # Remove any zero value from the observed data--this could occurs due to sparse sampling of the simulated data.
                y_obs=list(np.compress(np.array(avg_obs_ldata)[i,j].flatten()>0.,np.array(avg_obs_ldata)[i,j].flatten()))
                y_obs_range=np.compress(np.array(avg_obs_ldata)[:,j].flatten()>0.,np.array(avg_obs_ldata)[:,j].flatten())
                y_pred_range=np.compress(np.array(avg_pred_ldata)[:,j].flatten()>0.,np.array(avg_pred_ldata)[:,j].flatten())
               
     
                rnd_colour=np.random.rand(3)
                if i in _BG_test_plot and (f'{np.mean(nonflat_permx[i][...]):.3f}' in ['2.387','2.845','3.044','2.590',] or plot_all_series):  #['2.59',]
                    label_pred=f'predicted kavg={np.mean(nonflat_permx[i][...]):.3f} mD'
                    label_obs=f'observed kavg={np.mean(nonflat_permx[i][...]):.3f} mD'
                                       
                    
                    if not box_plots:
                        ax1[0,j].plot(list(nonflat_time[i]),avg_pred_ldata[i][j], label='Average '+label_pred, color=rnd_colour, linestyle='--',linewidth=1.5,alpha=1.0,zorder=2)      # Color azure
                        ax1[0,j].plot(list(nonflat_time[i]),[np.max(np.reshape(nonflat_pred_ldata[i][j_idx][k],-1)) for k in range(np.shape(nonflat_pred_ldata[i][j])[0])], label='Max '+label_pred, color=rnd_colour, linestyle='', marker='+', markerfacecolor='None',markeredgecolor=rnd_colour,markersize=4,markeredgewidth=1,alpha=1.0,zorder=2)      # Color azure                       
                        ax1[0,j].plot(list(nonflat_time[i]),[np.min(np.reshape(nonflat_pred_ldata[i][j_idx][k],-1)) for k in range(np.shape(nonflat_pred_ldata[i][j])[0])], label='Min '+label_pred, color=rnd_colour, linestyle='', marker='_', markerfacecolor='None',markeredgecolor=rnd_colour,markersize=4,markeredgewidth=1,alpha=1.0,zorder=2)      # Color azure                       
                    else:
                        ax1[0,j].boxplot([np.reshape(nonflat_pred_ldata[i][j_idx][k],-1) for k in range(np.shape(nonflat_pred_ldata[i][j])[0])], positions=list(nonflat_time[i]),showfliers=False,labels=list(nonflat_time[i]),widths=2.5)
                    if y_obs!=[]:
                        if not box_plots:
                            ax1[0,j].plot(list(nonflat_time[i]),avg_obs_ldata[i][j], label='Average '+label_obs, linestyle='', marker='o', markerfacecolor='None',markeredgecolor=rnd_colour,markersize=4,markeredgewidth=1,alpha=1.0,zorder=100) #CB9978 #A9561E
                        else:
                            ax1[0,j].boxplot([np.reshape(nonflat_obs_ldata[i][j_idx][k],-1) for k in range(np.shape(nonflat_obs_ldata[i][j])[0])], positions=list(nonflat_time[i]),showfliers=False,labels=list(nonflat_time[i]))
                            
                    if j==len(output_keys)-1 and ((label_obs is not None) or (label_pred is not None)):
                        fig1.legend(loc="upper left",  prop={'size': xy_label_size},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.02))  

                else:
                    if not box_plots:
                        label_pred=None; label_obs=None                        
                        ax1[0,j].plot(list(nonflat_time[i]),avg_pred_ldata[i][j], label=label_pred, color=rnd_colour, linestyle=output_linestyles[j],linewidth=0.5,alpha=0.0)      # Color azure
                        
                if i>0:
                     ax1[0,j].fill_between(list(nonflat_time[i]),list(avg_pred_ldata[i-1][j]),list(avg_pred_ldata[i][j]),color='#87CEFA',alpha=0.05)
                    
                if i==0:
                    ax1[0,j].set_title(f'Average {output_keys[j][0:5].lower()}, Realization ({j})',fontsize=xy_label_size, y=1.0)
                
                if output_keys[j].lower()=='pressure': 
                    units='(psia)'
                elif output_keys[j].lower()=='timestep': 
                    units='(days)'  
                else:
                    units=''
                
                ax1[0,j].set_ylabel(f'{output_keys[j][0:4]} {units}',fontsize=xy_label_size)               
                ax1[0,j].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)
                if output_keys[j].lower()[0:3]=='pre':
                    round_fac=100
                elif output_keys[j].lower()[0:3] in ['gsat','gas','timestep']:
                    round_fac=0.05
                else:
                    round_fac=0.025
                    #ax[ji,kk].set_yticks(np.round_(np.linspace(0.5,1,7),2))
    
                if len(y_pred_range)!=0:
                    # Get the min and max values for the plot. The min and max is scaled by a factor 0f 0.9 and 1.10 respectively, then rounded to the nearest 50
                    if xy_limits['y'] is not None:
                        y_min=xy_limits['y']
                    else:
                        y_min=np.floor((1.0*np.min([np.min(np.array(avg_pred_ldata)[:,j]),np.min(y_pred_range)]))/round_fac)*round_fac
                    #y_max=np.ceil((1.0*np.max([np.max(np.array(avg_pred_ldata)[:,j_idx]),np.max(y_pred_range)]))/round_fac)*round_fac
                    y_max=np.ceil((1.0*np.max([np.max([np.reshape(nonflat_pred_ldata[i][j_idx][k],-1) for k in range(np.shape(nonflat_pred_ldata[i][j])[0])]),np.max(y_pred_range)]))/round_fac)*round_fac
                    
                    n_ticks=5
                    
                    ax1[0,j].set_ylim(y_min,y_max)
                    ax1[0,j].set_xlim(0,np.max(nonflat_time[i]))
                    #ax1[0,j].set_yscale('log')
                
                ax1[0,j].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                if j==0:
                    ax1[0,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                elif j==1:
                    ax1[0,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
                else:
                    ax1[0,j].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
                
                ax1[0,j].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))                  
        plt.rcParams["font.family"] =["Times New Roman"]
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.show()
    return avg_pred_ldata,avg_obs_ldata,mae

# A function that returns an image plot of the different quantities of interest (e.g., pressure, gas saturation and condensate saturation) at certain time point indexes.
# This is used with a 2D model output (e.g., convolutional-based encoder-decoder network).
def img_plot_2D(model=None, BG_test=None, tstep_idx=[0,20,40,60],bound_int=50,plot_title=['Pressure','Gas Saturation','Condensate Saturation'],cmap=[plt.cm.Reds,plt.cm.Greens,plt.cm.Oranges,plt.cm.Blues],no_realizations=6,max_error=5.,\
                timestep=[0,10],perm=[],xy_label_size=7.,derived_osat=True,hist_bin=20,dump_excel=True,folder_path=None,return_only_relative_error=False):
    # Check that tstep_idx exists
    for val in tstep_idx:
        if int(val) > 100:
            tstep_idx.remove(val)
    
    tstep_idx=[int(tf.math.ceil(tstep_idx[i]*(len(BG_test[0][0][0]))/100.))for i in range(len(tstep_idx[:len(BG_test[0][0][0])]))]

    x_mgrid=[BG_test[i][0][0] for i in range(len(BG_test))]                      
    y_mgrid=[BG_test[i][0][1] for i in range(len(BG_test))] 
    extent_ =[[x_mgrid[i].min(), x_mgrid[i].max(), y_mgrid[i].min(), y_mgrid[i].max()] for i in range(len(BG_test))]                 

    nonflat_pred_ldata=np.array([model(BG_test[i][0])[0:len(plot_title)] for i in range(len(BG_test))])                     # Predicts for the entire prediction range
    nonflat_obs_ldata=np.array([[BG_test[i][1][j] for j in range(len(plot_title))] for i in range(len(BG_test))])

    if derived_osat and len(plot_title)>1:
        nonflat_obs_ldata[:,2,...]=1-model.cfd_type['SCAL']['End_Points']['Swmin']-nonflat_obs_ldata[:,1,...]
        
    res_mgrid_mae=(tf.math.abs((nonflat_obs_ldata-nonflat_pred_ldata))/nonflat_obs_ldata)*100

    res_min=[tf.constant(0.,dtype=res_mgrid_mae[i].dtype) for i in range(len(plot_title))]
    res_max=[tf.constant(max_error,dtype=res_mgrid_mae[i].dtype) for i in range(len(plot_title))]

    nonflat_time=[BG_test[i][0][3] for i in range(len(BG_test))]
    nonflat_phi=[BG_test[i][0][4] for i in range(len(BG_test))]
    nonflat_permx=[BG_test[i][0][5] for i in range(len(BG_test))]

    # Un normalize if outputs are normalized prior to training.
    for i in range(len(nonflat_pred_ldata)):
        nonflat_pred_ldata[i],nonflat_obs_ldata[i],nonflat_time[i],_,nonflat_permx[i]=unnormalize_2D(model,nonflat_pred_ldata[i],nonflat_obs_ldata[i],inputs={'time':nonflat_time[i],'phi':nonflat_phi[i],'permx':nonflat_permx[i]},output_keys=plot_title)                
    
    z_min = [None for l in range(len(plot_title))]
    z_max = [None for l in range(len(plot_title))]
    
    if dump_excel and folder_path is not None:
        excel_file_path=f'{folder_path}/grid_{(np.mean(nonflat_permx)):.2f}mD_{len(BG_test)}test_realiz.xlsx'
        excel_data={}

    # Reshape the predicted data.
    # cmap = mpl.colors.ListedColormap(['yellow','orange','red']).
    # Create the subplots.
    if not return_only_relative_error:
        for i in range(len(BG_test)):  #Indicates the Realizations.
    
            # z=+/-5% observed value
            # z_min = [np.min(nonflat_obs_ldata[i][l][0:np.max(tstep_idx),...]) for l in range(len(plot_title))]
            # z_max = [np.max(nonflat_obs_ldata[i][l][0:np.max(tstep_idx),...]) for l in range(len(plot_title))]
    
            for k in range(len(plot_title)):
                height_ratios=[1]*len(tstep_idx); width_ratios=[1]*3
                fig, ax = plt.subplots(len(tstep_idx),3,figsize=(5,6),dpi=1200)                  #gridspec_kw={'height_ratios':height_ratios,'width_ratios':width_ratios,'wspace':0.4}
                fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.2, wspace=0.4)
                fig.suptitle(f"Predicted and observed {plot_title[k].lower()}; residual Maps - realization ({i+1}), kavg={np.mean(nonflat_permx[i][...]):.2f} mD",fontsize=xy_label_size,y=1.0,weight='regular')
    
                for j in range(len(tstep_idx)):
                    
                    if np.mean(nonflat_obs_ldata)!=0:
                        z_min = [1.00*np.min(nonflat_obs_ldata[i][l][tstep_idx[j],...]) for l in range(len(plot_title))]    
                        z_max = [1.00*np.max(nonflat_obs_ldata[i][l][tstep_idx[j],...]) for l in range(len(plot_title))]
                                    
                    img1=ax[j,0].imshow(nonflat_pred_ldata[i][k][tstep_idx[j],...],cmap =cmap[k], vmin = z_min[k], vmax = z_max[k],interpolation='nearest',origin='lower')
                    img2=ax[j,1].imshow(nonflat_obs_ldata[i][k][tstep_idx[j],...],cmap =cmap[k], vmin = z_min[k], vmax = z_max[k],interpolation='nearest',origin='lower')
                    img3=ax[j,2].imshow(res_mgrid_mae[i][k][tstep_idx[j],...],cmap =cmap[-1], vmin = res_min[k], vmax = res_max[k],interpolation='nearest',origin='lower')
    
                    avg_res_err=np.mean(res_mgrid_mae[i][k][tstep_idx[j],...])
                    if j==0:
                        ax[j,0].set_title(f'Pred. {plot_title[k][0:5].lower()} time={np.mean(nonflat_time[i][tstep_idx[j],...]):.1f} D',fontsize=7,y=0.95)
                        ax[j,1].set_title(f'Obs. {plot_title[k][0:5].lower()} time={np.mean(nonflat_time[i][tstep_idx[j],...]):1f} D',fontsize=7,y=0.95)
                        ax[j,2].set_title(f'%Res. {plot_title[k][0:5].lower()} time={np.mean(nonflat_time[i][tstep_idx[j],...]):.1f} D [{avg_res_err:.2f}%]',fontsize=7,y=0.95)
                    else:
                        ax[j,0].set_title(f'Time={np.mean(nonflat_time[i][tstep_idx[j],...]):.1f} D',fontsize=7,y=0.95)
                        ax[j,1].set_title(f'Time={np.mean(nonflat_time[i][tstep_idx[j],...]):.1f} D',fontsize=7,y=0.95)
                        ax[j,2].set_title(f'Time={np.mean(nonflat_time[i][tstep_idx[j],...]):.1f} D [{avg_res_err:.2f}%]',fontsize=7,y=0.95)
                    
                    ax[j,0].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                    ax[j,1].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                    ax[j,2].tick_params(axis='both', which='major', length=2, width=1,labelsize=2)
                    
                    divider=[make_axes_locatable(ax[j,l]) for l in range(3)]
                    cax = [divider[l].append_axes("right", size="8%", pad=0.05) for l in range(3)]
                    
                    cax[0].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
                    cax[1].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
                    cax[2].tick_params(axis='both', which='major', length=2, width=1,labelsize=7)
        
                    if k==1:
                        cbar_fmt=mpl.ticker.FormatStrFormatter('%.3f')
                    else:
                        cbar_fmt=None
                        
                    fig.colorbar(img1,cax=cax[0],format=cbar_fmt)
                    fig.colorbar(img2,cax=cax[1],format=cbar_fmt)
                    fig.colorbar(img3,cax=cax[2],format=cbar_fmt)
                
                plt.show() 
                
    # Residual error histogram.
    no_realiz=np.shape(res_mgrid_mae)[0]
    no_tsteps=np.shape(res_mgrid_mae)[2]
    fig, ax = plt.subplots(1,2,figsize=(10,5),dpi=1200)                  
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.4, wspace=0.2)
    fig.suptitle(f'Average relative L1 and L2 errors ({plot_title[0].lower()}) for {no_realiz} realizations, {no_tsteps} timesteps',fontsize=xy_label_size,y=1.01,weight='bold')
    
    L1_error=(tf.reduce_sum(tf.math.abs(nonflat_obs_ldata-nonflat_pred_ldata),axis=[3,4,5],keepdims=True)/tf.reduce_sum((nonflat_obs_ldata),axis=[3,4,5],keepdims=True))
    L2_error=(tf.reduce_sum(tf.math.square(nonflat_obs_ldata-nonflat_pred_ldata),axis=[3,4,5],keepdims=True)/tf.reduce_sum(tf.math.square(nonflat_obs_ldata),axis=[3,4,5],keepdims=True))
    rel_error=[np.reshape(tf.reduce_mean(L1_error,axis=[3,4,5]),-1),np.reshape(tf.reduce_mean(L2_error,axis=[3,4,5]),-1)]
    fcolour=['b','g']
    error_type=['L1-','L2']
    
    for hi in range(2):
        img1=ax[hi].hist(rel_error[hi],hist_bin, facecolor=fcolour[hi])
        ax[hi].set_xlabel(f'relative {error_type[hi]} error %', fontsize=10)
        ax[hi].set_ylabel('Frequency', fontsize=10)
        ax[hi].tick_params(axis='both', which='major', length=2, width=1,labelsize=8)
        ax[hi].tick_params(axis='both', which='major', length=2, width=1,labelsize=8)
        if dump_excel and folder_path is not None:
            excel_header_name=f'relative_{error_type[hi]}_error_%_avg'
            excel_data[excel_header_name]=list(rel_error[hi])
    
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.show()
    if dump_excel and folder_path is not None:
        df=pd.DataFrame(excel_data).transpose()
        df.to_excel(excel_writer=excel_file_path,sheet_name='Sheet_'+str(datetime.datetime.now())[:20].replace(':','_'))
    return

# A function that returns plots of the wells' grid block vs. time for the different quantities of interest (e.g., pressure, gas saturation and condensate saturation).
# This function also prints these values on the screen at selected time point indexes.  
# This function is used with a 2D model output (e.g., convolutional-based encoder-decoder network).
def predict_plot_wells_2D(model=None, BG_test=None, tstep=10,tstep_idx=[0,20,40,60,80],bound_int=50,wells_idx=[(14,14,0)],plot_title=['Pressure','Gas Saturation','Condensate Saturation'],\
                          output_colours=['lightcoral','limegreen','orange'],output_linestyles=['dashed','dotted','dashdot'],no_realizations=None,perm=[],xy_label_size=7.5,no_subplots=3,\
                              xy_limits={'y':None},legend_interval=5,dump_excel=True,rate_bhp=True,folder_path=None,plot_all_series=False):
    start_time=time.time()
    # Check that the selected time point indexes (tstep_idx) exists.
    for val in tstep_idx:
        if int(val) > 100:
            tstep_idx.remove(val)
    
    def flat_list(BG_test,obs_idx=[13]):
        nonflat_pred_ldata=[]
        nonflat_obs_ldata=[]
        for i in range(len(BG_test)):
            pred_out_flat=[]; obs_out_flat=[]
            pred_out=model(BG_test[i][0])
            for j_idx,j in enumerate(pred_out):
                if (len(np.shape(j))!=5): #or (j_idx not in obs_idx):
                    pred_out_flat.append(j[:,...])
                    if j_idx>1:
                        obs_out_flat.append(np.zeros_like(BG_test[i][1][0]))
                    else:
                        obs_out_flat.append(BG_test[i][1][j_idx])                        
                else:
                    # outer dimension of the array can be iterated 
                    for k in j:
                        pred_out_flat.append(k[:,...])
                        if j_idx>1:
                            obs_out_flat.append(np.zeros_like(BG_test[i][1][0]))
                        else:
                            obs_out_flat.append(BG_test[i][1][j_idx])     
            nonflat_pred_ldata.append(np.stack(pred_out_flat))
            nonflat_obs_ldata.append(obs_out_flat)
        return np.array(nonflat_pred_ldata),np.array(nonflat_obs_ldata)
    nonflat_pred_ldata,nonflat_obs_ldata=flat_list(BG_test,obs_idx=[13])
    
    output_idx=list(range(len(plot_title)))
    # Update the output_idx.
    def set_output_idx(output_idx=None,plot_title=None,flat_len=0):
        output_idx_n=output_idx
        for idx, value in enumerate(plot_title):
            if 'pre' in value.lower():
                output_idx_n[idx]=0
            elif 'gas' in value.lower():
                output_idx_n[idx]=1
            elif 'cond' in value.lower():
                output_idx_n[idx]=2
            elif 'timestep' in value.lower():
                output_idx_n[idx]=flat_len-4
            elif 'qfgrate' in value.lower():
                if flat_len>=15:
                    output_idx_n[idx]=flat_len-5
                else:
                    output_idx_n[idx]=flat_len-2
            elif 'qdgrate' in value.lower():
                if flat_len>=15:
                    output_idx_n[idx]=flat_len-4
                else:
                    output_idx_n[idx]=flat_len-2
            elif 'qforate' in value.lower():
                if flat_len>=15:
                    output_idx_n[idx]=flat_len-3
                else:
                    output_idx_n[idx]=flat_len-2
            elif 'qvorate' in value.lower():
                if flat_len>=15:
                    output_idx_n[idx]=flat_len-2
                else:
                    output_idx_n[idx]=flat_len-2
            elif 'bhp' in value.lower():
                output_idx_n[idx]=flat_len-1
            else:
                output_idx_n[idx]=flat_len-2
        return output_idx_n
    output_idx=set_output_idx(output_idx,plot_title,flat_len=len(nonflat_pred_ldata[0]))
    
    tstep_idx=[int(tstep_idx[i]*(len(BG_test[0][0][0]))/100.) for i in range(len(tstep_idx))]
    x_mgrid=[BG_test[i][0][0] for i in range(len(BG_test))]                      
    y_mgrid=[BG_test[i][0][1] for i in range(len(BG_test))] 
    extent_ =[[x_mgrid[i].min(), x_mgrid[i].max(), y_mgrid[i].min(), y_mgrid[i].max()] for i in range(len(BG_test))]                 

    #nonflat_pred_ldata=np.array([model(BG_test[i][0])[0:len(plot_title)] for i in range(len(BG_test))])                     # Predicts for the entire prediction range
    #nonflat_obs_ldata=np.array([[BG_test[i][1][j] for j in range(len(plot_title))] for i in range(len(BG_test))])
    
    nonflat_time=[BG_test[i][0][3] for i in range(len(BG_test))]
    nonflat_phi=[BG_test[i][0][4] for i in range(len(BG_test))]
    nonflat_permx=[BG_test[i][0][5] for i in range(len(BG_test))]

    # Un normalize if outputs are normalized prior to training.
    for i in range(len(nonflat_pred_ldata)):
        nonflat_pred_ldata[i],nonflat_obs_ldata[i],nonflat_time[i],nonflat_phi[i],nonflat_permx[i]=unnormalize_2D(model,nonflat_pred_ldata[i],nonflat_obs_ldata[i],inputs={'time':nonflat_time[i],'phi':nonflat_phi[i],'permx':nonflat_permx[i]},output_keys=plot_title)                
    
    well_pred_ldata=[]
    well_obs_ldata=[]
    mae=[]

    def abs_error(x_pred,x_obs):
        return tf.math.abs((x_pred-x_obs)/x_obs)

    # Create the excel dump file
    if dump_excel and folder_path is not None:
        excel_file_path=f'{folder_path}/avgkx_{(np.mean(nonflat_permx)):.2f}mD_{len(BG_test)}test_realiz.xlsx'
        excel_data={}

    for i in range(len(nonflat_pred_ldata)):        # Perturbation
        well_pred_ldata_per_key=[]
        well_obs_ldata_per_key=[]
        mae_per_key=[]
        for j in wells_idx:                        # Wells Connection Index  
            well_pred_ldata_per_wellidx=[]
            well_obs_ldata_per_wellidx=[]
            mae_per_wellidx=[]
            for k in range(nonflat_pred_ldata[0][0].shape[0]): # Timestep
                well_pred_ldata_per_tstep=[]
                well_obs_ldata_per_tstep=[]
                mae_per_tstep=[]
                for kk in output_idx:    # Plot Title
                    # Volumetric weighted average
                    well_pred=nonflat_pred_ldata[i][kk][k,j[0],j[1],j[2]]
                    well_obs=nonflat_obs_ldata[i][kk][k,j[0],j[1],j[2]]
                    well_pred_ldata_per_tstep.append(well_pred)
                    well_obs_ldata_per_tstep.append(well_obs)
                    mae_per_tstep.append(abs_error(well_pred,well_obs))
                well_pred_ldata_per_wellidx.append(well_pred_ldata_per_tstep)  
                well_obs_ldata_per_wellidx.append(well_obs_ldata_per_tstep)
                mae_per_wellidx.append(mae_per_tstep)
            well_pred_ldata_per_key.append(well_pred_ldata_per_wellidx)  
            well_obs_ldata_per_key.append(well_obs_ldata_per_wellidx)
            mae_per_key.append(mae_per_wellidx)
        well_pred_ldata.append(well_pred_ldata_per_key)
        well_obs_ldata.append(well_obs_ldata_per_key)
        mae.append(mae_per_key)

    # Print output on screen.
    for i in range(len(nonflat_pred_ldata)):  #Perturbation
        print('Realization_*',i+1)
        for j in range(len(wells_idx)):
            for k in tstep_idx:
                for kk in range(len(plot_title)):
                    if tstep_idx.index(k)==0:
                        if kk<len(plot_title)-1:
                            print('Pred_Well_{} {}  == Obs_Avg_{} {} ||'.format(str(wells_idx[j]),plot_title[kk],str(wells_idx[j]),plot_title[kk]),end=' ')
                        else:
                            print('Pred_Well_{} {}  == Obs_Avg_{} {}'.format(str(wells_idx[j]),plot_title[kk],str(wells_idx[j]),plot_title[kk]))

                    if kk<len(plot_title)-1:
                        print('{}  {} == '.format(well_pred_ldata[i][j][k][kk],well_obs_ldata[i][j][k][kk]),end=' ')
                    else:
                        print('{} {}'.format(well_pred_ldata[i][j][k][kk],well_obs_ldata[i][j][k][kk]))
            print('\n')
        
            # Print the average at each realization. 
            print('MAE for Realization_*',i+1)
            for jk in range(len(plot_title)):
                print('MAE {}  == {} '.format(plot_title[jk],tf.math.reduce_mean(mae[i],axis=[1])[j][jk]))       
        print('\n')
        
    # Split the total batch based on the reported no_of_realizations. 
    _BG_test=_BG_test_plot=list(range(len(BG_test)))
    if no_realizations is not None:
        _BG_test_plot=list(range(0,len(BG_test),no_realizations)) 
  
    # List of plot legends to show.
    show_legend=[]
    for l in (_BG_test_plot):
        if _BG_test_plot.index(l)%legend_interval==0:
            show_legend.append(True)
        else:
            show_legend.append(False)
            
    # Split the realizations in blocks subplots of five to allow viewing to scale.
    rdiv=no_subplots
    rem_rs=rem_js=[]
    js=[wells_idx[i:i+rdiv] for i in range(0, len(wells_idx), rdiv)]
    js1=[wells_idx[i:i+1] for i in range(0, len(wells_idx), 1)]
    rs=[_BG_test[i:i+rdiv] for i in range(0, len(_BG_test), rdiv)]

    # reduce the time axis.
    nonflat_time=np.mean(nonflat_time,axis=(2,3,4))    

    # Create plot figure
    fig1=plt.figure('Group_Plot')
    fig2=plt.figure('Individual_Plot')

    for jb in js1:
        # Create the wells subplots.
        fig1, ax1 = plt.subplots(np.shape(jb)[0],len(plot_title),dpi=1200)   
        if len(plot_title)==1 or np.shape(jb)[0]==1:
            # Reshape the axis to a two dimensional array.
            ax1=np.reshape(ax1,(np.shape(jb)[0],len(plot_title)))
            fig1.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.4)
            fig1.suptitle(f'Wells grid block '+', '.join(plot_title).lower(),fontsize=9,y=1.02,weight='regular')   
            
        for j in jb:
            np.random.seed(5)
            ji=jb.index(j)
            for i in _BG_test:
                for kk in range(len(plot_title)):
                    # Remove any zero value from the observed data--this could occurs due to sparse sampling of the simulated data
                    x_obs=list(np.compress(nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]>0.,nonflat_time[i]))
                    y_obs=list(np.compress(nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]>0.,nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]))
  
                    rnd_colour=np.random.rand(3)
                    if i in _BG_test_plot and (f'{np.mean(nonflat_permx[i][...]):.3f}' in ['2.387','3.044'] or plot_all_series):  #['2.59',]
                        label_pred=f'Predicted kavg={np.mean(nonflat_permx[i][...]):.3f} mD'
                        label_obs=f'Observed kavg={np.mean(nonflat_permx[i][...]):.3f} mD'
                        ax1[ji,kk].plot(x_obs,y_obs, label=label_obs, linestyle='', marker='o', markerfacecolor='None',markeredgecolor=rnd_colour,markersize=4,markeredgewidth=1,alpha=1.0,zorder=100) #CB9978 #A9561E
                        ax1[ji,kk].plot(list(nonflat_time[i]),list(nonflat_pred_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]), label=label_pred, color=rnd_colour, linestyle='solid',linewidth=1.25,alpha=1.0,zorder=2)      # Color azure
                    else:
                        label_pred=None; label_obs=None                        
                        ax1[ji,kk].plot(list(nonflat_time[i]),list(nonflat_pred_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]), label=label_pred, color=rnd_colour, linestyle=output_linestyles[kk],linewidth=0.5,alpha=0.1)      # Color azure

                    if _BG_test.index(i)>0:
                        ax1[ji,kk].fill_between(list(nonflat_time[i]),list(nonflat_pred_ldata[i-1][output_idx[kk]][:,j[0],j[1],j[2]]),list(nonflat_pred_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]),color='#87CEFA',alpha=0.1)
                        
                    if ji==0:
                        ax1[ji,kk].set_title(f'Well grid block {plot_title[kk][0:5].lower()}, Well ({j})',fontsize=xy_label_size, y=1.0)
                        if kk==len(plot_title)-1 and ((label_obs is not None) or (label_pred is not None)):
                            fig1.legend(loc="upper left",  prop={'size': xy_label_size},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.02))  
                    else:
                        ax1[ji,kk].set_title(f'Well ({j})',fontsize=xy_label_size,y=1.0)
                    
                    if plot_title[kk].lower()=='pressure': 
                        units='(psia)'
                    else: 
                        units=''           
                    
                    ax1[ji,kk].set_ylabel(f'{plot_title[kk][0:3]} {units}',fontsize=xy_label_size)
                    if j==jb[-1]:
                        ax1[ji,kk].set_xlabel('Time (Days)',fontsize=xy_label_size)
                    
                    ax1[ji,kk].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)

                    if plot_title[kk].lower()[0:3]=='pre':
                        round_fac=100
                    elif plot_title[kk].lower()[0:3] in ['gsat','gas']:
                        round_fac=0.05
                    else:
                        round_fac=0.025
                        #ax[ji,kk].set_yticks(np.round_(np.linspace(0.5,1,7),2))
                    
                    y_obs_range=np.compress(nonflat_obs_ldata[:,output_idx[kk],:,j[0],j[1],j[2]].flatten()>0.,nonflat_obs_ldata[:,output_idx[kk],:,j[0],j[1],j[2]].flatten())

                    if len(y_obs_range)!=0:
                        # Get the min and max values for the plot. The min and max is scaled by a factor 0f 0.9 and 1.10 respectively, then rounded to the nearest 50
                        if xy_limits['y'] is not None:
                            y_min=xy_limits['y']
                        else:
                            y_min=np.floor((1.0*np.min([np.min(nonflat_pred_ldata[:,output_idx[kk],:,j[0],j[1],j[2]]),np.min(y_obs_range)]))/round_fac)*round_fac
                        y_max=np.ceil((1.0*np.max([np.max(nonflat_pred_ldata[:,output_idx[kk],:,j[0],j[1],j[2]]),np.max(y_obs_range)]))/round_fac)*round_fac
                        n_ticks=5
                    
                        ax1[ji,kk].set_yticks(np.linspace(y_min,y_max,n_ticks))
                    
                    ax1[ji,kk].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                    if kk==0:
                        ax1[ji,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                    elif kk==1:
                        ax1[ji,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
                    else:
                        ax1[ji,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
                    
                    ax1[ji,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
                    ax1[ji,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
        plt.rcParams["font.family"] =["Times New Roman"]
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.show()
   
    
    for j in wells_idx:
        for r in rs:
            # Create the realizations subplots.
            fig2, ax2 = plt.subplots(np.shape(r)[0],len(plot_title),dpi=1200)
            if len(plot_title)==1 or np.shape(r)[0]==1: 
                # Reshape the axis to a two dimensional array.
                ax2=np.reshape(ax2,(np.shape(r)[0],len(plot_title)))
            fig2.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.4)
            fig2.suptitle(f'Wells grid block '+', '.join(plot_title).lower()+' '+str(j),fontsize=9,y=1.02,weight='regular')    

            for i in r:
                ri=r.index(i)
                for kk in range(len(plot_title)):
                    # Remove any zero value from the observed data--this could occurs due to sparse sampling of the simulated data
                    x_obs=list(np.compress(nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]>0.,nonflat_time[i]))
                    y_obs=list(np.compress(nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]>0.,nonflat_obs_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]))

                    ax2[ri,kk].plot(list(nonflat_time[i]),list(nonflat_pred_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]]), label='Predicted Well '+plot_title[kk], color='#0343DF', linestyle=output_linestyles[kk],linewidth=1.5,zorder=100)      # Color azure
                    ax2[ri,kk].plot(x_obs,y_obs, label='Observed Well '+plot_title[kk], linestyle='', marker='s', markerfacecolor=output_colours[kk],markeredgecolor='#A9561E',markersize=4,markeredgewidth=0.75) #CB9978 #A9561E
                    if i==0:
                        ax2[ri,kk].set_title(f'Well grid block {plot_title[kk][0:5].lower()}, Realiz.({i}), kavg={np.mean(nonflat_permx[i][...]):.2f} mD',fontsize=xy_label_size, y=0.95)
                        if kk==len(plot_title)-1:
                            fig2.legend(loc="upper left",  prop={'size': xy_label_size},ncol=3,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.02))  
                    else:
                        ax2[ri,kk].set_title(f'Realiz. ({i}), kavg={np.mean(nonflat_permx[i][...]):.2f} mD',fontsize=xy_label_size, y=0.95)
                    
                    if plot_title[kk].lower()=='pressure': 
                        units='(psia)'
                    else: 
                        units=''           
                    
                    ax2[ri,kk].set_ylabel(f'{plot_title[kk][0:3]} {units}',fontsize=xy_label_size)
                    if i==r[-1]:
                        ax2[ri,kk].set_xlabel('Time (Days)',fontsize=xy_label_size)
                    
                    ax2[ri,kk].grid(True,which='major', axis='both', linestyle="--", color='grey',linewidth=0.5)
                    #ax[ri,kk].set_xticks(np.linspace(0,nonflat_obs_ldata[i][kk].shape[0],6))
                    if plot_title[kk].lower()[0:3]=='pre':
                        round_fac=100
                    elif plot_title[kk].lower()[0:3] in ['gsat','gas']:
                        round_fac=0.05
                    else:
                        round_fac=0.025
                    
                    y_obs_range=np.compress(nonflat_obs_ldata[:,output_idx[kk],:,j[0],j[1],j[2]].flatten()>0.,nonflat_obs_ldata[:,output_idx[kk],:,j[0],j[1],j[2]].flatten())

                    if len(y_obs_range)!=0:
                        # Get the min and max values for the plot. The min and max is scaled by a factor 0f 0.9 and 1.10 respectively, then rounded to the nearest 50.
                        if xy_limits['y'] is not None:
                            y_min=xy_limits['y']
                        else:
                            y_min=np.floor((1.0*np.min([np.min(nonflat_pred_ldata[:,output_idx[kk],:,j[0],j[1],j[2]]),np.min(y_obs_range)]))/round_fac)*round_fac
                        y_max=np.ceil((1.0*np.max([np.max(nonflat_pred_ldata[:,output_idx[kk],:,j[0],j[1],j[2]]),np.max(y_obs_range)]))/round_fac)*round_fac
                        n_ticks=5                    
                        ax2[ri,kk].set_yticks(np.linspace(y_min,y_max,n_ticks))
                    
                    ax2[ri,kk].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                    if kk==0:
                        ax2[ri,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
                    elif kk==1:
                        ax2[ri,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
                    else:
                        ax2[ri,kk].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
                    
                    ax2[ri,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
                    ax2[ri,kk].tick_params(axis='both', which='major', length=2, width=1,labelsize=(0.8*xy_label_size))
                   
                    # plt.text(0.800, 0.95,f'Number of Perturbations: {no_perturbations}', size=7.0, ha='center', va='center', transform=axes1.transAxes, zorder=100)
                    # from matplotlib.legend import _get_legend_handles_labels
                    if dump_excel and folder_path is not None:
                        if 'Time' not in excel_data.keys():
                            excel_data['Time']=list(nonflat_time[i])   #Use the zero index
                        excel_header_name=f'{plot_title[kk][0:3]}_{np.mean(nonflat_permx[i][...]):.2f}mD_{str(j).replace(",","")}'
                        excel_data[excel_header_name]=list(nonflat_pred_ldata[i][output_idx[kk]][:,j[0],j[1],j[2]])

            #fig=plt.gcf()
            #size = fig.get_size_inches()
            #[ax[len(plot_range)-1,k].legend(loc="lower left",  prop={'size': 8},ncol=1,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -2.25)) for k in range(len(output_keys))] 
            plt.rcParams["font.family"] = ["Times New Roman"]
            plt.rcParams["font.serif"] = ["Times New Roman"]
            plt.show()
    end_time=time.time()
    print((end_time-start_time))
    if dump_excel and folder_path is not None:
        df=pd.DataFrame(excel_data).transpose()
        df.to_excel(excel_writer=excel_file_path,sheet_name='Sheet_'+str(datetime.datetime.now()))
        if rate_bhp:
            df_rb=pd.DataFrame(excel_data_rate_bhp).transpose()
            df_rb.to_excel(excel_writer=excel_file_path,sheet_name='Sheet_'+str(datetime.datetime.now()))
    
    return well_pred_ldata,well_obs_ldata,mae
# =================================================== Plot Auxiliary Functions ==================================================
# These function are used prior to generating the plots or just after the training. 
from pickle import dump
from pickle import load
import re

# A function that returns a model predictions, and their normalized values. 
def predict_range_score(model=None, obs_fdata=None,obs_ldata=None):
    # Both observed features and labels are lists of input and output parameters respectively
    if model.cfd_type['Data_Arr']==2:
        output_keys=['pressure','sbu_gas','sbu_oil']
    else:
        output_keys=['pressure','sgas','soil']
    
    if model.cfd_type['Optimizer']=='first-order':
        # Prediction with the model using the test data
        score=model.evaluate(obs_fdata,obs_ldata,verbose=1)
        print(score)
    
    mae={key:[] for key in output_keys}
    def abs_error(x_pred,x_obs):
        return tf.math.abs((x_pred-x_obs)/x_obs)
    #Evaluaion with the standalone Keras accuracy class
    pred_ldata=model.predict(obs_fdata)  #use type() tc check datatype--it's a list of ndarrays 
    
    print('Pred. {}  Observed {}  ==  Predicted {}  Observed {}  ==  Predicted {}  Observed {}\n'\
          .format(output_keys[0],output_keys[0],output_keys[1],output_keys[1],output_keys[2],output_keys[2]))
    for i in range(0,len(pred_ldata[0]),int(len(pred_ldata[0])/30)):
        for j in range(3):
            mae[output_keys[j]].append(abs_error(pred_ldata[j][i],obs_ldata[j][i]))
        print(pred_ldata[0][i], ' ', obs_ldata[0][i],'==', pred_ldata[1][i], ' ', obs_ldata[1][i],'==',pred_ldata[2][i], ' ', obs_ldata[2][i] )
       
    nonorm_obs_ldata=[]
    nonorm_pred_ldata=[]
    if model.cfd_type['Output_Normalization']!=None:
        a=model.cfd_type['Norm_Limits'][0]
        b=model.cfd_type['Norm_Limits'][1]
        ts=pd.DataFrame(model.ts.numpy(),index=model.ts_idx_keys[0],columns=model.ts_idx_keys[1])
        mae={key:[] for key in output_keys}
        for key in output_keys:
            if model.cfd_type['Output_Normalization']=='linear-scaling':
                nonorm_obs=((obs_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
                nonorm_pred=((pred_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
            else:
                nonorm_obs=obs_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
                nonorm_pred=pred_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
            nonorm_obs_ldata.append(nonorm_obs)
            nonorm_pred_ldata.append(nonorm_pred)
        
        # Output un normalized values to console
        print('\n**********NON NORMALIZED VALUES**********\n\
              Pred. {}  Observed {}  ==  Predicted {}  Observed {}  ==  Predicted {}  Observed {}\n'\
                  .format(output_keys[0],output_keys[0],output_keys[1],output_keys[1],output_keys[2],output_keys[2]))
        for i in range(0,len(nonorm_pred_ldata[0]),int(len(nonorm_pred_ldata[0])/20)):
            for j in range(3):
                mae[output_keys[j]].append(abs_error(nonorm_pred_ldata[j][i],nonorm_obs_ldata[j][i]))
            print(nonorm_pred_ldata[0][i], ' ', nonorm_obs_ldata[0][i],'==', nonorm_pred_ldata[1][i], ' ', nonorm_obs_ldata[1][i],'==', nonorm_pred_ldata[2][i], ' ', nonorm_obs_ldata[2][i])
    
    print(f'MAE {output_keys[0]}: {tf.math.reduce_mean(mae[output_keys[0]])}\nMAE {output_keys[1]}: {tf.math.reduce_mean(mae[output_keys[1]])}\nMAE {output_keys[2]}: {tf.math.reduce_mean(mae[output_keys[2]])}')
    return pred_ldata, nonorm_obs_ldata, nonorm_pred_ldata

# A function that unnormalizes a given dataset (features and labels), and returns the unnormalized outputs.
def unnormalize_2D(model=None,pred_ldata=None,obs_ldata=None,inputs={'time':0.,'phi':0.,'permx':0.},output_keys=['pressure','gsat','osat']):
    a=model.cfd_type['Norm_Limits'][0]
    b=model.cfd_type['Norm_Limits'][1]
    def_output_keys=['pressure','gsat','osat',]
    ts=pd.DataFrame(model.ts.numpy(),index=model.ts_idx_keys[0],columns=model.ts_idx_keys[1])
    if model.cfd_type['Input_Normalization']!=None:
        if model.cfd_type['Input_Normalization']=='linear-scaling':
            time_=((inputs['time']-a)/float(b-a))*(ts.loc['time','max']-ts.loc['time','min'])+ts.loc['time','min']
            permx_=(ts.loc['permx','max']-ts.loc['permx','min'])*((inputs['permx']-a)/(b-a))+tf.math.log(ts.loc['permx','min'])
            phi_=((inputs['phi']-a)/float(b-a))*(ts.loc['poro','max']-ts.loc['poro','min'])+ts.loc['poro','min']
        elif model.cfd_type['Input_Normalization']=='lnk-linear-scaling':
            time_=((inputs['time']-a)/float(b-a))*(ts.loc['time','max']-ts.loc['time','min'])+ts.loc['time','min']
            permx_=tf.math.exp(tf.math.log(ts.loc['permx','max']/ts.loc['permx','min'])*((inputs['permx']-a)/(b-a))+tf.math.log(ts.loc['permx','min']))
            phi_=((inputs['phi']-a)/float(b-a))*(ts.loc['poro','max']-ts.loc['poro','min'])+ts.loc['poro','min']
        elif model.cfd_type['Input_Normalization']=='z-score':
            time_=inputs['time']*ts.loc['time','std']+ts.loc['time','mean']
            permx_=tf.math.exp(inputs['permx']*ts.loc['poro','std']+ts.loc['poro','mean'])
            phi_=inputs['phi']*ts.loc['poro','std']+ts.loc['poro','mean']

    if model.cfd_type['Output_Normalization']!=None:
        pred_ldata_=[]
        obs_ldata_=[]
        for key in def_output_keys[:len(output_keys)]:
            if model.cfd_type['Output_Normalization']=='linear-scaling':
                 nonorm_pred=((pred_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
                 nonorm_obs=((obs_ldata[output_keys.index(key)]-a)/float(b-a))*((ts.loc[key,'max'])-(ts.loc[key,'min']))+(ts.loc[key,'min'])
            else:
                nonorm_pred=pred_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
                nonorm_obs=obs_ldata[output_keys.index(key)]*ts.loc[key,'std']+ts.loc[key,'mean']
            pred_ldata_.append(nonorm_pred)
            obs_ldata_.append(nonorm_obs)
    else:
        pred_ldata_=pred_ldata
        obs_ldata_=obs_ldata
    return pred_ldata_,obs_ldata_,time_,phi_,permx_

# A function that saves the weights and biases of a trained model.
def save_model_weights(model,folder_path=None,dname=None,layer_wise=False,model_name='trn_err_model'):
    if hasattr (model,'cfd_type'):
        model_name=model.cfd_type['DNN_Type']
    else:
        if model_name==None:
            return print('Error!!! Enter Model Name')
    pattern_1='\,|\['
    pattern_2='\]'
    file_name_wb_best=f'/DNN_Weight_Bias_Best'+re.sub(pattern_2,'',re.sub(pattern_1,'_',str(dname)))
    file_name_wb_full=f'/DNN_Weight_Bias_Full'+re.sub(pattern_2,'',re.sub(pattern_1,'_',str(dname)))
    file_name_history=f'/DNN_History'+re.sub(pattern_2,'',re.sub(pattern_1,'_',str(dname)))
    file_path_wb_best=folder_path+file_name_wb_best+'_'+model_name
    file_path_wb_full=folder_path+file_name_wb_full+'_'+model_name
    file_path_history=folder_path+file_name_history+'_'+model_name

    if not layer_wise:
        model_weights_list=model.get_weights()
    else:
        model_weights_list={}
        for ix, layer in enumerate(model.layers):
            if hasattr(model.layers[ix], 'kernel_initializer') and (hasattr(model.layers[ix], 'bias_initializer') or hasattr(model.layers[ix],'recurrent_initializer')) :
                if not layer.name in model_weights_list:
                    model_weights_list[layer.name]=model.layers[ix].get_weights()
                else:
                    model_weights_list[layer.name].update(model.layers[ix].get_weights())
    dump(model_weights_list, open(file_path_wb_best,'wb'))  
    if hasattr (model,'history_ens'):     
        # Add the best ensemble training time (mins) to the dictionary. Useful for later reporting
        model.history_ens[model.best_ens]['train_time']=model.wblt_epoch_ens[model.best_ens]['train_time']/60
        dump(model.history_ens[model.best_ens],open(file_path_history,'wb'))
    return
# e.g., save_model_weights(model,save_model_folder_path)  

# A function that loads and sets saved weights and biases to a built model.
# The saved weights and biases are loaded from a saved file, and the saved data dimension must correspond to that of the model trainable variables.     
def load_model_weights(model,folder_path=None,dname=None,layer_wise=False,search_patterns=['Trn_Err',]):
    pattern_1='\,|\['
    pattern_2='\]'    
    if hasattr (model,'cfd_type'):
        file_name_wb_best=r'/DNN_Weight_Bias_Best'+re.sub(pattern_2,'',re.sub(pattern_1,'_',str(dname)))
        file_path_wb_best=folder_path+file_name_wb_best+'_'+model.cfd_type['DNN_Type']
    else:
        # load the first file with pattern
        pattern_1='DNN_Weight_Bias_Best'
        for file in os.listdir(folder_path):
            if file.startswith(pattern_1) and any(i in search_patterns for i in file):
                file_path_wb_best=folder_path+f'/{file}'
                break
    model_weights_list = load(open(file_path_wb_best,'rb'))
    if not layer_wise:
        model.set_weights(model_weights_list)
    else:
        for ix, layer in enumerate(model.layers):
            if hasattr(model.layers[ix], 'kernel_initializer') and (hasattr(model.layers[ix], 'bias_initializer') or hasattr(model.layers[ix],'recurrent_initializer')) :
                if not layer.name in model_weights_list:
                    continue
                else:
                    model.layers[ix]=model.layers[ix].set_weights(model_weights_list[layer.name])   
    return 

# A function that return a superimposed plots of the training losses vs. time, where the losses are obtained from a saved dictionary file. 
def plot_files(folder_path=None,plot_key=['td_loss_p','val_loss'],color=['blue','green','brown','orange',],label=['Without_Regularization','With_Regularization'], use_label=True):
    # Create a list for loading saving the opened file.
    # Loop through the folder to get the file.
    fig, ax= plt.subplots(1,2,figsize=(8,4),sharey=True,dpi=1200)          
    fig.subplots_adjust(left=0.01, bottom=0.06, right=0.95, top=0.95, hspace=0.4, wspace=0.2)
    fig.suptitle('Loss (MSE) for Non Physics-Informed Training',fontsize=9,y=1.01,weight='bold')

    pattern_1='DNN_History_'
    axis=plt.gca()
    
    # Define a random colour map.
    cmap=plt.cm.get_cmap(name='hsv',lut=10)
    num_colors=8
    import random
    cmap = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colors)]
    
    # Add the default colours
    cmap=color+cmap
    col_idx=0
    train_time_text='Train Time (mins.):'+'\n'
    for file in os.listdir(folder_path):
        if file.startswith(pattern_1):
            # Load file with pickle.
            file_path=folder_path+f'/{file}'
            history = load(open(file_path,'rb'))
            plt_label=re.sub(pattern_1,'',file)
            for key in plot_key:
                ax[plot_key.index(key)].set_title(key,fontsize=8)
                ax[plot_key.index(key)].set_yscale('log')
                ax[plot_key.index(key)].set_ylabel('MSE value')
                ax[plot_key.index(key)].set_xlabel('No. Iterations/Epoch')
                ax[plot_key.index(key)].grid(True)
                if key=='td_loss_p':
                    line_style='solid'
                else:
                    line_style='dashed'
                if use_label:
                    lbl=label[col_idx]
                else:
                    lbl_full='[{}]-{} min'.format(plt_label,'{0:.2f}'.format(history['train_time']))
                    if 'REN'.upper() in lbl_full and 'ADM-MT' in lbl_full: lbl='ResNet: Without Regularization'
                    elif 'REN'.upper() in lbl_full and 'ADW-MT' in lbl_full: lbl='ResNet: With Regularization'
                    elif 'PLN'.upper() in lbl_full and 'ADM-MT' in lbl_full: lbl='FCDNN: Without Regularization'
                    elif 'PLN'.upper() in lbl_full and 'ADW-MT' in lbl_full: lbl='FCDNN: With Regularization'
                ax[plot_key.index(key)].plot(history[key], label=lbl,color=cmap[col_idx],linestyle=line_style)
            train_time_text=train_time_text+str('{0:.2f}'.format(history['train_time']))+'\n'
            #ax1.legend(loc="upper left",  prop={'size': 8},ncol=1,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(1, 1))  
            col_idx+=1
    ax[1].text(0.825, 0.850,train_time_text, size=8.0, ha='center', va='center', transform=axis.transAxes, zorder=100)    
    plt.rc('grid', linestyle="--", color='grey',linewidth=0.5)
    ax[0].legend(loc="lower left",  prop={'size': 8},ncol=2,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0, -0.25))  
    plt.show()
    return

# A function that returns a histogram plot of the relative L1 and L2 errors, which are read from a saved Excel file. 
ticks_font = font_manager.FontProperties(family='times')
latex_map={'a0':r'$\alpha_0$','DOM':r'$\lambda_{\Omega}$','IBC':r'$\lambda_{\partial\Omega_{IBC}}$','MBC':r'$\lambda_{MBC}$','Non_Physics_Reg.':r"$\lambda_{\eta_0}$",'gamma':r'$\theta_\mathcal{h}$','L1':r'$L_1$','L2':r'$L_2$',}
def plot_hist(folder_path=None, dt_type='float32', file_name='Sensitivity',end_idx=7008,search_key=latex_map['Non_Physics_Reg.'],header_map=latex_map,\
              title_label=r'$[\lambda_{\Omega}=$0.33; $\lambda_{\partial\Omega_{\mathrm{IBC}}}=$0.33; $\lambda_{\mathrm{MBC}}=$0.33; $\alpha_0=$0.005]',add_key_string='',xlim=[[None,None],[None,None]],ylim=[None,None],hist_scale='lin',plot_type='histogram',alpha=1.,\
                  use_features=False,logscale='linear',xlabel=None,ylabel=None,density=False,colour=None):  #\alpha_0=0.005
    if folder_path is None:
        folder_path=f'C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Training_Data/Sensitivity'

    error_data=pd.read_excel(f'{folder_path}/{file_name}.xlsx', sheet_name=None,engine='openpyxl',dtype=dt_type)
    end_idx=error_data[list(error_data.keys())[0]].shape[0]
    old_key=list(error_data.keys())
    new_key=[]
    for key in error_data.keys():
        for subkey, subvalue in header_map.items():
            key=key.replace(subkey,subvalue)
        key="".join([key,add_key_string])
        new_key.append(key)
    error_data={new_key[i]:error_data[old_key[i]] for i in range(len(old_key))}
    csfont = {'fontname':'Times New Roman'}
    if plot_type.lower()=='histogram':
        end_idx-=2

        no_bins=max([int(np.divide(np.max(error_data[key].iloc[:end_idx,0])*4,np.std(error_data[key].iloc[:end_idx,0]),out=np.zeros(()),where=np.std(error_data[key].iloc[:end_idx,0])!=0)) for key in error_data.keys()])
        zord=[np.mean(error_data[key].iloc[:end_idx,0]) for key in error_data.keys()]
        zord=list(np.max(np.argsort(np.argsort(zord)))-np.argsort(np.argsort(zord)))

        dim=(1,2)
        fig,ax=plt.subplots(*dim,dpi=300)
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.3)
        ax=np.reshape(ax,dim)
        #fig.suptitle(f'Average grid relative $L_1$ and $L_2$ errors for 146 by 44 permeability-time realizations \n{title_label}',fontsize=9,y=1.05,weight='regular',**csfont)
        fig.suptitle(f'{title_label}',fontsize=9,y=-0.05,weight='regular',**csfont)
        subplot_label=['A','B']
        for key in error_data.keys():
            if search_key in key: 
                key_idx=list(error_data.keys()).index(key)
                edge_colour='grey'
                linewidth=0.25
                if key_idx>=0:
                    if key_idx==np.max(zord):
                        edge_colour='grey'
                        linewidth=0.25
                    if hist_scale=='lin':
                        bins=[np.linspace(min(error_data[key].iloc[:end_idx,i]),max(error_data[key].iloc[:end_idx,i]),no_bins) for i in range(dim[1])]
                    else:
                        bins=[10.**np.linspace(np.log10(min(error_data[key].iloc[:end_idx,i])),np.log10(max(error_data[key].iloc[:end_idx,i])),no_bins) for i in range(dim[1])]
                        [ax[0,i].set_xscale('log') for i in range(dim[1])]
                    
                    [ax[0,i].hist(error_data[key].iloc[:end_idx,i], bins[i], alpha=1., label=key,edgecolor=edge_colour,color=colour,zorder=zord[key_idx],linewidth=linewidth,density=density) for i in range(dim[1])]
                    [ax[0,i].legend(loc='upper right',prop={'size': 8},) for i in range(dim[1])]
                    [ax[0,i].set_xlabel(f'Relative $L_{i+1}$ error', fontsize=9,) for i in range(dim[1])]
                    [ax[0,i].set_ylabel('Frequency', fontsize=9,) for i in range(dim[1])]
                    [ax[0,i].text(0.025,0.95, subplot_label[i], fontsize = 12,transform=ax[0,i].transAxes) for  i in range(dim[1])]
                    
                    [[ax[0,i].spines[axis].set_linewidth(0.5) for axis in ['top', 'bottom', 'left', 'right']] for i in range(dim[1])]
        
                    [ax[0,i].tick_params(axis='both', which='major',labelsize=9) for i in range(dim[1])]
                    [ax[0,i].set_xlim([xlim[i][0],xlim[i][1]]) for i in range(dim[1])]
                    if ylim[0] is not None:
                        [ax[0,i].set_ylim([ylim[i][0],ylim[i][1]]) for i in range(dim[1])]
                    #ax[0,1].Axes.set_xticks(ax[0,1].get_xticks(),fontname='Times New Roman')
    elif plot_type.lower()=='xyscatter':
        zord=[np.mean(error_data[key].iloc[:end_idx,0]) for key in error_data.keys()]
        zord=list(np.max(np.argsort(np.argsort(zord)))-np.argsort(np.argsort(zord)))
        
        dim=(1,1)
        fig,ax=plt.subplots(*dim,dpi=300)
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.95, hspace=0.5, wspace=0.3)
        ax=np.reshape(ax,dim)
        fig.suptitle(f'{title_label}',fontsize=9,y=-0.05,weight='regular',**csfont)
        for key in error_data.keys():
            if search_key in key: 
                key_idx=list(error_data.keys()).index(key)
                edge_colour='grey'
                linewidth=1.5
                if key_idx>=0:
                    if key_idx==np.max(zord):
                        edge_colour='grey'
                        linewidth=2.0
                    if not use_features:
                        ax[0,0].plot(error_data[key].iloc[:end_idx,1], label=key,linewidth=linewidth)
                    else:
                        ax[0,0].plot(error_data[key].iloc[:end_idx,0],error_data[key].iloc[:end_idx,1], label=key,linewidth=linewidth,marker='s',markersize=4.,markeredgewidth=1.,markeredgecolor='black')
                    ax[0,0].set_ylabel(ylabel)
                    ax[0,0].set_xlabel(xlabel)
                    ax[0,0].set_yscale(logscale)
                    ax[0,0].legend(loc="upper right",  prop={'size': 8,},)  
                    [[ax[0,i].spines[axis].set_linewidth(0.5) for axis in ['top', 'bottom', 'left', 'right']] for i in range(dim[1])]
                    ax[0,0].grid(True)
                    if xlim[0] is not None:
                        [ax[0,i].set_xlim([xlim[i][0],xlim[i][1]]) for i in range(dim[1])]
                    if ylim[0] is not None:
                        [ax[0,i].set_ylim([ylim[i][0],ylim[i][1]]) for i in range(dim[1])]
                    plt.rc('grid', linestyle="--", color='grey',linewidth=0.5)



    plt.show()        
# Examples: 
# plot_hist(file_name='Sensitivity_Fit',hist_scale='lin',xlim=[[0.,0.015],[0.,0.0002]],ylim=[[0,2500],[0,2500]],colour='orange')
# plot_hist(file_name='Sensitivity',hist_scale='log',xlim=[[0.,0.03],[0.,0.001]],ylim=[[0,1500],[0,1500]])
# plot_hist(file_name='Sensitivity_MBC',title_label=r"[$\lambda_{\eta_0}=$0.0005$\alpha_0$; $\alpha_0$=0.005]",search_key='MBC',xlim=[[0.,0.5],[0.,0.25]],ylim=[[0,2500],[0,2500]],hist_scale='log')
# plot_hist(file_name='MSE_losses_MBC',title_label='',add_key_string="; "+r"$\lambda_{\eta_0}=$"+"0.0005"+r"$\alpha_0;\alpha_0=$"+r"0.005",search_key='MBC',plot_type='xyscatter',xlabel='Epoch',ylabel='Total loss',logscale='log')
# plot_hist(file_name='Sensitivity_IBC',title_label=r"$[\lambda_{\eta_0}=0.0005\alpha_0;\:\alpha_0=0.005]$",search_key='MBC',xlim=[[0.,0.02],[0.,0.001]],ylim=[[0,1000],[0,1000]],hist_scale='log',density=False)
# plot_hist(file_name='Sensitivity_Hard_Enforcement',title_label=r"[$\lambda_{\Omega}=$0.33; $\lambda_{\partial\Omega_{IBC}}=$0.33; $\lambda_{MBC}=$0.33; $\lambda_{\eta_0}=$0.0005$\alpha_0$; $\alpha_0=$0.005]",search_key='',xlim=[[1e-4,10.],[1e-6,1]],ylim=[[0,2000],[0,2000]],hist_scale='log')
# plot_hist(file_name='Sensitivity_Timestep',title_label=r"[$\lambda_{\Omega}=$0.33; $\lambda_{\partial\Omega_{IBC}}=$0.33; $\lambda_{MBC}=$0.33; $\lambda_{\eta_0}=$0.0005$\alpha_0$; $\alpha_0=$0.005]",search_key='',xlim=[[1e-4,0.1],[1e-7,0.01]],ylim=[[0,2200],[0,2200]],hist_scale='log')
# plot_hist(file_name='Non_Physics_Reg_Grid_Search',title_label="",search_key='L',plot_type='xyscatter',xlabel=r'Non-physics-based regularization weight, $\lambda_{\eta_0}\left(\times\alpha_0\right)$',ylabel=r'Mean relative $L_1$ $\left(\times10^{-3}\right)$ and $L_2$ $\left(\times10^{-5}\right)$ errors',logscale='linear',use_features=True,xlim=[[-0.00005,0.05]],ylim=[[1.,6.]])

def plot_MSE(folder_path=None, dt_type='float32', sheet_name='Total_Loss'):
    if folder_path is None:
        folder_path=f'C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Training_Data'

    total_losses=pd.read_excel(f'{folder_path}/MSE_Losses.xlsx', sheet_name=sheet_name,engine='openpyxl',dtype=dt_type)
    fig,ax=plt.subplots(1,1,dpi=1200)
    fig.suptitle(f'Total MSE losses',fontsize=8,y=0.925,weight='bold')   
    for key in total_losses.keys():
        ax.plot(total_losses[key], label=key,linewidth=2.0)   
 
    ax.set_ylabel('MSE value')
    ax.set_xlabel('No. Iterations/Epoch')
    ax.set_yscale('log')
    ax.legend(loc="upper left",  prop={'size': 8},ncol=1,handleheight=1, handlelength=1,labelspacing=0.05,bbox_to_anchor=(0.5, 1))  
    ax.grid(True)
    plt.rc('grid', linestyle="--", color='grey',linewidth=0.5)

