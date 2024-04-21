# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:46:05 2021

@author: Victor Molokwu
Heriot-Watt University
"""
# =================================================== Directory Settings =========================================================
import sys
# Set system path and 'KLE' module directory. This is to be changed to the directory of the user.
sys.path.append(r'C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Main_Library')
sys.path.append(r'C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Main_Library/kle')

# Set the save folder directory. This is to be changed to the directory of the user.
save_folder_path=r'C:/Users/vcm1/Documents/PHD_HW_Machine_Learning/ML_Cases/Training_Data/PINN_Data_Dumps'
sim_folder_path=r'C:/Users/vcm1/Documents/PHD_HW_Machine_Learning/ML_Cases/Training_Data/Simulation_Data' 
#sim_folder_path=r'//petfiler/pgr$/vcm1/Documents/PHD_HW_Machine_Learning/ML_Cases/Training_Data/Simulation_Data'
# ================================================================================================================================
import pandas as pd
import re
import os
import numpy as np
import time as t
from scipy.stats import lognorm
import gc

import kle
import PVT_models as pvt
from pickle import load 
from pickle import dump 

# Set a datatype to use where necessary
dt_type=np.float32
start_time=t.time()
# ===================================================== Fluid Type Settings ======================================================
fluid_type='DG'                                                       #'dry-gas'
fluid_comp='MC'                                                       # MC: Multi component | BC: Binary Component | SC: Single component
# ===================================================== Grid Dimensions ==========================================================
w_info={'nwells':4,'ncon':4}
init_rate={'gas':[10000.,10000.,5000.,5000.,]}                      # Used to read the simulation restart file for the dataset
min_bhp=[4000.,3000.,2000.,1000.,]                                  # Used to read the simulation restart file for the dataset

grid_dim={'Number':(39,39,1),'Measurement':(2900,2900,80), 'Datum':11000}
conn_idx=[(9,29,0),(9,9,0),(29,9,0),(29,29,0),] 
gridblock_dim=(grid_dim['Measurement'][0]/grid_dim['Number'][0],grid_dim['Measurement'][1]/grid_dim['Number'][1],grid_dim['Measurement'][2]/grid_dim['Number'][2])
t0=np.zeros(grid_dim['Number'],dtype=dt_type)
x=((np.linspace(0,grid_dim['Measurement'][0],grid_dim['Number'][0]+1,dtype=dt_type)[:-1]+np.linspace(0,grid_dim['Measurement'][0],grid_dim['Number'][0]+1,dtype=dt_type)[1:])*0.5)
y=((np.linspace(0,grid_dim['Measurement'][1],grid_dim['Number'][1]+1,dtype=dt_type)[:-1]+np.linspace(0,grid_dim['Measurement'][1],grid_dim['Number'][1]+1,dtype=dt_type)[1:])*0.5)
z=grid_dim['Datum']+((np.linspace(0,grid_dim['Measurement'][2],grid_dim['Number'][2]+1,dtype=dt_type)[:-1]+np.linspace(0,grid_dim['Measurement'][2],grid_dim['Number'][2]+1,dtype=dt_type)[1:])*0.5)
x_mgrid,y_mgrid,z_mgrid=np.meshgrid(x,y,z)

di=gridblock_dim[0]*np.ones(grid_dim['Number'],dtype=dt_type)
dj=gridblock_dim[1]*np.ones(grid_dim['Number'],dtype=dt_type)
dk=gridblock_dim[2]*np.ones(grid_dim['Number'],dtype=dt_type)
phi=0.2*np.ones(grid_dim['Number'],dtype=dt_type)
ani=0.1
lbl=np.zeros(grid_dim['Number']+(2,),dtype=dt_type)
train_labels=True                                                    
# ====================================================== Realization Settings ====================================================
tstep=1.
kfunc='lognormal'                                           # 'linear'|'power'|'lognormal'.
static_prop_type='RP'                                       # RP: RandProp| UN: Uniform Property.
m=3                                                         # The mean of the distribution.
sd=1                                                        # The standard deviation of the distribution
sigma=np.sqrt(np.log(((sd/m)**2)+1))                        # The standard deviation of the unique normal distribution of the log variables.
mu=np.log(m)-0.5*sigma**2                                   # The mean of the unique normal distribution of the log variable. 

# Sampling of the training, validation and testing datasets  
nrealz=60                                                   # The number of (traning+validation+testing) permeability realizations. 
sen_samples=200                                             # An extended permeability realization dataset, which includes more test permeability realizations

ntsteps=int(540/1)                                          # Number of timepoints at which each permeabily realization is to generated/run. Note: This excludes the initial timestep t==0
ltone=[]#[0.0001,0.001,0.01,0.1]                            # A list which contains timepoints less than 1, i.e., very early decimal timepoints.
start_bu=152                                                # Timepoint (if any) where a shut-in starts 
end_bu=243                                                  # Timepoint (if any) where a shut-in ends 
step=5
tstep_bu=[start_bu,end_bu]
ntsteps_list=sorted(ltone+list(range(ntsteps+1)))


# The training, validation and test permeability samples are sampled such that a realization instance occurs in only one of the 
# training, testing or validation datasets. 
train_realz_start=int(1)                                    # The start index for sampling the training permeability from the realizations. 
train_realz_step=int(1)                                     # The step during sampling the train permeability from the realizations. 
sen_realz_start=int(0)                                      # The start index for sampling the (validation+test) permeability from the realizations.                                       
sen_realz_step=int(10)                                      # The step during sampling the (validation+test) permeability from the realizations.    

# ===================================================== Permeability Realization =================================================
# Conditional statment which allow for uniform permeabilty realizations.
# In a uniform permeability realization, the permeabilty distribution across the domain is constant (i.e., permeability constrast in the 
# domain is zero) for each realization but varies between different realizations. The variation of the permeability across the realizations
# is controlled using a function, which depends on the realization index. 
train_realz_end=nrealz
sen_realz_end=nrealz
sen_realz=list(range(sen_realz_start,sen_realz_end,sen_realz_step))
train_realz=sorted(set(range(nrealz))-set(sen_realz))  
if static_prop_type=='RP':  
    filename_prefix=f'{grid_dim["Number"][0]}x{grid_dim["Number"][1]}x{grid_dim["Number"][2]}_{m}mD_{sd}SD_{sen_samples}KLE'
    _,kx=kle.KLE_samples(no_samples=sen_samples,rseed=50,mu=m,sd=sd,grid_dim=grid_dim['Number'],gridblock_size=gridblock_dim,corr_len_fac=0.2,energy_fac=0.98,train_test_samp=1,write_out=True,filename_prefix=filename_prefix)
else:
    k=lambda x:1.16**(x)
    ks=lambda x:1.15**(x)
    if kfunc=='power' :
        sen_realz_end=int(np.log(k(nrealz-1))/np.log(ks(1)))+1
        sen_realz=list(range(sen_realz_start,sen_realz_end,sen_realz_step))
        train_realz=sorted(set(range(nrealz))-set(sen_realz))  
        kx=[k(i)*np.ones(grid_dim['Number'],dtype=dt_type) for i in range(len(train_realz))]
    elif kfunc=='lognormal':
        sen_realz_end=int(np.log(lognorm.ppf(0.99,sigma,loc=0.,scale=np.exp(mu)))/np.log(ks(1)))+1
        sen_realz=list(range(sen_realz_start,sen_realz_end,sen_realz_step))
        train_realz=sorted(set(range(nrealz))-set(sen_realz))  
        kx=[lognorm.ppf(i,sigma,loc=0.,scale=np.exp(mu))*np.ones(grid_dim['Number'],dtype=dt_type) for i in np.linspace(0.015,0.99,len(train_realz)+len(sen_realz))]
    else:
        max_k=k(nrealz-1)
        min_k=k(1-1)
        no_k=len(train_realz)
        diff=(max_k-1)/(no_k-1)
        k_lin=lambda x:min_k+(x*diff)
        kx=[k_lin(i)*np.ones(grid_dim['Number'],dtype=dt_type) for i in range(len(train_realz))]
        # Update the permeability values with simulation-based func. validation data at index points list: sen_realz
    kx_sen=[np.mean(kx[i])*np.ones(grid_dim['Number'],dtype=dt_type) for i in sen_realz]   

kz=[ani*kx[i] for i in range(len(kx))]    

ext_sen_realz=list(set(range(sen_samples))-set(train_realz)) 
if len(train_realz)%2!=0:
    import sys
    print('Train realization not even, might experience problem creating batch sizes to run in graph mode')
    sys.exit()
# ====================================================== Recurrent Property Generation ===========================================
# Pressure and saturation obtained from simulation restart 'UNRST' file.
# These are used to compare the accuracy of the physics-based semi-supervised learning or used a labels during 
# the non-physics-based supervised learning.
# Load the pressure restart files.
folder_path= (f'{sim_folder_path}/{grid_dim["Number"][0]}x{grid_dim["Number"][1]}x{grid_dim["Number"][2]}_'
            f'{fluid_type}_{fluid_comp}_{static_prop_type}_Square_Boundary_{m}mD_{sd}SD_{w_info["nwells"]}W_{int(np.sum(init_rate["gas"]))}MScf_{int(np.mean(min_bhp))}psi_{sen_samples}KLE')
gr_dynamic=load(open(f'{folder_path}/Python_Input/gr_dynamic','rb'))

# Create a dictionary of pressure and saturation outputs.
pre_sat={}

for i in range(sen_samples): #ext_sen_realz:
    dict_list=gr_dynamic[(gr_dynamic['perturbation']==i)][['pressure','sgas']].to_dict(orient='list')
    
    # Reshape the dictionary list as array
    for key,value in dict_list.items():
        dict_list[key]=np.reshape(np.array(value),(-1,*grid_dim['Number']))

    # Add to the parent list
    pre_sat[str(i)]=dict_list
# ============================================================ Dataset Generation ================================================
# Sequential generation of the permeabilit-time feature-label dataset.
# The features are generated using mesh-grid functions, KLE expansion and the labels are copied from a simulation 
# restart file already loaded in the workspace.
# Features (inputs): 
coord_x=[]
coord_y=[]
coord_z=[]
time=[]
poro=[]
permx=[]
permz=[]
label=[]
timestep=[]
perturb=[]

# Labels (outputs):
pre=[]
gsat=[]
grate=[]

for i in range(len(train_realz)+len(ext_sen_realz)):
    # Create list variables to hold the generated permeability-time dataset for each realization.
    # Features (inputs):
    coord_x_realz=[]
    coord_y_realz=[]
    coord_z_realz=[]
    time_realz=[]
    poro_realz=[]
    permx_realz=[]
    permz_realz=[]
    label_realz=[]
    timestep_realz=[]
    perturb_realz=[]
    
    # Labels (outputs):
    pre_realz=[]
    gsat_realz=[]
    grate_realz=[]
    
    # Set additional input identifiers for each realization
    # This can be useful to distinguish the training dataset from the (test+validation) datasets
    if i in sen_realz:
        lbl[...,1]=1.
    else:
        lbl[...,1]=0.
    
    for j in (ntsteps_list):
        coord_x_realz.append(x_mgrid)
        coord_y_realz.append(y_mgrid)
        coord_z_realz.append(z_mgrid)
        time_realz.append([t0+(j)*tstep])
        poro_realz.append(phi)
        permx_realz.append(kx[i])
        permz_realz.append(kz[i])
        label_realz.append(lbl)
        timestep_realz.append(j*np.ones(grid_dim['Number'],dtype=dt_type))
        perturb_realz.append(i*np.ones(grid_dim['Number'],dtype=dt_type))
        
        # Add the well block rates. A cumulative production can also be created within the conditional statment
        qi=np.zeros(grid_dim['Number'],dtype=dt_type)
        j_idx=ntsteps_list.index(j)
        if j<ntsteps_list[-1]:
            if j>0:
                for k in range(len(conn_idx)):
                    qi[conn_idx[k]]=init_rate['gas'][k]
            
            grate_realz.append(qi)
        else:
            grate_realz.append(qi)
       
        # Add the grid block pressures and saturation (if any)
        if j not in ltone:
            if (i in ext_sen_realz): # and static_prop_type!='rand':
                pre_realz.append(pre_sat[str(i)]['pressure'][int(j),...])
                gsat_realz.append(pre_sat[str(i)]['sgas'][int(j),...])
            else: 
                if train_labels:
                    pre_realz.append(pre_sat[str(i)]['pressure'][int(j),...])
                    gsat_realz.append(pre_sat[str(i)]['sgas'][int(j),...])
                else:
                    pre_realz.append(np.zeros(grid_dim['Number'],dtype=dt_type))
                    gsat_realz.append(np.zeros(grid_dim['Number'],dtype=dt_type))                                   
        else:
            # Decimal timepoints less than one day are sampled as zeros as the 
            # simulator restart files are saved at a 1-day timestep
            pre_realz.append(np.zeros(grid_dim['Number'],dtype=dt_type))
            gsat_realz.append(np.zeros(grid_dim['Number'],dtype=dt_type))    

    # Append to the parent (outer) list
    coord_x.append(coord_x_realz)
    coord_y.append(coord_y_realz)
    coord_z.append(coord_z_realz)
    time.append(time_realz)
    poro.append(poro_realz)
    permx.append(permx_realz)
    permz.append(permz_realz)
    label.append(label_realz)
    timestep.append(timestep_realz)
    perturb.append(perturb_realz)
    
    pre.append(pre_realz)
    gsat.append(gsat_realz)
    grate.append(grate_realz)

# Reshape features and labels to 1D for use with a variety of input-output dimensions. 
# 1-D features (inputs):
coord_x=np.reshape(coord_x,(-1))
coord_y=np.reshape(coord_y,(-1))
coord_z=np.reshape(coord_z,(-1))
time=np.reshape(time,(-1))
poro=np.reshape(poro,(-1))
permx=np.reshape(permx,(-1))
permz=np.reshape(permz,(-1))
label=np.reshape(label,(-1,2))
timestep=np.reshape(timestep,(-1))
perturb=np.reshape(perturb,(-1))

# 1-D labels (outputs): 
pre=np.reshape(pre,(-1))
gsat=np.reshape(gsat,(-1))
grate=np.reshape(grate,(-1))

#End of File       
# Create a dictionary of the input-output dataset.
df_data={'x_coord':coord_x,'y_coord':coord_y,'z_coord':coord_z,'time':time,'poro':poro,'permx':permx,'permz':permz,\
         'Label_1':label[:,0],'Label_2':label[:,1],'pressure':pre,'gsat':gsat,'grate':grate,\
          'timestep':timestep,'perturbation':perturb}

# Release the variables to free memory.
del coord_x; del coord_x_realz
del coord_y; del coord_y_realz
del coord_z; del coord_z_realz
del time; del time_realz
del poro; del poro_realz
del permx; del permx_realz
del permz; del permz_realz
del label; del label_realz
del timestep; del timestep_realz
del perturb; del perturb_realz
del pre; del pre_realz
del gsat; del gsat_realz
del grate; del grate_realz
del pre_sat
del x; del y; del z;
del x_mgrid; del y_mgrid; del z_mgrid
del lbl; del phi; del qi; del t0
del kx; del kz; del di; del dj; del dk
del gr_dynamic
gc.collect()

# Set datatype
dtype={i:'float32' for i in df_data if i not in ['perturbation','timestep']}|{i:'int32' for i in df_data if i in ['perturbation','timestep']}
gr_domain=pd.DataFrame(data=df_data,dtype=dt_type)
# Save the dataset as a DataFrame.      
def update_dumps(save_type='zip'):
    if save_type.lower()=='zip':
        gr_domain.astype(dtype).to_csv(f'{folder_path}/Python_Input/gr_domain.zip', compression={'method': 'zip', 'compresslevel': 9})
    else:   
        dump(gr_domain.astype(dtype), open(f'{folder_path}/Python_Input/gr_domain','wb'))
    # Or to get more control
    

update_dumps(save_type='normal')

#gr_domain = pd.read_csv(f'{folder_path}/Python_Input/gr_domain.zip')
gr_domain=load(open(f'{folder_path}/Python_Input/gr_domain','rb'))

# ========================================================= Dataset Splitting ====================================================
# Dataset spliting: the generated data is splitted into training, validation and test Sets, and thereafter normalized

# Dataset splitting settings: 
dim_model='/2D'                                     # Model dimension: used to create a unique file name
arr_type='1'
static_prop_type=f'{static_prop_type}_0'            # Options: 'uni-prop_**' | 'rand-prop_**'   ** is the count
numeric_type='IM'                                   # IMP: Implicit | EXP: Explicit.

data_frac=1.                                        # Fraction of generated dataset, along the time axis, used for (training+validation+testing)
                                                    # A fraction of 1 implies all the generated dataset, along the time axis, are used.
tstep_samp=10.                                      # Timestep: decimal interval at which the generated dataset is sampled along the time axis.
train_frac=0.7                                      # Fraction of the dataset used for training+testing.
FVTDS_bool=True                                     # Full Val_Test Data Sampling (FVTDS).
input_norm='lnk-linear-scaling'                     # linear-scaling, z-score lnk-linear-scaling.
output_norm=None
lscale={'Min':-1.,'Max':1}                          # linear-scaling range (if used).

rng=[0,7,12]                                        # Column index indicating the [start index of features: end index of feature/start index of label:
                                                    # end index of label] of the generated dataset.

# Splitting function
# @nb.jit(nopython=True)
def split_df(dataset, split_frac={'Train':0.7,'Val':0.1,'Samp_File':[False,None],'Full_Samp_Val_Test':True}, f_index={'Start':0,'End':12}, l_index={'Start':12,'End':16},\
             samp_par={'Tstep_step':1,'Ntsteps':1096,'Ntsteps_Frac':0.7,'Nrealz':50,'Pert_List':list(range(0,50)),'Sen_List':list(range(5,50,5)),'Tscale':'Normal','Log_Step':3.5,'Val_Split_Position':'end'}):

    if f_index['Start']==None:
        f_index['Start']=0

    if f_index['End']==None:
        f_index['End']=len(dataset.columns)-4           #     

    if l_index['Start']==None:
        l_index['Start']=len(dataset.columns)-4

    if l_index['End']==None:
        l_index['End']=len(dataset.columns)             #     
    
    if split_frac['Train']+split_frac['Val']>1:
        print ('Validation Split Fraction Set to Zero')
        split_frac['Val']=0.0
        #return
    if split_frac['Val']==None:
        split_frac['Val']=1-(split_frac['Train'])

    # Determine the maximum number of timesteps used for training+validation+testing.
    max_ntstep=np.int32(samp_par['Ntsteps']*samp_par['Ntsteps_Frac'])

    if samp_par['Tscale']=='log':
        tstep_range=np.unique(np.logspace(0,np.log10(max_ntstep+1),num=np.int32(max_ntstep/samp_par['Log_Step'])).astype(np.int32))-1     # -1 is used to return to zero-based indexing
    else:
        tstep_range=np.array(range(0,max_ntstep,int(samp_par['Tstep_step'])))
    

    no_samp_tsteps=len(tstep_range)
    train_maxtstep_idx=np.int32(np.ceil(split_frac['Train']*no_samp_tsteps))
    train_val_maxtstep_idx=np.int32(np.ceil((split_frac['Train']+split_frac['Val'])*no_samp_tsteps))
    no_val_idx=np.int32(split_frac['Val']*no_samp_tsteps) 
    
    tstep_range_train=tstep_range[0:train_maxtstep_idx]
    if samp_par['Val_Split_Position']=='inbetween':
        tstep_range_val=[]
        for i in range(no_val_idx):
            dropval_idx=np.int32(val_interval*(1+i))
            if dropval_idx<len(tstep_range_train_val):
                tstep_range_val.append(tstep_range_train_val[dropval_idx])
    elif samp_par['Val_Split_Position']=='end' or samp_par['Val_Split_Position']==None:
        tstep_range_val=tstep_range[train_maxtstep_idx:train_val_maxtstep_idx] 
    
    tstep_range_test=tstep_range[train_val_maxtstep_idx:no_samp_tsteps]
   
    if len(tstep_range_train)!=0:
        # Check if the train tstep is odd. Even is desirable to allow stacking in graph mode. Any extra timestep is added to the validation data
        tstep_add=int(tstep_samp) ##Set at default of 10
        tstep_bu1=tstep_bu[:(np.abs(np.asarray(tstep_bu) - np.max(tstep_range_train))).argmin()+1]
        tstep_range_train=np.unique(ltone+list(range(int(tstep_add)))+list(tstep_range_train)+tstep_bu1)

        if len(tstep_range_train)%2!=0:
            # remove a timestep between 1 and 10
            tstep_range_train=np.delete(tstep_range_train,tstep_add-1)
        
        # Check its a multiple of 4 - efficient and equal batch sizing
        if len(tstep_range_train)%4!=0:           
            for d in range(len(tstep_range_train)%4):
                tstep_range_train=np.delete(tstep_range_train,-1)
        print('No of Train Tsteps:',len(tstep_range_train), '| Max Train Time:',np.max(tstep_range_train)) 

    else:
        print('No of Train Tsteps:',len(tstep_range_train), '| Max Train Time:',0.) 
        
    # Check if validation loss is empty.
    if len(tstep_range_val)==0:
        # if empty, the last 5% of the training timesteps is used to create a timestep range for both validation and test data.
        # This is done to prevent null dataframe datasets.
        perc_val=0.05; perc_test=0.05
        st_val_idx=int((1-perc_val)*len(tstep_range_train))
        st_test_idx=int((1-perc_test)*len(tstep_range_train))
        tstep_range_val=tstep_range_train[st_val_idx:]
        tstep_range_test=tstep_range_train[st_test_idx:]
    
    # Get the timepoints from the timesteps (series-like data type without duplicates). 
    time_range_train=dataset[dataset['timestep'].isin(tstep_range_train)]['time'].drop_duplicates()
    time_range_val=dataset[dataset['timestep'].isin(tstep_range_val)]['time'].drop_duplicates()
    time_range_test=dataset[dataset['timestep'].isin(tstep_range_test)]['time'].drop_duplicates()

    par_idx_list=list(set(samp_par['Pert_List']).union(set(samp_par['Sen_List'])))
    tstep_idx_list=np.unique(np.concatenate([tstep_range_train,tstep_range_val,tstep_range_test]))
    f_keys=list(dataset.keys()[f_index['Start']:f_index['End']])
    l_keys=list(dataset.keys()[l_index['Start']:l_index['End']])
    
    def np_reshape_append(dataset,dataset_ext,field_list=None,reshape_dim=(39,39,1)):
        for k in range(len(field_list)):
            dataset[k].append(np.reshape(dataset_ext[k],grid_dim['Number'])) 
        return
    def np_flatten_to_df(np_array_list=None,index_list=None):
        np_flat_list=[]
        for k in range(len(index_list)):
            np_flat_list.append(np.reshape(np_array_list[k],(-1)))
        return pd.DataFrame(np_flat_list,index=index_list).T 
    
    for j in par_idx_list:
        for i in tstep_idx_list:
            dataset_tstep=dataset[(dataset['timestep']==i) & (dataset['perturbation']==j)]     #Due to overload, use bitwise operators |,&
            fdataset_tstep=dataset_tstep[f_keys]
            ldataset_tstep=dataset_tstep[l_keys]
            # fdataset_tstep=[dataset_tstep[f_keys][k].tolist() for k in f_keys]
            # ldataset_tstep=[dataset_tstep[l_keys][k].tolist() for k in l_keys]
            #=====================================================================================================================        
            if j==par_idx_list[0] and i==tstep_idx_list[0]:
                # Initialize DataFrame object for each variable
                # Features
                train_fdata=pd.DataFrame(dtype=dt_type); val_fdata=pd.DataFrame(dtype=dt_type); test_fdata=pd.DataFrame(dtype=dt_type)
                # train_fdata=[list() for k in f_keys]; val_fdata=[list() for k in f_keys]; test_fdata=[list() for k in f_keys];
                
                # Labels
                train_ldata=pd.DataFrame(dtype=dt_type); val_ldata=pd.DataFrame(dtype=dt_type); test_ldata=pd.DataFrame(dtype=dt_type)
                #train_ldata=[list() for k in l_keys]; val_ldata=[list() for k in l_keys]; test_ldata=[list() for k in l_keys];
                  
            if i in tstep_range_train and j in samp_par['Pert_List']:
                # Split the Dataset within a timestep
                #======================================== DOMAIN CONDITION PDE SOLUTION DATASET ==================================
                # Create the feature and labels of current timestep dataset with previous timesteps.  Info: Concatenate the DataFrame to index 0--concatenate is more efficient that append
                train_fdata=pd.concat([train_fdata, fdataset_tstep], ignore_index=False)
                train_ldata=pd.concat([train_ldata, ldataset_tstep], ignore_index=False)

                # np_reshape_append(train_fdata,fdataset_tstep,field_list=f_keys,reshape_dim=grid_dim['Number'])
                # np_reshape_append(train_ldata,ldataset_tstep,field_list=l_keys,reshape_dim=grid_dim['Number'])

            if i in tstep_range_val and j in samp_par['Sen_List']:
                #======================================== DOMAIN VALIDATION DATASET ==============================================
                val_fdata=pd.concat([val_fdata, fdataset_tstep], ignore_index=False)
                val_ldata=pd.concat([val_ldata, ldataset_tstep], ignore_index=False)
                # np_reshape_append(val_fdata,fdataset_tstep,field_list=f_keys,reshape_dim=grid_dim['Number'])
                # np_reshape_append(val_ldata,ldataset_tstep,field_list=l_keys,reshape_dim=grid_dim['Number'])

            if i in tstep_range_test and j in samp_par['Sen_List']: 
                #======================================== DOMAIN TEST DATASET ====================================================
                test_fdata=pd.concat([test_fdata, fdataset_tstep], ignore_index=False)
                test_ldata=pd.concat([test_ldata, ldataset_tstep], ignore_index=False)
                # np_reshape_append(test_fdata,fdataset_tstep,field_list=f_keys,reshape_dim=grid_dim['Number'])
                # np_reshape_append(test_ldata,ldataset_tstep,field_list=l_keys,reshape_dim=grid_dim['Number'])

    # train_fdata=np_flatten_to_df(train_fdata,index_list=f_keys)
    # train_ldata=np_flatten_to_df(train_ldata,index_list=l_keys)
    # val_fdata=np_flatten_to_df(val_fdata,index_list=f_keys)
    # val_ldata=np_flatten_to_df(val_ldata,index_list=l_keys)
    # test_fdata=np_flatten_to_df(test_fdata,index_list=f_keys)
    # test_ldata=np_flatten_to_df(test_ldata,index_list=l_keys)
                    
    # Generator function for the dataframe
    # def generator_perm_time(par_idx_list=None,tstep_idx_list=None,samp_tstep_idx_list=None,samp_par_idx_list=None,field_type='features'):
    #     for j in par_idx_list:
    #         for i in tstep_idx_list:
    #             dataset_tstep=dataset.loc[(dataset['timestep']==i) & (dataset['perturbation']==j)]     #Due to overload, use bitwise operators |,&
    #             if field_type.lower()=='features':
    #                 fdataset_tstep=dataset_tstep.iloc[:,f_index['Start']:f_index['End']]
    #             else:
    #                 fdataset_tstep=dataset_tstep.iloc[:,l_index['Start']:l_index['End']]
    #             #=====================================================================================================================        
    #             if i in samp_tstep_idx_list and j in samp_par_idx_list:
    #                 # Split the Dataset within a timestep
    #                 yield fdataset_tstep
                        
    # Generate the concatenated dataframes
    # train_fdata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_train,samp_par_idx_list=samp_par['Pert_List'],field_type='features'))
    # train_ldata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_train,samp_par_idx_list=samp_par['Pert_List'],field_type='labels'))
    # val_fdata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_val,samp_par_idx_list=samp_par['Sen_List'],field_type='features'))
    # val_ldata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_val,samp_par_idx_list=samp_par['Sen_List'],field_type='labels'))
    # test_fdata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_test,samp_par_idx_list=samp_par['Sen_List'],field_type='features'))
    # test_ldata=pd.concat(generator_perm_time(par_idx_list,tstep_idx_list,samp_tstep_idx_list=tstep_range_test,samp_par_idx_list=samp_par['Sen_List'],field_type='labels'))

    return train_fdata, train_ldata, val_fdata, val_ldata, test_fdata, test_ldata

# Generate the data
train_fdata,train_ldata,val_fdata,val_ldata,test_fdata,test_ldata=split_df(gr_domain, split_frac={'Train':train_frac,'Val':0.05,'Samp_File':[False,None],'Full_Samp_Val_Test':True}, f_index={'Start':0,'End':9}, l_index={'Start':9,'End':12},\
            samp_par={'Tstep_step':int(tstep_samp/tstep),'Ntsteps':(ntsteps+1),'Ntsteps_Frac':data_frac,'Nrealz':nrealz,'Pert_List':train_realz,'Sen_List':sen_realz,'Tscale':'Normal','Log_Step':4.0,'Val_Split_Position':'end'})

# Generate Sensitivity data from realizations
sen_fdata,sen_ldata,_,_,_,_=split_df(gr_domain, split_frac={'Train':0.99,'Val':0.,'Samp_File':[False,None],'Full_Samp_Val_Test':True}, f_index={'Start':0,'End':9}, l_index={'Start':9,'End':12},\
             samp_par={'Tstep_step':int(tstep_samp/tstep),'Ntsteps':(ntsteps+1),'Ntsteps_Frac':data_frac,'Nrealz':nrealz,'Pert_List':ext_sen_realz,'Sen_List':ext_sen_realz,'Tscale':'Normal','Log_Step':4.0,'Val_Split_Position':'end'})

#======================================================== Dataset Normalization ==================================================
# Normalization Function
# Normalization is done using the training data statistics both on training and validation set
# Normalization is mostly performed on the features; labels are usually not normalized in a 
# regression problem except one is considering more than one dimension of labels or in some cases of classification
def scaling_fun(x,train_dtset,*rngs,itype='features',normalization_method='z-score',lin_scale={'Min':-1,'Max':1}):
    if len(x.columns)!=len(train_dtset.columns) :
        return print('Check Train_Test Data','\n','Check..........1','\n','Check..........2'\
                      ,'\n','Check..........3')
    else:
        try: 
            if normalization_method=='z-score':
                if itype=='features':
                    train_dtset_stats=train_dtset.iloc[:,rngs[0]:rngs[1]].describe().transpose()
                    norm=(x.iloc[:,rngs[0]:rngs[1]]-train_dtset_stats['mean'])/train_dtset_stats['std']
                    diff_=(x.iloc[:,(rngs[1]):])

                    return pd.concat([norm.fillna(0),diff_],axis=1)
                else:
                    train_dtset_stats=train_dtset.describe().transpose()
                    norm=(x-train_dtset_stats['mean'])/train_dtset_stats['std']
                    return norm.fillna(0)
            elif normalization_method=='linear-scaling':
                # Use a scaling of a=-1 to b=1; x(scale)=(b-a)*((x-xmin)/(xmax-xmin))+a
                a=lin_scale['Min']
                b=lin_scale['Max']
                if itype=='features':
                    # Log min-min scaling is used for the permeability to prevent outliers
                    dtset_key=list(train_dtset.keys())
                    train_dtset_stats=train_dtset.iloc[:,rngs[0]:rngs[1]].describe().transpose()
                    xmin=train_dtset_stats['min']; xmax=train_dtset_stats['max']
                    norm=(b-a)*((x.iloc[:,rngs[0]:rngs[1]]-xmin)/(xmax-xmin))+a
                    diff_=(x.iloc[:,(rngs[1]):])
                    
                    return pd.concat([norm.fillna(0),diff_],axis=1)
                else:
                    train_dtset_stats=train_dtset.describe().transpose()
                    xmin=train_dtset_stats['min']; xmax=train_dtset_stats['max']
                    norm=(b-a)*((x-xmin)/(xmax-xmin))+a
                    return norm.fillna(0)
            elif normalization_method=='lnk-linear-scaling':
                # Use a scaling of a=-1 to b=1; x(scale)=(b-a)*((x-xmin)/(xmax-xmin))+a
                a=lin_scale['Min']
                b=lin_scale['Max']
                if itype=='features':
                    # Log min-min scaling is used for the permeability to prevent outliers
                    dtset_key=list(train_dtset.keys())
                    train_dtset_stats_k=train_dtset[['permx','permz']].describe().transpose()
                    train_dtset_stats=train_dtset.iloc[:,rngs[0]:rngs[1]].describe().transpose().drop(['permx','permz'])
                    xmin=train_dtset_stats['min']; xmax=train_dtset_stats['max']
                    xmin_k=train_dtset_stats_k['min']; xmax_k=train_dtset_stats_k['max']
                    norm=(b-a)*((x.iloc[:,rngs[0]:rngs[1]].drop(columns=['permx','permz'])-xmin)/(xmax-xmin))+a
                    norm_k=(b-a)*((np.log(x[['permx','permz']]/xmin_k))/np.log((xmax_k/xmin_k)))+a
                    diff_=(x.iloc[:,(rngs[1]):])
                    # Concatenate the individual data frames and reorder the columns
                    return pd.concat([norm.fillna(0),norm_k.fillna(0),diff_],axis=1)[dtset_key]
                else:
                    train_dtset_stats=train_dtset.describe().transpose()
                    xmin=train_dtset_stats['min']; xmax=train_dtset_stats['max']
                    norm=(b-a)*((x-xmin)/(xmax-xmin))+a
                    return norm.fillna(0)                         
            else: 
                return x
        except pd.errors.EmptyDataError:
            return

# Apply the normalization function
norm_train_fdata=scaling_fun(train_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_val_fdata=scaling_fun(val_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_test_fdata=scaling_fun(test_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_sen_fdata=scaling_fun(sen_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 

norm_train_ldata=scaling_fun(train_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_val_ldata=scaling_fun(val_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_test_ldata=scaling_fun(test_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_sen_ldata=scaling_fun(sen_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 

# Convert the normalized data to a list of arrays--inputs for the developed ANN models
train_fdata_list=[norm_train_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_train_fdata.columns)-2)]+[(norm_train_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
val_fdata_list=[norm_val_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_val_fdata.columns)-2)]+[(norm_val_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
test_fdata_list=[norm_test_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_test_fdata.columns)-2)]+[(norm_test_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
sen_fdata_list=[norm_sen_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_sen_fdata.columns)-2)]+[(norm_sen_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]

train_ldata_list=[norm_train_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_train_ldata.columns))]
val_ldata_list=[norm_val_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_val_ldata.columns))]
test_ldata_list=[norm_test_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_test_ldata.columns))]
sen_ldata_list=[norm_sen_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_sen_ldata.columns))]

# =================================================== Normalized Dataset Save ====================================================
# Save function 
def update_dumps(save_df=False,save_PVT=True):
    from os import path
    save_folder_fullpath = (f'{save_folder_path}{dim_model}_{grid_dim["Number"][0]}x{grid_dim["Number"][1]}x{grid_dim["Number"][2]}_{fluid_type.upper()}[{fluid_comp.upper()}_{static_prop_type.upper()}]'
                            f'_K[{kfunc.upper()[:2]}_{m}_{sd}]_AT[{arr_type}_{numeric_type.upper()[:3]}]_SP[{train_realz_start}_{train_realz_step}_{sen_realz_start}_{sen_realz_step}_{nrealz}_{sen_samples}]_TF[{train_frac}_{tstep_samp}D_{data_frac}E]'
                            f'_IO[{str(input_norm).upper()[0:2]}_{str(output_norm).upper()[0:2]}]_[{w_info["nwells"]}W_R{int(np.sum(init_rate["gas"]))}_P{int(np.mean(min_bhp))}]')
    if not path.exists(save_folder_fullpath):
        os.mkdir(save_folder_fullpath)
        
    dump(train_fdata_list, open(f'{save_folder_fullpath}/train_fdata_list','wb'))
    dump(val_fdata_list, open(f'{save_folder_fullpath}/val_fdata_list','wb'))
    dump(test_fdata_list, open(f'{save_folder_fullpath}/test_fdata_list','wb'))
    dump(sen_fdata_list, open(f'{save_folder_fullpath}/sen_fdata_list','wb'))
    
    dump(train_ldata_list, open(f'{save_folder_fullpath}/train_ldata_list','wb'))
    dump(val_ldata_list, open(f'{save_folder_fullpath}/val_ldata_list','wb'))
    dump(test_ldata_list, open(f'{save_folder_fullpath}/test_ldata_list','wb'))
    dump(sen_ldata_list, open(f'{save_folder_fullpath}/sen_ldata_list','wb'))

    dump(train_fdata, open(f'{save_folder_fullpath}/train_fdata','wb')) 
    dump(train_ldata, open(f'{save_folder_fullpath}/train_ldata','wb'))
    if save_df:
        dump(norm_train_fdata, open(f'{save_folder_fullpath}/train_fdata_df','wb')) 
        dump(norm_val_fdata, open(f'{save_folder_fullpath}/val_fdata_df','wb'))
        dump(norm_test_fdata, open(f'{save_folder_fullpath}/test_fdata_df','wb'))
        dump(norm_sen_fdata, open(f'{save_folder_fullpath}/sen_fdata_df','wb'))
        
        dump(norm_train_fdata, open(f'{save_folder_fullpath}/train_fdata_df','wb')) 
        dump(norm_val_ldata, open(f'{save_folder_fullpath}/val_ldata_df','wb'))
        dump(norm_test_ldata, open(f'{save_folder_fullpath}/test_ldata_df','wb'))
        dump(norm_sen_ldata, open(f'{save_folder_fullpath}/sen_ldata_df','wb'))   
    if save_PVT:
        open_folder_fullpath_PVT = f'{save_folder_path}{dim_model}_PVTMODEL_{fluid_type.upper()}_AT[3]'
        dump(pvt.dump_PVT(folder_path=open_folder_fullpath_PVT,fluid_type=fluid_type,dt_type=dt_type), open(f'{save_folder_fullpath}/PVT_list','wb'))     
    return

# Dump a DataFrame of the normalized training, validation and test datasets
if __name__ == '__main__': 
    update_dumps()    
    
end_time=t.time()

# Checking the total time spent for the pretraining operation
print('Data sorting took: {:.4F}'.format(end_time-start_time))
# Load the dumped files