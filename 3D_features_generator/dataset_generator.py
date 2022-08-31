"""
Creates a 2D or 3D dataset features used for training the AI-based SRM
The permeability field can be generated using a uniform distribution or 
using an included KLE module (@Ilias Bilionis 2014), which has been adapted to 
the work.

The first section includes the main configuration settings for the AI-based SRM

Created on Wed Jun 23 16:46:05 2021

@author: Victor Molokwu
Heriot-Watt University
"""

import pandas as pd
import re
import os
import numpy as np
import time as t
from scipy.stats import lognorm
import kle

# Set a datatype to use where necessary
dt_type=np.float32
start_time=t.time()
#================================================Configuration Settings=================================================
# Sort the data into Training, Validation and Test Sets; Normalize Data
dim_model='2D'
arr_type='1'
training_type='nonphysics'
fluid_type='dry_gas'
fluid_comp='mult_comp'
static_prop_type='uni'                                      # 'rand'|'uni'numeric_type='implicit'

# Realization settings:
grid_shape=(29,29,1)                                        # Grid dimension
well_idx=[(14,14,0)]                                        # Well connections in the grid -- a list of connection tuples
di=100*np.ones(grid_shape)                                  # Grid block size x-axis
dj=100*np.ones(grid_shape)                                  # Grid block size y-axis
dk=80*np.ones(grid_shape)                                   # Grid block size z-axis

tstep=1.                                                    # Timestep in which the permeability fields are generated
ntsteps=int(540/1)                                          # Note: This excludes the initial timestep t==0
kfunc='lognormal'                                           # Permeability generator function--'linear'|'power'|'lognormal'

nrealz=12  #13 50 30 30 30 20 50
m=2.5                                                       # The mean of the  distribution
sd=1                                                        # The standard deviation of the total distributioin  
phi=0.2*np.ones(grid_shape)
ani=0.1
qg_init=5000   

data_frac=1.0                                               # Fraction of raw data used for surrogate modelling (timestep is used as an indicator)
tstep_samp=10.                                              # Timestep interval at which the data is sampled
train_frac=0.5                                              # Fraction of the data used for training-testing
FVTDS_bool=True                                             # Full Val_Test Data Sampling (FVTDS)-- portion of the grid domain used for validation and testing 
input_norm='lnk-linear-scaling'                             # linear-scaling, z-score, lnk-linear-scaling (ln for the permeability feature while other inputs are linearly scaled)
output_norm=None
lscale={'Min':-1.,'Max':1}
rng=[0,7,12]                                                # Feature data columns thats to be normalized
#===============================================Grid Dimensions=================================================
x=np.linspace(50,2850,grid_shape[0])
y=np.linspace(50,2850,grid_shape[1])
z=np.linspace(11040,11040,grid_shape[2])
x_mgrid,y_mgrid,z_mgrid=np.meshgrid(x,y,z)
#=============================================Realization Settings==============================================
sigma=np.sqrt(np.log(((sd/m)**2)+1))                        # The standard deviation of the unique normal distribution of the log variables
mu=np.log(m)-0.5*sigma**2                                   # The mean of the unique normal distribution of the log variable 
sen_realz_start=int(0); sen_realz_step=int(10)
train_realz_start=int(1); train_realz_step=int(1)

if static_prop_type=='rand':
    sen_realz_end=nrealz
    train_realz_end=nrealz
    sen_realz=list(range(sen_realz_start,sen_realz_end,sen_realz_step))
else:
    k=lambda x:1.16**(x)                                    # Uniform field permeability function used to generate the train and validation data
    ks=lambda x:1.15**(x)                                   # Uniform field permeability function used to generate the test dataset
    if kfunc=='power' :
        sen_realz_end=int(np.log(k(nrealz-1))/np.log(ks(1)))+1
    elif kfunc=='lognormal':
        sen_realz_end=int(np.log(lognorm.ppf(0.99,sigma,loc=0.,scale=np.exp(mu)))/np.log(ks(1)))+1
    sen_realz=list(range(sen_realz_start,sen_realz_end,sen_realz_step))
    train_realz_end=nrealz+len(sen_realz)
   
train_realz=list(range(train_realz_start,train_realz_end,train_realz_step))  
train_realz=sorted(list(set(train_realz)-set(sen_realz)))      

if len(train_realz)%2!=0:
    import sys
    print('Train realization not even, might experience problem creating batch sizes to run in graph mode')
    sys.exit()
#================================================Static Properties==============================================
if static_prop_type=='rand':
    kx,kx_sen=kle.KLE_samples(no_samples=nrealz,rseed=50,mu=m,sd=sd,grid_dim=grid_shape,gridblock_size=(100,100,80),corr_len_fac=0.2,energy_fac=0.98,train_test_samp=sen_realz_step,write_out=True)
else:
    if kfunc=='power':
        kx=[k(i)*np.ones(grid_shape) for i in range(len(train_realz))]
    elif kfunc=='lognormal':
        kx=[lognorm.ppf(i,sigma,loc=0.,scale=np.exp(mu))*np.ones(grid_shape) for i in np.linspace(0.015,0.99,len(train_realz))]
    else:
        max_k=k(nrealz-1)
        min_k=k(1-1)
        no_k=len(train_realz)
        diff=(max_k-1)/(no_k-1)
        k_lin=lambda x:min_k+(x*diff)
        kx=[k_lin(i)*np.ones(grid_shape) for i in range(len(train_realz))]
        # Update the permeability values with simulation-based func. validation data at index points list: sen_realz
    
    kx_sen=[ks(i)*np.ones(grid_shape) for i in sen_realz]

kz=[ani*kx[i] for i in range(len(kx))]
kz_sen=[ani*kx_sen[i] for i in range(len(kx_sen))] 

for i in sen_realz:
    kx.insert(i,kx_sen[sen_realz.index(i)])
    kz.insert(i,kz_sen[sen_realz.index(i)])

#==========================================Recurrent Properties (Not Really Needed!)===========================
qg=[[[0.]+[qg_init+(i-1)*0. for i in range(int(ntsteps))] for j in range(len(train_realz)+len(sen_realz))],]              # Outer -- Inner: No Wells ==> Realizations ==> Timesteps

# Pressure and saturation obtained from simulation restart file: To compare the accuracy of the PINN results
# Load the pressure restart files (in same folder)--restart file obtained from a commercial resevoir simulator: Eclipse (TM)

sim_file=os.path.join(os.getcwd(),'gr_dynamic')
if os.path.isfile(sim_file):
    from pickle import load
    gr_dynamic=load(os.path.join(os.getcwd(),'gr_dynamic'),'rb')
    
    # Create a dictionary of pressure and saturation outputs
    pre_sat={}
    for i in sen_realz:
        dict_list=gr_dynamic[(gr_dynamic['perturbation']==i)][['pressure','sgas']].to_dict(orient='list')
        
        # Reshape the dictionary list as array
        for key,value in dict_list.items():
            dict_list[key]=np.reshape(np.array(value),(-1,*grid_shape))
    
        # Add to the parent list
        pre_sat[str(i)]=dict_list

#=====================================================================================================================
t0=np.zeros(grid_shape)
lbl=np.zeros(grid_shape+(2,))
coord_x=[]
coord_y=[]
coord_z=[]
time=[]
poro=[]
permx=[]
permz=[]
dx=[]
dy=[]
dz=[]
label=[]
timestep=[]
perturb=[]

# Outputs
pre=[]
gsat=[]
grate=[]
gcum=[]

for i in range(len(train_realz)+len(sen_realz)):
    coord_x_realz=[]
    coord_y_realz=[]
    coord_z_realz=[]
    time_realz=[]
    poro_realz=[]
    permx_realz=[]
    permz_realz=[]
    dx_realz=[]
    dy_realz=[]
    dz_realz=[]
    label_realz=[]
    timestep_realz=[]
    perturb_realz=[]
    
    # Outputs
    pre_realz=[]
    gsat_realz=[]
    grate_realz=[]
    gcum_realz=[]
    
    # set the label for the realization
    if i in sen_realz:
        lbl[...,1]=1.
    else:
        lbl[...,1]=0.
    
    for j in range(int(ntsteps+1)):
        coord_x_realz.append(x_mgrid)
        coord_y_realz.append(y_mgrid)
        coord_z_realz.append(z_mgrid)
        time_realz.append([t0+(j)*tstep])
        poro_realz.append(phi)
        permx_realz.append(kx[i])
        permz_realz.append(kz[i])
        dx_realz.append(di)
        dy_realz.append(dj)
        dz_realz.append(dk)
        label_realz.append(lbl)
        timestep_realz.append(j*np.ones(grid_shape))
        perturb_realz.append(i*np.ones(grid_shape))
        
        # Create a grid rate and cumulative -using the implicit formulation
        qi=np.zeros(grid_shape)
     
        if j<int(ntsteps):
            for k in range(len(well_idx)):
                qi[well_idx[k]]=qg[k][i][j+1]
            
            grate_realz.append(qi)
            # Calculate the incremental volume using a square
            dv=qi*tstep   
            if j==0:
                gcum_realz.append(dv)
            else:
                gcum_realz.append(gcum_realz[j-1]+dv)
        else:
            grate_realz.append(qi)
            dv=np.zeros(grid_shape)
            gcum_realz.append(gcum_realz[j-1]+dv)

       
        # Pressure can be updated
        if i in sen_realz and os.path.isfile(sim_file):
            pre_realz.append(pre_sat[str(i)]['pressure'][j,...])
            gsat_realz.append(pre_sat[str(i)]['sgas'][j,...])
        else:
            pre_realz.append(np.zeros(grid_shape))
            gsat_realz.append(np.zeros(grid_shape))

    # Append to the main list
    coord_x.append(coord_x_realz)
    coord_y.append(coord_y_realz)
    coord_z.append(coord_z_realz)
    time.append(time_realz)
    poro.append(poro_realz)
    permx.append(permx_realz)
    permz.append(permz_realz)
    dx.append(dx_realz)
    dy.append(dy_realz)
    dz.append(dz_realz)
    label.append(label_realz)
    timestep.append(timestep_realz)
    perturb.append(perturb_realz)
    
    # Outputs
    pre.append(pre_realz)
    gsat.append(gsat_realz)
    grate.append(grate_realz)
    gcum.append(gcum_realz)


# Reshape to 1D
coord_x=np.reshape(coord_x,(-1))
coord_y=np.reshape(coord_y,(-1))
coord_z=np.reshape(coord_z,(-1))
time=np.reshape(time,(-1))
poro=np.reshape(poro,(-1))
permx=np.reshape(permx,(-1))
permz=np.reshape(permz,(-1))
dx=np.reshape(dx,(-1))
dy=np.reshape(dy,(-1))
dz=np.reshape(dz,(-1))
label=np.reshape(label,(-1,2))
timestep=np.reshape(timestep,(-1))
perturb=np.reshape(perturb,(-1))

# Outputs
pre=np.reshape(pre,(-1))
gsat=np.reshape(gsat,(-1))
grate=np.reshape(grate,(-1))
gcum=np.reshape(gcum,(-1))       
#End of File       

df_data={'x_coord':coord_x,'y_coord':coord_y,'z_coord':coord_z,'time':time,'poro':poro,'permx':permx,'permz':permz,\
         'dx':dx,'dy':dy,'dz':dz,'Label_1':label[:,0],'Label_2':label[:,1],'pressure':pre,'gsat':gsat,'grate':grate,'cum_gprod':gcum,'timestep':timestep,'perturbation':perturb}

gr_domain=pd.DataFrame(data=df_data)

# Set datatype
dtype={'x_coord':'float32','y_coord':'float32','z_coord':'float32','time':'float32','poro':'float32','permx':'float32','permz':'float32',\
         'dx':'float32','dy':'float32','dz':'float32','Label_1':'float32','Label_2':'float32','pressure':'float32','gsat':'float32',\
         'grate':'float32','cum_gprod':'float32','timestep':'int32','perturbation':'int32'}

from pickle import dump        
def update_dumps():
    dump(gr_domain.astype(dtype), open(os.path.join(os.getcwd(),'learning_dataset'),'wb'))

update_dumps()
end_time=t.time()

# Load the dumped files
'''
from pickle import load
gr_domain=load(open(os.path.join(os.getcwd(),'learning_dataset'),'rb'))
'''

# Create a splitting function

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
    
    if split_frac['Train']+split_frac['Val']>=1:
        print ('Check split fraction')
        return
    if split_frac['Val']==None:
        split_frac['Val']=1-(split_frac['Train'])
    
    # Create a log-range array for the DOM time steps
    max_tstep=np.int32(samp_par['Ntsteps']*samp_par['Ntsteps_Frac'])
    
    if samp_par['Tscale']=='log':
        tstep_range=np.unique(np.logspace(0,np.log10(max_tstep+1),num=np.int32(max_tstep/samp_par['Log_Step'])).astype(np.int32))-1     # -1 is used to return to zero-based indexing
    else:
        tstep_range=np.array(range(0,max_tstep,int(samp_par['Tstep_step'])))
    

    no_samp_tsteps=len(tstep_range)
    train_val_maxtstep_idx=np.int32(np.ceil((split_frac['Train']+split_frac['Val'])*no_samp_tsteps))
        
    tstep_range_train_val=tstep_range[0:train_val_maxtstep_idx]
    
    val_interval=1/((split_frac['Val'])+1e-16)
    no_val_idx=np.int32(split_frac['Val']*len(tstep_range_train_val))
    
    if samp_par['Val_Split_Position']=='inbetween':
        tstep_range_val=[]
        for i in range(no_val_idx):
            dropval_idx=np.int32(val_interval*(1+i))
            if dropval_idx<len(tstep_range_train_val):
                tstep_range_val.append(tstep_range_train_val[dropval_idx])
    elif samp_par['Val_Split_Position']=='end' or samp_par['Val_Split_Position']==None:
        tstep_range_val=tstep_range_train_val[-1-no_val_idx:] 
    
    tstep_range_train=list(set(tstep_range_train_val)-set(tstep_range_val))
    tstep_range_test=tstep_range[train_val_maxtstep_idx:no_samp_tsteps]

    # Check if the train tstep is odd. Even is desirable to allow stacking in graph mode. Any extra timestep is added to the validation data
    if len(tstep_range_train)%2!=0:
        # Remove the max training timestep
        max_tstep_train=max(tstep_range_train)
        tstep_range_train.remove(max_tstep_train)
        
        # Add the max training timestep to the validation timestep
        tstep_range_val=np.insert(tstep_range_val,0,max_tstep_train)
    
    
    # Add some extra datapoints at early time -- number corresponds to the timestep
    if len(tstep_range_train)!=0:
        tstep_range_train=np.unique(list(range(int(samp_par['Tstep_step'])-1))+(tstep_range_train))
        print('No of Train Tsteps:',len(tstep_range_train), '| Max Train Time:',np.max(tstep_range_train)) 
    else:
        print('No of Train Tsteps:',len(tstep_range_train), '| Max Train Time:',0.) 

    # Get the absolute time (series-like data type without duplicates) of the timestep range from either the domain or boundary data
    time_range_train=dataset[dataset['timestep'].isin(tstep_range_train)]['time'].drop_duplicates()
    time_range_val=dataset[dataset['timestep'].isin(tstep_range_val)]['time'].drop_duplicates()
    time_range_test=dataset[dataset['timestep'].isin(tstep_range_test)]['time'].drop_duplicates()
    

    train_sen_idx_list=list(set(samp_par['Pert_List']).union(set(samp_par['Sen_List'])))
    for j in train_sen_idx_list:
        for i in np.concatenate([tstep_range_train,tstep_range_val,tstep_range_test]):
            dataset_tstep=dataset.loc[(dataset['timestep']==i) & (dataset['perturbation']==j)]     #Due to overload, use bitwise operators |,&
            fdataset_tstep=dataset_tstep.iloc[:,f_index['Start']:f_index['End']]
            ldataset_tstep=dataset_tstep.iloc[:,l_index['Start']:l_index['End']]
            #=====================================================================================================================================
            if i==0 and j==train_sen_idx_list[0]:
                # Initialize DataFrame object for each variable
                # Features
                train_fdata=pd.DataFrame(); val_fdata=pd.DataFrame(); test_fdata=pd.DataFrame()

                # Labels
                train_ldata=pd.DataFrame(); val_ldata=pd.DataFrame(); test_ldata=pd.DataFrame()
           
            if i in tstep_range_train and j in samp_par['Pert_List']:
                # Split the Dataset within a timestep
                #=======================DOMAIN CONDITION PDE SOLUTION dataset_tstep=============
                # Create the feature and labels of each solution dataset with previous timesteps.  Info: Concatenate the DataFrame to index 0--concatenate is more efficient that append
                train_fdata=pd.concat([train_fdata, fdataset_tstep], ignore_index=False)
                train_ldata=pd.concat([train_ldata, ldataset_tstep], ignore_index=False)
            if i in tstep_range_val and j in samp_par['Sen_List']:
                #=====================DOMAIN VALIDATION dataset_tstep============================================================
                val_fdata=pd.concat([val_fdata, fdataset_tstep], ignore_index=False)
                val_ldata=pd.concat([val_ldata, ldataset_tstep], ignore_index=False)
                
            if i in tstep_range_test and j in samp_par['Sen_List']: 
                #=======================DOMAIN TEST dataset_tstep=============================================
                test_fdata=pd.concat([test_fdata, fdataset_tstep], ignore_index=False)
                test_ldata=pd.concat([test_ldata, ldataset_tstep], ignore_index=False)

    return train_fdata, train_ldata, val_fdata, val_ldata, test_fdata, test_ldata

# Generate the data
train_fdata,train_ldata,val_fdata,val_ldata,test_fdata,test_ldata=split_df(gr_domain, split_frac={'Train':train_frac,'Val':0.1,'Samp_File':[False,None],'Full_Samp_Val_Test':True}, f_index={'Start':0,'End':12}, l_index={'Start':12,'End':16},\
            samp_par={'Tstep_step':int(tstep_samp/tstep),'Ntsteps':(ntsteps+1),'Ntsteps_Frac':data_frac,'Nrealz':nrealz,'Pert_List':train_realz,'Sen_List':sen_realz,'Tscale':'Normal','Log_Step':4.0,'Val_Split_Position':'end'})

# Generate Sensitivity data from realizations
sen_fdata,sen_ldata,_,_,_,_=split_df(gr_domain, split_frac={'Train':0.99,'Val':0.,'Samp_File':[False,None],'Full_Samp_Val_Test':True}, f_index={'Start':0,'End':12}, l_index={'Start':12,'End':16},\
             samp_par={'Tstep_step':int(tstep_samp/tstep),'Ntsteps':(ntsteps+1),'Ntsteps_Frac':data_frac,'Nrealz':nrealz,'Pert_List':sen_realz,'Sen_List':sen_realz,'Tscale':'Normal','Log_Step':4.0,'Val_Split_Position':'end'})

#=====================================================================================================================
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
#=====================================================================================================================
# Normalize the dataset
norm_train_fdata=scaling_fun(train_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_val_fdata=scaling_fun(val_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_test_fdata=scaling_fun(test_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 
norm_sen_fdata=scaling_fun(sen_fdata, train_fdata,rng[0],rng[1],itype='features',normalization_method=input_norm,lin_scale=lscale) 

norm_train_ldata=scaling_fun(train_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_val_ldata=scaling_fun(val_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_test_ldata=scaling_fun(test_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 
norm_sen_ldata=scaling_fun(sen_ldata, train_ldata,rng[0],rng[1],itype='labels',normalization_method=output_norm,lin_scale=lscale) 

# Convert the normalized data to a list of arrays which is fed to the PINN
train_fdata_list=[norm_train_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_train_fdata.columns)-2)]+[(norm_train_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
val_fdata_list=[norm_val_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_val_fdata.columns)-2)]+[(norm_val_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
test_fdata_list=[norm_test_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_test_fdata.columns)-2)]+[(norm_test_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]
sen_fdata_list=[norm_sen_fdata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_sen_fdata.columns)-2)]+[(norm_sen_fdata.iloc[:,-2:].to_numpy(dtype=dt_type))]

train_ldata_list=[norm_train_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_train_ldata.columns))]
val_ldata_list=[norm_val_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_val_ldata.columns))]
test_ldata_list=[norm_test_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_test_ldata.columns))]
sen_ldata_list=[norm_sen_ldata.iloc[:,i].to_numpy(dtype=dt_type) for i in range(len(norm_sen_ldata.columns))]

# Dump the training, validation and test data as lists of tensors
def update_dumps(save_df=False):
    from os import path
    save_folder_fullpath = f'{dim_model}_{fluid_type.replace("-","").upper()}[{fluid_comp.replace("-","").upper()}_{static_prop_type.replace("-","").upper()}]_K[{kfunc.upper()[:2]}_{m}_{sd}]_ART[{arr_type}]_SPI[{train_realz_start}_{train_realz_step}_{sen_realz_start}_{sen_realz_step}_{nrealz}]_TRF[{train_frac}_{tstep_samp}D_{data_frac}E]_FVTDS[{str(FVTDS_bool)[0]}]_INPN[{str(input_norm).upper()[0:2]}]_OUTN[{str(output_norm).upper()[0:2]}]'
    # Check if folder exists
    import shutil
    if os.path.exists(save_folder_fullpath):
        shutil.rmtree(save_folder_fullpath)
        os.mkdir(save_folder_fullpath)
    else:
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
    
    # Optional: To dump the training, validation and test data as dataframes
    if save_df:
        dump(norm_train_fdata, open(f'{save_folder_fullpath}/train_fdata_df','wb')) 
        dump(norm_val_fdata, open(f'{save_folder_fullpath}/val_fdata_df','wb'))
        dump(norm_test_fdata, open(f'{save_folder_fullpath}/test_fdata_df','wb'))
        dump(norm_sen_fdata, open(f'{save_folder_fullpath}/sen_fdata_df','wb'))
        
        dump(norm_train_fdata, open(f'{save_folder_fullpath}/train_fdata_df','wb')) 
        dump(norm_val_ldata, open(f'{save_folder_fullpath}/val_ldata_df','wb'))
        dump(norm_test_ldata, open(f'{save_folder_fullpath}/test_ldata_df','wb'))
        dump(norm_sen_ldata, open(f'{save_folder_fullpath}/sen_ldata_df','wb'))        

    return

# Dumps the training, validation and test dataset
if __name__ == '__main__': 
    update_dumps()    
print('Data sorting took: {:.4F}'.format(end_time-start_time))    
