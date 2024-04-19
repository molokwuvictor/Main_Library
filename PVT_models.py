# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 00:28:12 2023

@author: vcm1
"""

import tensorflow as tf
import tensorflow_addons as tfa
from scipy.interpolate import CubicSpline
import pandas as pd

def dump_PVT(folder_path=None,fluid_type=None,dt_type=tf.float32):
    # Dump the PVT data for intepolation spline
    # Read the PVT input sheet from an excel file -- viscosity and gas compressibility factor
    pvt_data=pd.read_excel(f'{folder_path}/PVT Properties.xlsx', sheet_name=f'Fluid_Properties_{fluid_type.upper()}',engine='openpyxl',dtype=dt_type).iloc[:-1]
    
    def update_df(df=None,column_name=None,max_value=1.):
        idx_1=df.loc[(df[column_name]>=max_value)].index
        upd_val=df.loc[idx_1+1,column_name].reset_index()+np.abs(df.loc[idx_1+1,column_name].reset_index()-df.loc[idx_1+2,column_name].reset_index())
        upd_val.index=idx_1
        df.loc[(df[column_name]>=max_value),column_name]=np.array(upd_val[column_name])
        return
    
    #update_df(df=pvt_data,column_name='Gas_OGR_(stb/Mscf)')
    # Training statistics for the oil and gas data
    end_idx=9
    ts_col=['min', 'max', 'mean', 'std', 'count']
    ts_idx=list(pvt_data.iloc[:,:end_idx].describe().transpose().index[:end_idx])
    ts_tf=tf.convert_to_tensor(pvt_data.iloc[:,:end_idx].describe().transpose()[ts_col])
    ts=[ts_tf,ts_idx,ts_col]
    
    # Fill empty data cells after taking the training statistics
    pvt_data=pvt_data.fillna(0.)
    #po=tf.constant(pvt_data['PSAT_(psia)'],name='oil_pressure')  #Saturation pressure
   
    temp=tf.constant(pvt_data['Temperature_(degF)'],name='Temperature')
    pg=tf.constant(pvt_data['Gas_Pressure_(psia)'],name='gas_pressure')  #Saturation pressure
    input_shape=(1,len(pg),1)
    
    Bgd=tf.constant(pvt_data['Gas_FVF_Dry_(rb/Mscf)'],name='gas_FVF')
    ugd=tf.constant(pvt_data['Gas_Visc_Dry_(cp)'],name='gas_visc')
    dinvBgd=tf.constant(pvt_data['d_Inv_Gas_FVF_Dry_(1/rb/Mscf)'],name='d_inv_gas_FVF_dry')
    dinvugd=tf.constant(pvt_data['d_Inv_Gas_Visc_Dry_(1/cp)'],name='d_inv_gas_visc_dry')
    

    if fluid_type in ['GC','gas_cond']:
        Bgw=tf.constant(pvt_data['Gas_FVF_Wet_(rb/Mscf)'],name='gas_FVF')
        ugw=tf.constant(pvt_data['Gas_Visc_Wet_(cp)'],name='gas_visc')
        Bo=tf.constant(pvt_data['Oil_FVF_(rb/stb)'],name='oil_FVF')
        uo=tf.constant(pvt_data['Oil_Visc_(cp)'],name='oil_visc')
        Rs=tf.constant(pvt_data['Oil_GOR_(Mscf/stb)'],name='oil_GOR')
        Rv=tf.constant(pvt_data['Gas_OGR_(stb/Mscf)'],name='gas_OGR')
    
        invBgw=tf.math.divide_no_nan(1.,Bgw)
        invugw=tf.math.divide_no_nan(1.,ugw)
        invBo=tf.math.divide_no_nan(1.,Bo)
        invuo=tf.math.divide_no_nan(1.,uo)
        invBgd=tf.math.divide_no_nan(1.,Bgd)
        invugd=tf.math.divide_no_nan(1.,ugd)
    
        dinvBgw=tf.constant(pvt_data['d_Inv_Gas_FVF_Wet_(1/rb/Mscf)'],name='d_inv_gas_FVF_wet')
        dinvBo=tf.constant(pvt_data['d_Inv_Oil_FVF_(1/rb/Stb)'],name='d_inv_oil_FVF')
        dinvugw=tf.constant(pvt_data['d_Inv_Gas_Visc_Wet_(1/cp)'],name='d_inv_gas_visc_wet')
        dinvuo=tf.constant(pvt_data['d_Inv_Oil_Visc_(1/cp)'],name='d_inv_oil_visc')
        dRs=tf.constant(pvt_data['d_Oil_GOR_(Mscf/stb)'],name='d_oil_GOR')
        dRv=tf.constant(pvt_data['d_Gas_OGR_(stb/Mscf)'],name='d_gas_OGR')
        
        vro=tf.constant(pvt_data['Vro_CCE'],name='vro_CCE')
    
    #Dumped as a Dictionary of arrays
    # Pressure | Bgw | Bo | ugw | uo | GOR | OGR | Bg | ug
    
    #Reshape
    pg=tf.reshape(pg,input_shape)
    invBgd=tf.reshape(tf.math.divide_no_nan(1.,Bgd),input_shape)
    invugd=tf.reshape(tf.math.divide_no_nan(1.,ugd),input_shape)
    dinvBgd=tf.reshape(dinvBgd,input_shape)
    dinvugd=tf.reshape(dinvugd,input_shape)
    
    # NOTE: The wet gas formulation in PVTi (i.e., PVTO is given as the volume of wet gas at reservoir/volume of gas at std (excluding the condensate))
    if fluid_type in ['GC','gas_cond']:
        invBgw=tf.reshape(tf.math.divide_no_nan(1.,Bgw),input_shape)
        invugw=tf.reshape(tf.math.divide_no_nan(1.,ugw),input_shape)
        invBo=tf.reshape(tf.math.divide_no_nan(1.,Bo),input_shape)
        invuo=tf.reshape(tf.math.divide_no_nan(1.,uo),input_shape)
        Rs=tf.reshape(Rs,input_shape)
        Rv=tf.reshape(Rv,input_shape)

        dinvBgw=tf.reshape(dinvBgw,input_shape)
        dinvugw=tf.reshape(dinvugw,input_shape)
        dinvBo=tf.reshape(dinvBo,input_shape)
        dinvuo=tf.reshape(dinvuo,input_shape)  
        dRs=tf.reshape(dRs,input_shape)
        dRv=tf.reshape(dRv,input_shape) 
        
        vro=tf.reshape(vro,input_shape) 
    
    out={'Pre':pg,'InvBg':invBgd,'Invug':invugd,'dInvBg':dinvBgd,'dInvug':dinvugd}
    if fluid_type in ['GC','gas_cond']:
        # out={'Pre':pg,'InvBg':invBgd,'InvBo':invBo,'Invug':invugd,'Invuo':invuo,'Rs':Rs,'Rv':Rv,'InvBgd':invBgd,'Invugd':invugd,\
        #         'dInvBg':dinvBgd,'dInvBo':dinvBo,'dInvug':dinvugd,'dInvuo':dinvuo,'dRs':dRs,'dRv':dRv,'dInvBgd':dinvBgd,'dInvugd':dinvugd}
        d_out={'dInvBg':dinvBgw,'dInvBo':dinvBo,'dInvug':dinvugw,'dInvuo':dinvuo,'dRs':dRs,'dRv':dRv,'dInvBgd':dinvBgd,'dInvugd':dinvugd}
        out={'Pre':pg,'InvBg':invBgw,'InvBo':invBo,'Invug':invugw,'Invuo':invuo,'Rs':Rs,'Rv':Rv,'InvBgd':invBgd,'Invugd':invugd,'Vro':vro}

    return out

import tensorflow as tf
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
    