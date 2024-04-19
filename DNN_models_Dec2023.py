# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:51:35 2021

@author: User
"""
import os
import batch_loss as bl
import tensorflow_probability as tfp
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disable GPU: -1

import tensorflow as tf
import numpy as np
import DNN_models_1 as dnn1

dt_type='float64'

def plain_modular_model(cfd_type=None,inputs=None,width={'ps':{'Bottom_Size':None,'No_Gens':None,'Growth_Type':None,'Growth_Rate':None},'ko':{},'Bu':{}},depth={'ps':None,'ko':None,'Bu':None},\
                        config={'ps':None,'ko':None,'Bu':None,'qs':None},kernel={'Init':None,'Regu':None},hlayer_activation_func={'ps':None,'Bu':None,'ko':None,'qr':None,'sl':None},olayer_activation_func=None, batch_norm={}, \
                        residual_module_par={'NStacks':3,'Kernel_Init':'glorot_normal','Batch_Norm':{},'Activation_Func':tf.nn.relu},poly_coeff_Bu=None,train_stats=None,trn_inputs=None,aux_labels=None,policy=None):
    # Data Arrangement:
    # Type 0: Three outputs--pressure, gas saturation and condensate saturation
    # Type 1: Seven outputs--pressure, gas saturation, condensate saturation, gas FVF, condensate FVF, gas viscosity and condensate viscosity
    # Type 2: Three outputs--pressure, (gas saturation/(gas FVF*gas viscosity)), (condensate saturation/(condensate FVF*condensate viscosity))
    q_well_idx=expand_dims_Layer(conn_idx=cfd_type['Conn_Idx'],qoi=tf.ones_like(cfd_type['Init_Grate']),dim=cfd_type['Dimension']['Dim'],layer_name='expand_dims_q_well_idx',)(inputs)

    def model_wrapper (cfd_type,train_stats): 
        model_wrapper.cfd_type=cfd_type
        model_wrapper.ts=tf.Variable(tf.cast(train_stats[0],dtype=inputs.dtype),trainable=False)
        model_wrapper.dtype=inputs.dtype
        return model_wrapper
    model=model_wrapper(cfd_type,train_stats)

        
    #inputs_list=split_Layer(no_splits=inputs.shape[-1]+1,axis=-1,layer_name='input_split_pre')(tf.keras.layers.Concatenate(axis=-1)([q_well_idx,inputs]))
    inputs_list=split_Layer(no_splits=inputs.shape[-1],axis=-1,layer_name='input_split_pre')(inputs)
    t_n=split_Layer(no_splits=aux_labels.shape[-1],axis=-1,layer_name='time_n',reshape=aux_labels.shape[:-1])(aux_labels)[0]
    
    #inputs_list[-2]=gaussian_blur_Layer(layer_name='perm_blur',sigma=0.5,trainable=True,lower_limit=0.,upper_limit=1.,model=model,f_idx=5)(inputs_list[-2])
    # Compute Pressure and Rate
    
    def iden_fun(x):
        return x
    # Quantity of Interest (QoI)-pressure ANN
    residual_module_par_ps=residual_module_par
    residual_module_par_ps['Network_Type']=config['ps'][0].upper()
    ps=dnn_sub_block(inputs=inputs,width=width['ps'],depth=depth['ps'][0],activation_func=hlayer_activation_func['ps'],kernel_init=kernel['Init'],kernel_regu=None,name='pre_sat_group',residual_module_par=residual_module_par_ps)

    # Intermediate split--pressure and saturation
    # Pressure
    residual_module_par_ps['Network_Type']=config['ps'][1].upper()

    p_hlayer=dnn_sub_block(inputs=ps,width=width['ps'],depth=depth['ps'][1],activation_func=hlayer_activation_func['ps'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['p'],name='pre_split',residual_module_par=residual_module_par_ps,layer_LSTM=depth['LSTM'],layer_CNN=depth['CNN'])
    
    if config['ko'][0] in ['plain','resn',None]:
        # Constitutive property (permeability) ANN
        residual_module_par_ko=residual_module_par
        residual_module_par_ko['Network_Type']=config['ko'][0].upper()
        if cfd_type['Aux_Layer']['ko']:
            phik=dnn_sub_block(inputs=inputs,width=width['ko'],depth=depth['ko'],activation_func=hlayer_activation_func['ko'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['phik'],name='poro-perm',residual_module_par=residual_module_par_ko)
            phi_out=tf.keras.layers.Dense(1, activation=olayer_activation_func['phi'], kernel_initializer=kernel['Init'], name='porosity')(phik)
            k_out=tf.keras.layers.Dense(1, activation=olayer_activation_func['k'], kernel_initializer=kernel['Init'], name='permeability')(phik)
    
    # Check if output of p,s is a Conv2D
    if config['ps'][0]!='cnn2d' or config['ps'][1]!='cnn2d':
        p_int=tf.keras.layers.Dense(1, activation=olayer_activation_func['p'], kernel_initializer=kernel['Init'], name='pressure')(p_hlayer)
    else:
        p_int=p_hlayer

    # Transform Pressure 0.57721*(1/3)
    min_bhp_ij=expand_dims_Layer(conn_idx=cfd_type['Conn_Idx'],qoi=(cfd_type['Min_BHP']),dim=cfd_type['Dimension']['Dim'],layer_name='expand_dims_min_bhp')(p_int)
    hardlayer_act_func=None#tf.keras.layers.ELU(alpha=0.5)
    pre_hardlayer=output_Hardlayer(norm_limits=[cfd_type['Norm_Limits'],cfd_type['Norm_Limits']],init_value=cfd_type['Pi'],co_var='no_correlate',input_activation=hardlayer_act_func,layer_name='pressure_hardlayer',\
                                   kernel_exponent=[{'Value':0.57721*(1/3),'Trainable':True,'Lower_Limit':0.,'Upper_Limit':0.57721,'Rate':1.},{'Value':1.,'Trainable':False,'Lower_Limit':0.1,'Upper_Limit':10,'Rate':1.}],regularization=0.00,weights=[],model=model,cfd_type=cfd_type,dtype=p_int.dtype)
    #pre_hardlayer=output_Hardlayer(norm_limits=[cfd_type['Norm_Limits'],cfd_type['Norm_Limits']],init_value=cfd_type['Pi'],input_activation=hardlayer_act_func,layer_name='pressure_hardlayer',kernel_exponent=[{'Value':1.*(1/1),'Trainable':False,'Lower_Limit':0.,'Upper_Limit':1,'Rate':1.},],regularization=0.00,weights=[],model=None)
    if cfd_type['Type'].upper()=='PINN':
        p_int=pre_hardlayer([[*inputs_list[-2:],1.],p_int],t_n0=t_n)
        #p_int=multiply_Layer(layer_name='pressure_hardlayer_scaled')(p_int,y=1.,constraint=min_bhp_ij)
    # Add a batch normalization after the pressure node with output that connects to the fluid property DNN model  
    if batch_norm['Use_Batch_Norm_After_Pressure']:
        p_int=tf.keras.layers.BatchNormalization(axis=1, momentum=batch_norm['Momentum'], epsilon=1e-6, name='batch_norm_pressure')(p_int)

    if batch_norm['Pre_Rect']['Use_Pre_Rect']:
        rect_func=batch_norm['Pre_Rect']['Act_Func']
        p_int=identity_Layer(layer_name='pre_rect_layer')(p_int)
        #p_int=tf.keras.layers.Lambda(lambda x: rect_func(x),name='pre_rect_layer')(p_int)
        
    # Create a Lambda to transform the pressure output
    #p=tf.keras.layers.Lambda(lambda x: iden_fun(x),name='identity_layer_pressure')(p_int)
    p=identity_Layer(layer_name='identity_layer_pressure')(p_int)

    #================================================================== Compute the Saturations (gas and condensate)=================================================
    # Gas Saturation
    #s_hlayer=saturation_Gas_Oil_Model(config=config,cfd_type=cfd_type,width=width,depth=depth,hlayer_activation_func=hlayer_activation_func,\
     #                                 olayer_activation_func=olayer_activation_func,kernel=kernel,residual_module_par=residual_module_par,layer_name='gsat_osat_layer')(p_int)

    #sg,so=s_hlayer[0][0],s_hlayer[0][1]
    #dS=s_hlayer[1]
    sat_model=[]
    # if cfd_type['Type'].upper()=='PINN':
    #      if cfd_type['Model_Pre_Sat'] or cfd_type['Fluid_Type'] in ['GC','gas_cond']:
    #         #p re_sat_inp_m=tf.keras.Input(shape=[1],name='pressure_saturation_input') #p_int.shape[1:]
    #         s_out=layers_Gas_Oil_Saturation(inputs=p_int,pre=pre_sat_inp,config=config,cfd_type=cfd_type,width=width,depth=depth,hlayer_activation_func=hlayer_activation_func,\
    #                                           olayer_activation_func=olayer_activation_func,kernel=kernel,residual_module_par=residual_module_par,layer_name='gsat_osat_layer',solu_type='')
    #         s_hlayer=saturation_Gas_Oil_Model(config=config['ps'],cfd_type=cfd_type,width=width['ps'],depth=depth['ps'],olayer_activation_func=olayer_activation_func,kernel=3,residual_module_par=residual_module_par,pressure_dependence=True,hard_enforcement=True,layer_name='')
    #         # s_out=s_hlayer(pre_sat_inp_m)
    #         sat_model=tf.keras.Model(inputs=p_int,outputs=s_out,name='pressure_saturation_model')
    #         config['ps'][2]=config['ps'][3]=None
            
    s_hlayer=layers_Gas_Oil_Saturation(inputs=inputs,pre=p_int,config=config,model=model,width=width,depth=depth,hlayer_activation_func=hlayer_activation_func,\
                                           olayer_activation_func=olayer_activation_func,kernel=kernel,residual_module_par=residual_module_par,layer_name='gsat_osat_layer',solu_type='')
    # s_hlayer=saturation_Gas_Oil_Model(config=config['ps'],cfd_type=cfd_type,width=width['ps'],depth=depth['ps'],olayer_activation_func=olayer_activation_func,kernel=3,residual_module_par=residual_module_par,pressure_dependence=True,hard_enforcement=True,compute_derivatives=False,reshape_output=p_int.shape[1:],layer_name='gsaturation_layer')
    # s_hlayer=s_hlayer(pre_sat_inp)
    s=s_hlayer
    #======================================================Add slack layer variable for optimization of gas rate=======================================================
    if cfd_type['Type'].upper()=='PINN':
        residual_module_par_ps['Network_Type']=config['qs'][0].upper()
        qg_hlayer=dnn_sub_block(inputs=ps,width=width['qs'],depth=depth['qs'][0],activation_func=hlayer_activation_func['qr'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['qr'],name='grate_split',residual_module_par=residual_module_par_ps)

        residual_module_par_ps['Network_Type']=config['qs'][1].upper()
        #ps_c=tf.keras.layers.Concatenate(axis=-1)([ps[...,:2],ps[...,-1:]])
        s1_hlayer=dnn_sub_block(inputs=ps,width=width['qs'],depth=depth['qs'][1],activation_func=hlayer_activation_func['sl'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['sl'],name='slack1_split',residual_module_par=residual_module_par_ps)

        # Compute rate and slack variables (if any)
        qg_int=tf.keras.layers.Dense(1, activation=olayer_activation_func['qr'], kernel_initializer=kernel['Init'], name='grate')(qg_hlayer)
        s1_int=tf.keras.layers.Dense(1, activation=olayer_activation_func['sl'], kernel_initializer=kernel['Init'], name='slack1')(s1_hlayer)
        qg=identity_Layer(layer_name='identity_layer_grate')(qg_int)
        s1=identity_Layer(layer_name='identity_layer_slack1')(s1_int)
        #================================================Pressure dependent layer for Formation Volume Factor (FVF) and viscosity=========================================  
        list_idx=[0,1]; 
        if cfd_type['Fluid_Type'] in ['gas_cond','GC']: list_idx=[0,1,2,3,4,5]
        #klayer=constant_Layer(constant=5000.)
        pvt_inp=(p_int)
        PVT_layer=PVT_Model(config=config,cfd_type=cfd_type,poly_coeff_Bu=poly_coeff_Bu,list_idx=list_idx,layer_name='PVT_layer')
        PVT_hlayer=PVT_layer(pvt_inp)
        
        if cfd_type['Aux_Layer']['Bu']['Use']:
            out_PVT=layers_PVT(inputs=pvt_inp,cfd_type=cfd_type,config=config,poly_coeff_Bu=poly_coeff_Bu,name='PVT_layer')
            invBg,invug=out_PVT[0],out_PVT[1]
            if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
                invBg,invBo,invug,invuo,Rs,Rv,Vro=out_PVT[0],out_PVT[1],out_PVT[2],out_PVT[3],out_PVT[4],out_PVT[5],out_PVT[6]
            #invBg,invug=PVT_hlayer[0][0],PVT_hlayer[0][1]
            dPVT=PVT_hlayer[1]
            #if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
            #invBg,invBo,invug,invuo,Rs,Rv=PVT_hlayer[0][0],PVT_hlayer[0][1],PVT_hlayer[0][2],PVT_hlayer[0][3],PVT_hlayer[0][4],PVT_hlayer[0][5]
        else:
            invBg,invBo,invug,invuo,Rs,Rv,Vro=None,None,None,None,None,None
    
        PVT_model_inp=tf.reshape(pvt_inp,(-1,))  
        PVT_model_out=PVT_Model(config=config,cfd_type=cfd_type,poly_coeff_Bu=poly_coeff_Bu,list_idx=list_idx,layer_name='PVT_layer')(PVT_model_inp)
        pvt_io=[PVT_model_inp,PVT_model_out]
        # Update the control mode indexes to 0 and 1 for rate and bhp respectively
        #rate_idx=tf.cast(np.where(np.array(cfd_type['Control_Mode'])=='Rate',1.,0.),inputs.dtype)
        #BHP_idx=tf.cast(np.where(np.array(cfd_type['Control_Mode'])=='BHP',1.,0.),inputs.dtype)
        #Reshapes the PVT outputs
        if config['ps'][0]=='cnn2d':  #Reshapes the FVF and viscosity if it is 2D layer
            # Reshape the FVF and viscosity if pressure layer is a CNN2D
            invBg=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_InvBg_layer_CNV2D')(invBg)
            invug=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_Invug_layer_CNV2D')(invug)
            
            # Condensates/Oils
            if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
                invBo=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_InvBo_layer_CNV2D')(invBo)
                invuo=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_Invuo_layer_CNV2D')(invuo)
                Rs=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_Rs_layer_CNV2D')(Rs)
                Rv=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_Rv_layer_CNV2D')(Rv)
                Vro=tf.keras.layers.Reshape(p_int.shape[1:], name='PVT_layer'+'Reshaped_Vro_layer_CNV2D')(Vro)
        
        # Create the Connection Index Tensor for the wells
        cfd_type['Control_Mode']={'Rate':tf.cast(np.where(np.array(cfd_type['Control_Mode'])=='Rate',1.,0.),inputs.dtype),'BHP':tf.cast(np.where(np.array(cfd_type['Control_Mode'])=='BHP',1.,0.),inputs.dtype)}
        grid_zero_idx=np.zeros(model.cfd_type['Dimension']['Reshape'])
        for idx in model.cfd_type['Conn_Idx']:
            grid_zero_idx[idx]=1.
        grid_zero_idx=tf.convert_to_tensor(grid_zero_idx,dtype=inputs.dtype)
    else:
        pvt_io=[]
    # List outputs 
    if cfd_type['Fluid_Type'] in ['dry_gas','dry-gas','DG']:
        if cfd_type['Data_Arr']==0:
            outputs=[p,*s]
        elif cfd_type['Data_Arr']==1:
            outputs=[p,*s]
            if cfd_type['Aux_Layer']['Bu']['Use'] and cfd_type['Type'].upper()=='PINN':
                outputs+=[invBg,invug,dPVT,]
        else:
            # Create a Lambda layer to perform operation (s/Bu) for Arrangement Type 2
            sbu_g=tf.keras.layers.Multiply(name='sbu_gas_layer')([sg,invBg,invug])
            outputs=[p,sbu_g]
        
    # Append to the output list if condensate is trained along side other labels 
    if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
        if cfd_type['Data_Arr']==0:
            outputs=[p,*s]
        elif cfd_type['Data_Arr']==1:
            outputs=[p,*s]
            if cfd_type['Aux_Layer']['Bu']['Use'] and cfd_type['Type'].upper()=='PINN':
                outputs+=[invBg,invBo,invug,invuo,Rs,Rv,Vro,dPVT]
        elif cfd_type['Data_Arr']==2:
            sbu_g=tf.keras.layers.Multiply(name='sbu_gas_layer')([sg,invBg,invug])
            sbu_o=tf.keras.layers.Multiply(name='sbu_oil_layer')([so,invBo,invuo])
            outputs=[p,sbu_g,sbu_o]
    
    # Add a separate Truncation Error Model
    if 'Truncation_Error_Model' in cfd_type and cfd_type['Type'].upper()=='PINN':
        #if cfd_type['Truncation_Error_Model']:
        #kernel_cons=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        residual_module_par_dt=residual_module_par
        #residual_module_par_dt['Network_Type']=config['tr'][0].upper()
        # width['tr']['Growth_Rate']*=2.0
        # width['tr']['Bottom_Size']*=0.5;
        residual_module_par_dt['Kernel_Init']='glorot_normal'
        residual_module_par_dt['Activation_Func']=tf.nn.swish
        residual_module_par_dt['Latent_Layer']['Width']*=1.0
        #dt_out=dnn_sub_block(inputs=inputs,width=width['tr'],depth=depth['tr'][0],activation_func=hlayer_activation_func['tr'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['tr'],name='time_step',olayer_act_func=olayer_activation_func['dt'],residual_module_par=residual_module_par_dt)       
        # inputs_pre=normalization_Layer(stat_limits=[model.cfd_type['Pi'],14.7],norm_limits=model.cfd_type['Norm_Limits'],layer_name='pre_normalize')(p_int)
        # inputs_tstep=tf.keras.layers.Concatenate(axis=-1)(inputs_list+[inputs_pre])

        #attn=self_attention_Layer(num_heads=2,shape=inputs.shape,layer_name='time_step_attention',filter_factor=1)
        rnn1=dnn1.RNN_Layer(depth=depth['tr'][0],width=width['tr'],kernel_size=3,res_par=residual_module_par_dt,layer_name='time_step',gaussian_process=None,out_act_func=olayer_activation_func['dt'],batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
        rnn2=dnn1.RNN_Layer(depth=depth['tr'][0],width=width['tr'],kernel_size=3,res_par=residual_module_par_dt,layer_name='trunc_error',gaussian_process=None,out_act_func=olayer_activation_func['tr'],batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
        
        dt_out=rnn1(inputs,output_layer=True)
        trn_out=rnn2(inputs,output_layer=True)

        # tstep_hardlayer=output_Hardlayer(norm_limits=[cfd_type['Norm_Limits'],cfd_type['Norm_Limits']],init_value=0.1,co_var='no_correlate',input_activation=lambda x:-x,layer_name='timestep_hardlayer',\
        #                             kernel_exponent=[{'Value':0.57721*(1/3),'Trainable':False,'Lower_Limit':0.,'Upper_Limit':0.57721,'Rate':1.},{'Value':1.,'Trainable':False,'Lower_Limit':0.1,'Upper_Limit':10,'Rate':1.}],regularization=0.00,weights=[],model=model,cfd_type=cfd_type,dtype=p_int.dtype)
        # dt_out=tstep_hardlayer([[inputs_list[-2],inputs_list[-1]],dt_out],t_n0=t_n)

        # tstep_hardlayer=timestep_Hardlayer(layer_name='timestep_hardlayer',norm_limits=[[-1,1],[-1,1]],init_value=1.,alpha={'Value':1.,'Trainable':True,'Lower_Limit':0.1,'Upper_Limit':5.,'Rate':1.},\
        #                     tshift=1.0,model=model,cfd_type=None,activation_func=tf.nn.softplus)
        # dt_out=tstep_hardlayer([[inputs_list[-2],t_n],dt_out])
        

        #trn_out=multiply_Layer(layer_name='bhp_output')(x=trn_out,y=q_well_idx,constraint=None) binary_Layer(values=[0.,lambda x:-tf.ones_like(x)])

        #pwf_hardlayer=output_Hardlayer(norm_limits=[cfd_type['Norm_Limits'],cfd_type['Norm_Limits']],init_value=cfd_type['Pi'],input_activation=hardlayer_act_func,layer_name='bhp_hardlayer',kernel_exponent=[{'Value':0.57721*(1/3),'Trainable':True,'Lower_Limit':0.,'Upper_Limit':1,'Rate':1.},],regularization=0.00,weights=[],model=None)
        #trn_out=pwf_hardlayer([[inputs_list[-2],],trn_out],init_layer=None)
        #trn_out=multiply_Layer(layer_name='bhp_output')(x=trn_out,y=q_well_idx,constraint=min_bhp_ij)
        
        mbc_tstep=[dt_out,trn_out]
        tstep_model=tf.keras.Model(inputs=[inputs,],outputs=mbc_tstep,name='time_step_model')
    else:
        mbc_tstep=[]
        tstep_model=[]
    # if 'Perm_Model' in cfd_type and cfd_type['Type'].upper()=='PINN':
    #     rnn_perm=dnn1.RNN_Layer(depth=depth['tr'][0],width=width['tr'],kernel_size=3,res_par=residual_module_par,layer_name='time_step',gaussian_process=True,out_act_func=olayer_activation_func['dt'],batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
    #     perm_out=rnn_perm(inputs_list[:2],output_layer=True)
        
    outputs+=mbc_tstep

    # Compute the rates and bottomhole pressure and append to output layer
    # Append rates and slack variables
    if cfd_type['Type'].upper()=='PINN':
         #outputs=outputs+[qg,s1]
         if cfd_type['Compute_Rate_BHP']:
             inputs_rate_bhp={'Time':inputs_list[-2],'Kx':inputs_list[-1],'Dx':cfd_type['Dimension']['Gridblock_Dim'][0],'Dy':cfd_type['Dimension']['Gridblock_Dim'][1],'Dz':cfd_type['Dimension']['Gridblock_Dim'][2],\
                     'Pre':p,'InvBg':invBg,'Invug':invug,'Sg':s[0],'t_n':t_n}
             if cfd_type['Fluid_Type'] in ['GC','gas_cond']:
                 inputs_rate_bhp.update({'InvBo':invBo,'Invuo':invuo,'Rs':Rs,'Rv':Rv,'Vro':Vro,'So':s[1],'trn':trn_out})
             opt_rate,BHP=computeRateBHP(model=model,config=config,poly_coeff_Bu=poly_coeff_Bu,PVT_layer=PVT_layer,shape=dt_out.get_shape()[0],dtype=p.dtype)(inputs_rate_bhp)
             outputs.extend([opt_rate,BHP])
         else:
             outputs.extend([])
    return outputs,mbc_tstep,pvt_io,sat_model,tstep_model#pre_hardlayer.kernel_exponent
         
class computeRateBHP(tf.keras.layers.Layer):
    def __init__(self,C=0.001127,D=5.6145833334,model=None,config=None,poly_coeff_Bu=None,scal=None,PVT_layer=None,cname='Compute_Rate_BHP_Layer',shape=[None,39,39,1],dtype=None,**kwargs):
        super(computeRateBHP, self).__init__(**kwargs,name=cname)
        self.C=tf.constant(C, dtype=dt_type, shape=(), name='const1')
        self.D=tf.constant(D, dtype=dt_type, shape=(), name='const2')
        self.model=model  #unitialized model wrapper
        self.config=config
        self.poly_coeff_Bu=poly_coeff_Bu 
        self.PVT_layer=PVT_layer

      #self.dt_n_1=tf.Variable(0.,shape=tf.TensorShape(None),trainable=False,validate_shape=False,dtype=self.dtype,import_scope=cname)
        self.dt_n_1=(tf.Variable([0.],shape=[None,],trainable=False,validate_shape=False,dtype=self.dtype))
    
    def call(self, x):
        t=bl.nonormalize(self.model,x['t_n'],stat_idx=3,compute=True)
        kx_ij=ky_ij=bl.nonormalize(self.model,x['Kx'],stat_idx=5,compute=True)
        p_ij=x['Pre']
        dx=x['Dx']
        dy=x['Dy']
        h=x['Dz']
        rw=tf.constant(self.model.cfd_type['Wellbore_Radius'], dtype=dt_type, shape=(), name='rw')
        hc=tf.constant(self.model.cfd_type['Completion_Ratio'], dtype=dt_type, shape=(), name='hc')
        ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
        q_well_idx=tf.expand_dims(tf.scatter_nd(self.model.cfd_type['Conn_Idx'], tf.ones_like(self.model.cfd_type['Init_Grate']), self.model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(kx_ij)
        shutins_idx=tf.reduce_mean([tf.reduce_mean([self.shut_days(limits=self.model.cfd_type['Connection_Shutins']['Days'][c][cidx],time=t,dtype=self.dtype) for cidx in self.model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in self.model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)
        q_t0_ij=shutins_idx*tf.expand_dims(tf.scatter_nd(self.model.cfd_type['Conn_Idx'], self.model.cfd_type['Init_Grate'], self.model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(kx_ij) 
        min_bhp_ij=tf.expand_dims(tf.scatter_nd(self.model.cfd_type['Conn_Idx'], self.model.cfd_type['Min_BHP'], self.model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(kx_ij)
        invBgug_ij=x['InvBg']*x['Invug']; invBg_ij=x['InvBg']
        Sg_ij=x['Sg']

        kro_ij,krg_ij=self.model.cfd_type['Kr_gas_oil'](Sg_ij)

        if self.config['ps'][2] is None:
            kro_ij,krg_ij=self.model.cfd_type['Kr_gas_oil'](1-self.model.cfd_type['SCAL']['End_Points']['Swmin'])

        Ck_ij=(2*(22/7)*hc*kx_ij*h*self.C)/(tf.math.log(ro/rw))
        mt_ij=krg_ij*invBgug_ij #Sg_ij,Sg_ij
      
        # Gas Cond Compute the PVT properties at Min_BHP for Linearization of Well Rates
        if self.model.cfd_type['Fluid_Type'] in ['GC','gas_cond']:
            invBouo_ij=x['InvBo']*x['Invuo']
            Rs_ij=x['Rs']
            Rv_ij=x['Rv']
            Sgi=1.-self.model.cfd_type['SCAL']['End_Points']['Swmin']
            Sorg=self.model.cfd_type['SCAL']['End_Points']['Sorg']
            p_dew=self.model.cfd_type['Dew_Point']
            # PVT_min_bhp=self.PVT_layer(tf.reshape(min_bhp_ij,(-1,)))[0]
            # invBgug_min_bhp_ij=tf.reshape(PVT_min_bhp[0]*PVT_min_bhp[1],tf.shape(p_ij))
            # invBgug_min_bhp_ij=tf.reshape(PVT_min_bhp[0]*PVT_min_bhp[2],tf.shape(p_ij))
            # invBouo_min_bhp_ij=tf.reshape(PVT_min_bhp[1]*PVT_min_bhp[3],tf.shape(p_ij))
            # mt_ij=(krg_ij*invBgug_ij)+(kro_ij*invBouo_ij)

            qfg_ij,qdg_ij,qfo_ij,qvo_ij,pwf_ij=bl.compute_rate_bhp_gas_oil(p_ij,Sg_ij,invBgug_ij,invBouo_ij,Rs_ij,Rv_ij,krg_ij,kro_ij,q_t0_ij,min_bhp_ij,Ck_ij,q_well_idx,_Sgi=Sgi,_Sorg=Sorg,_p_dew=p_dew,\
                                 _pi=self.model.cfd_type['Pi'],_pbase=0.,_shutins_idx=shutins_idx,_shift=0.001,_lmd=0.,_ctrl_mode=1,_no_intervals=12,_no_iter_per_interval=4,pre_sat_model=None,rel_perm_model=self.model.cfd_type['Kr_gas_oil'],model_PVT=self.PVT_layer)
            q_ij_opt,pwf=tf.stack([qfg_ij,qdg_ij,qfo_ij,qvo_ij]),pwf_ij
        else:
            q_ij_opt,pwf=bl.compute_rate_bhp_gas(p_ij,invBgug_ij,krg_ij,q_t0_ij,min_bhp_ij,Ck_ij,_shutins_idx=shutins_idx,_invBg_n1_ij=invBg_ij,_no_intervals=2,_ctrl_mode=1,_q_well_idx=q_well_idx,model_PVT=self.PVT_layer)
 
        # # Compute the blocking factor for well flow rates linearization
        # #blk_fac=compute_integral_mg(p_ij,invBgug_ij,min_bhp_ij,_no_intervals=5,model_PVT=PVT_layer)
        # blk_fac=tf.maximum((0.5*(invBgug_min_bhp_ij+invBgug_ij)*(p_ij-min_bhp_ij))/(invBgug_ij*(p_ij-min_bhp_ij)),0.)

        # # Compute the rate at minimum bottomhole pressure
        # qmax_ij_min_bhp=q_well_idx*((2*(22/7)*hc*kx_ij*(blk_fac*mt_ij)*self.C)*(1/(tf.math.log(ro/rw)))*tf.math.abs(p_ij-min_bhp_ij))
        # q_ij_opt=tf.math.minimum(q_ij,qmax_ij_min_bhp)
        # pwf=q_well_idx*(p_ij-(q_ij_opt/(2*(22/7)*hc*kx_ij*mt_ij*self.C))*(tf.math.log(ro/rw))) 
        #self.dt_n_1.assign(tf.expand_dims(tf.reduce_mean(dt),-1))
        return q_ij_opt,pwf
    
    def shut_days(self,limits=None,time=None,dtype=None,tscale=10.):
        return (tf.ones_like(time)-tf.cast((time>=limits[0])&(time<=limits[1])&(tf.reduce_sum(limits)>0),dtype)) 

def layers_PVT(inputs=None, config=None,cfd_type=None,poly_coeff_Bu=None,name=''):
    if config['ps'][0]=='cnn2d':
        # Flatten the 2D CNN (auto encoder) 
        cnn2d_shape=inputs.shape
        pre_input=tf.keras.layers.Flatten()(inputs)
        # Expand the inner dimension -- the suit the custom layer
        pre_input=tf.keras.layers.Reshape((*tuple(pre_input.shape[1:]),1), name='Reshaped_InvBu_layer_CNV2D'+name)(pre_input)

 
    if config['Bu'][0] in ['plain','res','cnn2d']:
        residual_module_par_bu=residual_module_par
        residual_module_par_bu['Network_Type']=config['Bu'][0].upper()
        # Constitutive behaviour (viscosity-FVF related to pseudopressure) ANN
        invBu=dnn_sub_block(inputs=pre_input,width=width['Bu'],depth=depth['Bu'][0],activation_func=hlayer_activation_func['Bu'],kernel_init=kernel['Init'],kernel_regu=None,name='invB_invu_group'+name,residual_module_par=residual_module_par_bu)
        # Intermediate splits--(1/Bg),(1/Bo),(1/ug),(1/uo)
        # (1/Bg)
        invBg_hlayer=dnn_sub_block(inputs=invBu,width=width['Bu'],depth=depth['Bu'][1],activation_func=hlayer_activation_func['Bu'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['Bu'],name='invBg_split'+name,residual_module_par=residual_module_par_bu)
  
        # (1/ug)
        invug_hlayer=dnn_sub_block(inputs=invBu,width=width['Bu'],depth=depth['Bu'][3],activation_func=hlayer_activation_func['Bu'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['Bu'],name='invug_split'+name,residual_module_par=residual_module_par_bu)
    
        invBg=tf.keras.layers.Dense(1, activation=olayer_activation_func['invB'], kernel_initializer=kernel['Init'], name='invBg')(invBg_hlayer)
        invug=tf.keras.layers.Dense(1, activation=olayer_activation_func['invu'], kernel_initializer=kernel['Init'], name='invug')(invug_hlayer)
        if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
            # (1/Bo)
            invBo_hlayer=dnn_sub_block(inputs=invBu,width=width['Bu'],depth=depth['Bu'][2],activation_func=hlayer_activation_func['Bu'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['Bu'],name='invBo_split'+name,residual_module_par=residual_module_par_bu)
            # (1/uo)
            invuo_hlayer=dnn_sub_block(inputs=invBu,width=width['Bu'],depth=depth['Bu'][4],activation_func=hlayer_activation_func['Bu'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['Bu'],name='invuo_split'+name,residual_module_par=residual_module_par_bu)
    
            invBo=tf.keras.layers.Dense(1, activation=olayer_activation_func['invB'], kernel_initializer=kernel['Init'], name='invBo'+name)(invBo_hlayer)
            invuo=tf.keras.layers.Dense(1, activation=olayer_activation_func['invu'], kernel_initializer=kernel['Init'], name='invuo'+name)(invuo_hlayer)
    else:  # has numerical value
        # Use an approximating function--a polynomial is used in this case
        name_Bg,order_Bg=str(config['Bu'][1][:-1]),int(config['Bu'][1][-1])
        name_ug,order_ug=str(config['Bu'][3][:-1]),int(config['Bu'][3][-1])
        
        lim_Bu=lambda p1: lambda p2: tf.where(p1<cfd_type['Dew_Point'],p2,tf.zeros_like(p2))
        
        
        if name_Bg.upper()=='POLY':
            #pre_input=tf.keras.layers.Lambda(lambda x: tf.transpose(x,perm=[1,0,2]),name='Transpose_InvBu_layer_CNV2D')(pre_input)
            #invBg=tf.keras.layers.Activation(hard_limit_func(lower_limit=0.,upper_limit=5.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['InvBg'],cname='invBg_polynomial')(pre_input))
            invBg=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['InvBg'],cname='invBg_polynomial'+name)(pre_input))
        else:
            #Fit with a spline
            invBg=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['InvBg'],order=order_Bg,layer_name='invBg_spline'+name)(pre_input))
                
        if name_ug.upper()=='POLY':
            #invug=tf.keras.layers.Activation(hard_limit_func(lower_limit=10.,upper_limit=100.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['Invug'],cname='invug_polynomial')(pre_input))
            invug=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['Invug'],cname='invug_polynomial'+name)(pre_input))
        else:
            #Fit with a spline
            invug=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Invug'],order=order_ug,layer_name='invug_spline'+name)(pre_input))

        if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
            name_Bo,order_Bo=str(config['Bu'][2][:-1]),int(config['Bu'][2][-1])
            name_uo,order_uo=str(config['Bu'][4][:-1]),int(config['Bu'][4][-1])
            name_Rs,order_Rs=str(config['Bu'][5][:-1]),int(config['Bu'][5][-1])
            name_Rv,order_Rv=str(config['Bu'][6][:-1]),int(config['Bu'][6][-1])
            if name_Bo.upper()=='POLY':
                invBo=tf.keras.layers.Activation(hard_limit_func(lower_limit=0.,upper_limit=5.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['InvBo'],cname='invBo_polynomial'+name)(pre_input))
                invuo=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['Invuo'],cname='invuo_polynomial'+name)(pre_input))
                Rs=tf.keras.layers.Activation(lambda x: tf.math.minimum(tf.math.maximum(x,0.),cfd_type['Init_Rs']))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['Rs'],cname='Rs_polynomial'+name)(pre_input))
                Rv=tf.keras.layers.Activation(lambda x: tf.math.minimum(tf.math.maximum(x,0.),cfd_type['Init_Rv']))(viscosity_FVF_inverse(pweights=poly_coeff_Bu['Rv'],cname='Rv_polynomial'+name)(pre_input))       
            else:
                #invBo=tf.keras.layers.Activation(hard_limit_func(lower_limit=0.,upper_limit=5.))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['InvBo'],order=order_Bo,layer_name='invBo_spline'+name)(pre_input))
                #invuo=tf.keras.layers.Activation(lambda x: tf.math.maximum(x,0.))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Invuo'],order=order_uo,layer_name='invuo_spline'+name)(pre_input))
                # Rs=tf.keras.layers.Activation(lambda x: tf.math.minimum(tf.math.maximum(x,0.),cfd_type['Init_Rs']))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Rs'],order=order_Rs,layer_name='Rs_spline'+name)(pre_input))
                # Rv=tf.keras.layers.Activation(lambda x: tf.math.minimum(tf.math.maximum(x,0.),cfd_type['Init_Rv']))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Rv'],order=order_Rv,layer_name='Rv_spline'+name)(pre_input))
                Vro=tf.keras.layers.Activation(lambda x: tf.math.minimum(tf.math.maximum(x,0.),1.))(spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Vro'],order=order_Rv,layer_name='Vro_spline'+name)(pre_input))
                invBo=restep_Layer(limits=[0.,cfd_type['Dew_Point']],values=[1.,cfd_type['Init_InvBo']],layer_name='InvBo_Limit_Dew_Point')([pre_input,spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['InvBo'],order=order_Rs,layer_name='InvBo_spline'+name)(pre_input)])
                invuo=restep_Layer(limits=[0.,cfd_type['Dew_Point']],values=[0.,cfd_type['Init_Invuo']],layer_name='Invuo_Limit_Dew_Point')([pre_input,spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Invuo'],order=order_uo,layer_name='Invuo_spline'+name)(pre_input)])
                Rs=restep_Layer(limits=[0.,cfd_type['Dew_Point']],values=[0.,cfd_type['Init_Rs']],layer_name='Rs_Limit_Dew_Point')([pre_input,spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Rs'],order=order_Rs,layer_name='Rs_spline'+name)(pre_input)])
                Rv=restep_Layer(limits=[0.,cfd_type['Dew_Point']],values=[0.,cfd_type['Init_Rv']],layer_name='Rv_Limit_Dew_Point')([pre_input,spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Rv'],order=order_Rs,layer_name='Rv_spline'+name)(pre_input)])

        # if cfd_type['Fit_Fluid_Derivatives'] and name_Bg.upper()!='POLY':
        #     dinvBg=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dInvBg'],order=order_Bg,layer_name='dinvBg_spline'+name)(pre_input)
        #     dinvug=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dInvug'],order=order_Bg,layer_name='dinvug_spline'+name)(pre_input)
        #     if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
        #         dinvBo=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dInvBo'],order=order_Bg,layer_name='dinvBo_spline'+name)(pre_input)
        #         dinvuo=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dInvuo'],order=order_Bg,layer_name='dinvuo_spline'+name)(pre_input)
        #         dRs=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dRs'],order=order_Rs,layer_name='dRs_spline'+name)(pre_input)
        #         dRv=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['dRv'],order=order_Rv,layer_name='dRv_spline'+name)(pre_input)
        #         dVro=spline_Interpolation(points_x=poly_coeff_Bu['Pre'],values_y=poly_coeff_Bu['Vro'],order=order_Rv,layer_name='dVro_spline'+name)(pre_input)

        # # Add the reshaped derivatives if using a polynomial function
        # if cfd_type['Fit_Fluid_Derivatives'] and name_Bg.upper()!='POLY':
        #     dinvBg=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dInvBg_layer_CNV2D'+name)(dinvBg)
        #     dinvug=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dInvug_layer_CNV2D'+name)(dinvug)
        #     if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
        #         dinvBo=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dInvBo_layer_CNV2D'+name)(dinvBo)
        #         dinvuo=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dInvuo_layer_CNV2D'+name)(dinvuo)
        #         dRs=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dRs_layer_CNV2D'+name)(dRs)
        #         dRv=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dRv_layer_CNV2D'+name)(dRv)
        #         dVro=tf.keras.layers.Reshape(cnn2d_shape[1:], name='Reshaped_dVro_layer_CNV2D'+name)(dVro)
        #         #[dinvBg,dinvBo,dinvug,dinvuo,dRs,dRv]
            
    # Return outputs
    out_PVT=[invBg,invug]
    if cfd_type['Fluid_Type'] in ['gas_cond','GC']:
        out_PVT=[invBg,invBo,invug,invuo,Rs,Rv,Vro]
    return out_PVT

def residual_module(inputs=None, width=None, depth=None, kernel_init=None,kernel_regu=None,activity_regu=None,batch_norm={'Use_Batch_Norm':False,'Before_Activation':True,'Momentum':0.99}, activation_func=None, dropout={'Add':False,'Dropout_Idx_Stack':0,'Rate':0.05},block_name=None ): 
    # The RELU activation function which is computationally cost effective is usually used due to a resulting deeper network configuration
    if depth<=0:
        outputs=inputs
    else:
        for i in range(0,depth):
            if i==0:
                int_outputs = tf.keras.layers.Dense(width, activation=None, kernel_initializer=kernel_init, kernel_regularizer=kernel_regu, activity_regularizer=activity_regu,name=block_name+'_sub_hlayer_'+str(i+1))(inputs)
            else:
                int_outputs = tf.keras.layers.Dense(width, activation=None, kernel_initializer=kernel_init, kernel_regularizer=kernel_regu, activity_regularizer=activity_regu,name=block_name+'_sub_hlayer_'+str(i+1))(int_outputs)
            
            # Apply Batch Normalization, usually before activation -- although some ML enthusisists show BN after activation gives a better match
            if i<(depth-1):
                if batch_norm['Use_Batch_Norm']:
                    if batch_norm['Before_Activation']:
                        int_outputs = tf.keras.layers.BatchNormalization(axis=1, momentum=batch_norm['Momentum'], epsilon=1e-6, name=block_name+'_batch_norm_sub_hlayer_'+str(i+1))(int_outputs) 
                        int_outputs = tf.keras.layers.Activation(activation_func, name=block_name+'_activation_sub_hlayer_'+str(i+1))(int_outputs)
                    else:
                        int_outputs = tf.keras.layers.Activation(activation_func, name=block_name+'_activation_sub_hlayer_'+str(i+1))(int_outputs)
                        int_outputs = tf.keras.layers.BatchNormalization(axis=1, momentum=batch_norm['Momentum'], epsilon=1e-6, name=block_name+'_batch_norm_sub_hlayer_'+str(i+1))(int_outputs) 
                else:
                    int_outputs = tf.keras.layers.Activation(activation_func, name=block_name+'_activation_sub_hlayer_'+str(i+1))(int_outputs)
            else:   
                if batch_norm['Use_Batch_Norm']:
                    if batch_norm['Before_Activation']:
                        int_outputs = tf.keras.layers.BatchNormalization(axis=1, momentum=batch_norm['Momentum'], epsilon=1e-6, name=block_name+'_batch_norm_sub_hlayer_'+str(i+1))(int_outputs) 
                    else:
                        int_outputs = tf.keras.layers.Activation(activation_func, name=block_name+'_activation_sub_hlayer_'+str(i+1))(int_outputs)
                else:
                    # If Batch Normalization is not used, the nth layer output, is the dense layer output
                    continue

        # Check the Input-Output dimensions. If input dimension = output dimension -- Identity block, else Dense block
        if inputs.shape[-1]!=int_outputs.shape[-1]:
            # Add a dense layer
            proj_inputs = tf.keras.layers.Dense(int_outputs.shape[-1], activation=None, kernel_initializer=kernel_init, kernel_regularizer=kernel_regu, activity_regularizer=activity_regu, name=block_name+'_proj_inputs')(inputs)
            if batch_norm['Use_Batch_Norm']:
                if batch_norm['Before_Activation']: 
                    # Add a Batch Normalization layer
                    proj_inputs=tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm['Momentum'], epsilon=1e-6, name=block_name+'_batch_norm_proj_inputs')(proj_inputs) 
                else:
                    proj_inputs = tf.keras.layers.Activation(activation_func, name=block_name+'_activation_proj_inputs')(proj_inputs)
        else:
            proj_inputs = inputs
        
        outputs=tf.keras.layers.add([int_outputs,proj_inputs])
        # Output Layer
        if batch_norm['Use_Batch_Norm']:
            if batch_norm['Before_Activation']:
                outputs=tf.keras.layers.Activation(activation_func, name=block_name+'_output_layer_activation')(outputs)
            else: 
                outputs=tf.keras.layers.BatchNormalization(axis=1, momentum=batch_norm['Momentum'], epsilon=1e-6, name=block_name+'_output_layer_batch_norm')(outputs) 
        else:
            outputs=tf.keras.layers.Activation(activation_func, name=block_name+'_output_layer_activation')(outputs)
            
        # Apply dropout regularization -- this is usually after the activation
        if dropout['Add'] in [True,1] and dropout['Dropout_Idx_Stack']==1:
            # Add a dropout layer
            outputs=tf.keras.layers.Dropout(dropout['Rate'], noise_shape=None, seed=None)(outputs)
  
    return outputs

def pad_stack_skip_conn(skip_conn=None, conv2d_hlayer=None, name=None, depth_idx=None,residual_module_par=None):
    if np.prod(skip_conn.shape[1:2])!=np.prod(conv2d_hlayer.shape[1:2]):
        # Pad the layer with offset starting left
        layer_diff=tf.math.abs(skip_conn.shape[1]-conv2d_hlayer.shape[1])
        pad_no=int(layer_diff/2)
        if layer_diff%2==0:  # Even
            skip_pad=tf.keras.layers.ZeroPadding2D(padding=((pad_no,pad_no),(pad_no,pad_no)), data_format=None, name=name+'_skipad_layer_'+str(depth_idx))(skip_conn)
        else:
            skip_pad=tf.keras.layers.ZeroPadding2D(padding=((pad_no+1,pad_no),(pad_no+1,pad_no)), data_format=None, name=name+'_skipad_layer_'+str(depth_idx))(skip_conn)   
    else:
        skip_pad=skip_conn
    if skip_conn.shape[-1]!=conv2d_hlayer.shape[-1]:
        # Pad the channels with a 1x1 convolution filter
        skip_conn_proj=tf.keras.layers.Conv2D(conv2d_hlayer.shape[-1], 1, strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_DEC_skipconn_layer_'+str(depth_idx))(skip_pad)
        #skip_conn_proj=tf.keras.layers.Conv2D(hlayer.shape[-1], residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_DEC_skipconn_layer_'+str(depth_idx))(skip_pad)
    else:
        skip_conn_proj=skip_pad
    return skip_conn_proj

def encoder(inputs=None, depth=None,width=None,residual_module_par={},name=None,nstack_idx=None):
    #inputs_list=split_Layer(no_splits=4,axis=-1,layer_name='input_split_pre')(inputs)
   
    filter_list_enc=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=depth,growth_rate=width['Growth_Rate'],growth_type='smooth',network_type='plain')              # Use of network_width_list function--ngens is equal to depth; growth type is set to smooth; network type as plain
    filter_groups=2
    if residual_module_par['Skip_Connections']['Add'][nstack_idx-1]:
        skip_conn={}

    for i in range(0,depth):
        if i==0:
            hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],activation=None, name=name+'_CNV2D_ENC_layer_'+str(i+1),)(inputs)
        else:
            if i<depth-1:
                # Pads the tensor
                hlayer=tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1)), data_format=None, name=name+'_CNV2D_ENC_pad_layer_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size']+2, strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_ENC_layer_'+str(i+1))(hlayer)
            else:
                hlayer=tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1)), data_format=None, name=name+'_CNV2D_ENC_pad_layer_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'_CNV2D_ENC_layer_'+str(i+1))(hlayer)
                for j in range(1):
                    hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'_CNV2D_ENC_layer_1'+str(i+1)+'_'+str(j+1))(hlayer)
                hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_ENC_layer_0'+str(i+1))(hlayer)
                # hlayer=self_attention_Layer(num_heads=2,shape=hlayer.shape,layer_name=name+'_CNV2D_ENC_self_attention',filter_factor=8)(hlayer,None)
                #hlayer=spatial_attention_Layer(layer_name='CNV2D_ENC_spatial_attention',shape=hlayer.shape,kernel_size=residual_module_par['Kernel_Size'],activation_func=residual_module_par['Activation_Func'])(hlayer)
        if residual_module_par['Skip_Connections']['Add'][nstack_idx-1] and residual_module_par['Skip_Connections']['Layers'][nstack_idx-1][i] not in [None,0]:
            skip_conn[i+1]=hlayer
        hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2D_ENC_activation_layer_'+str(i+1))(hlayer)
        
        if residual_module_par['Dropout']['Add'] in [True,'encoder'] and residual_module_par['Dropout']['Layer'][i]==1:
            # Add a dropout layer
            hlayer=tf.keras.layers.Dropout(residual_module_par['Dropout']['Rate'], noise_shape=None, seed=None)(hlayer) 
    return hlayer,skip_conn

def decoder(inputs=None,depth=None,width=None,latent=None,skip_conn={},residual_module_par={},name=None,nstack_idx=None):
    filter_list_enc=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=depth,growth_rate=width['Growth_Rate'],growth_type='smooth',network_type='plain')              # Use of network_width_list function--ngens is equal to depth; growth type is set to smooth; network type as plain

    for i in range(0,depth):
        import numpy as np
        if i==0:
            if residual_module_par['Latent_Layer']['Flatten'] and residual_module_par['Latent_Layer']['Depth']!=0:
                hlayer=tf.keras.layers.Dense(np.prod(vol[1:]), activation=None, kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],name=name+'_Dense_DEC_layer_CNV2D_'+str(depth-i))(latent)
                hlayer=tf.keras.layers.Reshape((vol[1],vol[2],vol[3]), name=name+'_Reshape_DEC_layer_CNV2D_'+str(depth-i))(hlayer)
            else:
                hlayer=latent
        else:
            if i<depth-1:
                hlayer=tf.keras.layers.Conv2DTranspose(int(filter_list_enc[-1-i]*residual_module_par['Decoder_Filter_Fac']), residual_module_par['Kernel_Size'], strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2DT_DEC_layer_'+str(depth-i))(hlayer)
            else:
                hlayer=tf.keras.layers.Conv2DTranspose(int(filter_list_enc[-1-i]*residual_module_par['Decoder_Filter_Fac']), residual_module_par['Kernel_Size'], strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2DT_DEC_layer_'+str(depth-i))(hlayer)
                
        if len(skip_conn)!=0:
            if residual_module_par['Skip_Connections']['Add'][nstack_idx-1] and residual_module_par['Skip_Connections']['Layers'][nstack_idx-1][depth-1-i] not in [None,0]:
                # nstack_idx-1 is used as there is no combined network for pressure+saturation in the CNN
                skip_conn_proj=pad_stack_skip_conn(skip_conn=skip_conn[depth-i], conv2d_hlayer=hlayer, name=name, depth_idx=depth-i,residual_module_par=residual_module_par)
                # Add the skip connection
                if i==0:
                    if residual_module_par['Latent_Layer']['Depth']!=0: #and residual_module_par['Latent_Layer']['Flatten']:
                        #skip_conn_proj=tf.keras.layers.Activation(residual_module_par['Activation_Func'])(skip_conn_proj)
                        hlayer=tf.keras.layers.Add()([skip_conn_proj,hlayer])
                else:
                    hlayer=tf.keras.layers.Add()([skip_conn_proj,hlayer])
        # Add the Activation Layer
        if i==0:
            if residual_module_par['Latent_Layer']['Depth']!=0 and residual_module_par['Latent_Layer']['Flatten']:
                hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2DT_DEC_activation_layer_'+str(depth-i))(hlayer)
        else:
            hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2DT_DEC_activation_layer_'+str(depth-i))(hlayer)
        # Add Dropout Layer if any
        if residual_module_par['Dropout']['Add'] in [True,'decoder'] and residual_module_par['Dropout']['Layer'][-1-i]==i:
            # Add a dropout layer
            hlayer=tf.keras.layers.Dropout(residual_module_par['Dropout']['Rate'], noise_shape=None, seed=None)(hlayer)
    return hlayer 
  
def encoder_decoder(inputs=None, depth=None,width=None,residual_module_par={},name=None,nstack_idx=None):
    filter_list_enc=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=depth,growth_rate=width['Growth_Rate'],growth_type='smooth',network_type='plain')              # Use of network_width_list function--ngens is equal to depth; growth type is set to smooth; network type as plain
    # Encoder
    hlayer,skip_conn=encoder(inputs=inputs, depth=depth,width=width,residual_module_par=residual_module_par,name=name,nstack_idx=nstack_idx)
    # Latent
    if residual_module_par['Latent_Layer']['Flatten'] and residual_module_par['Latent_Layer']['Depth']!=0:
        vol=hlayer.shape
        hlayer=tf.keras.layers.Flatten()(hlayer)
    latent=hlayer
    if residual_module_par['Latent_Layer']['Depth']!=0:
        if residual_module_par['Latent_Layer']['Network_Type'].upper() in ['DENSE','FCDNN']:
            if not residual_module_par['Latent_Layer']['Skip_Conn']:
                width_list_latent=network_width_list(depth=residual_module_par['Latent_Layer']['Depth'],width=residual_module_par['Latent_Layer']['Width'],ngens=residual_module_par['Latent_Layer']['NStacks'][nstack_idx-1],\
                                              growth_rate=residual_module_par['Latent_Layer']['Growth_Rate'],growth_type=residual_module_par['Latent_Layer']['Growth_Type'],network_type='plain')
                for j in range(0,residual_module_par['Latent_Layer']['Depth']):
                    latent=tf.keras.layers.Dense(width_list_latent[j], activation=residual_module_par['Latent_Layer']['Activation'], kernel_initializer=residual_module_par['Kernel_Init'], kernel_regularizer=residual_module_par['Latent_Layer']['Kernel_Regu'],activity_regularizer=None,name=name+'_Latent_Layer_'+str(j+1))(latent)
            else:
                #Create the stack list  
                # width_latent={'Bottom_Size':residual_module_par['Latent_Layer']['Width'],'Growth_Rate':residual_module_par['Latent_Layer']['Growth_Rate'],'Growth_Type':residual_module_par['Latent_Layer']['Growth_Type']}
                # rnn_latent=dnn1.RNN_Layer(depth=residual_module_par['Latent_Layer']['Depth'],width=width_latent,nstacks=residual_module_par['Latent_Layer']['NStacks'][nstack_idx-1],network_type=residual_module_par['Latent_Layer']['Network_Type'].upper(),kernel_size=3,res_par=residual_module_par,layer_name=name+'_latent_skip',gaussian_process=None,hlayer_act_func=None,out_act_func=None,batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
                # latent=rnn_latent(hlayer,output_layer=False)
                stack_list_latent=sparse_pad_list(depth=residual_module_par['Latent_Layer']['Depth'],nstacks=residual_module_par['Latent_Layer']['NStacks'][nstack_idx-1])
                width_list_latent=network_width_list(depth=residual_module_par['Latent_Layer']['Depth'],width=residual_module_par['Latent_Layer']['Width'],ngens=residual_module_par['Latent_Layer']['NStacks'][nstack_idx-1],\
                                              growth_rate=residual_module_par['Latent_Layer']['Growth_Rate'],growth_type=residual_module_par['Latent_Layer']['Growth_Type'],network_type='resn')
                
                for j in range(0,residual_module_par['Latent_Layer']['Depth']):
                    # Check if residual layer is to be added at index from stack_list
                    if stack_list_latent[j]!=0:
                        latent=residual_module(inputs=latent, width=width_list_latent[j], depth=stack_list_latent[j], kernel_init=residual_module_par['Kernel_Init'], kernel_regu=residual_module_par['Latent_Layer']['Kernel_Regu'], batch_norm=residual_module_par['Batch_Norm'], activation_func=residual_module_par['Latent_Layer']['Activation'], block_name=name+'_Dense_latent_res_block_'+str(j+1))                      
                    else:
                        continue
        elif residual_module_par['Latent_Layer']['Network_Type'].upper() in ['CNN']:
            for j in range(0,residual_module_par['Latent_Layer']['Depth']):
                if j!=residual_module_par['Latent_Layer']['Depth']-1:
                    latent=tf.keras.layers.Conv2D(filter_list_enc[-1],residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Latent_Layer']['Kernel_Regu'],activation=residual_module_par['Latent_Layer']['Activation'],name=name+name+'_CNV2D_hlayer_latent_output_'+str(j+1))(latent)
                else:
                    latent=tf.keras.layers.Conv2D(filter_list_enc[-1], 1, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Latent_Layer']['Kernel_Regu'],activation=None,name=name+name+'_CNV2D_hlayer_latent_output_'+str(j+1))(latent)
        elif residual_module_par['Latent_Layer']['Network_Type'].upper() in ['ATTN']:
            #context_latent=tf.keras.layers.Conv2D(filter_list_enc[-1], 1, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_context_embed_latent')(inputs)
            latent=tf.keras.layers.Dense(filter_list_enc[-1], activation=residual_module_par['Latent_Layer']['Activation'], kernel_initializer=residual_module_par['Kernel_Init'], kernel_regularizer=residual_module_par['Latent_Layer']['Kernel_Regu'],activity_regularizer=None, name=name+'_Latent_Layer_'+str(1))(hlayer)
            #latent_attn=tfa.layers.MultiHeadAttention(head_size=8, num_heads=1,name=name+'latent_attention_layer')([latent,hlayer]) 
            latent_attn=tf.keras.layers.MultiHeadAttention(key_dim=8,num_heads=1,output_shape=filter_list_enc[-1])(query=latent,key=hlayer,value=hlayer)  
            latent=tf.keras.layers.Add()([latent,latent_attn])
            latent=tf.keras.layers.LayerNormalization(axis=-1)(latent)

        def AveragePooling2D(hlayer_inputs=None,hlayer_outputs=None,pool_size=(3,3),padding='valid',use_resize=True):
            # Determine the stride size
            ps=hlayer_inputs.shape[1]//hlayer_outputs.shape[1]
            if use_resize:
                hlayer_inputs=tf.keras.layers.Resizing(hlayer_outputs.shape[1],hlayer_outputs.shape[2],interpolation='bicubic',crop_to_aspect_ratio=False)(hlayer_inputs)
            else:
                hlayer_inputs=tf.keras.layers.AveragePooling2D(pool_size=ps,strides=None, padding=padding,name=hlayer_inputs.name[:-10]+'_maxpooling_2D')(hlayer_inputs)
            if hlayer_inputs.shape[-1]!=hlayer_outputs.shape[-1]:
                
                hlayer_inputs=tf.keras.layers.Conv2D(hlayer_outputs.shape[-1], 1, strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Latent_Layer']['Kernel_Regu'],activity_regularizer=None,activation=residual_module_par['Latent_Layer']['Activation'],name=hlayer_inputs.name[:-10]+'reshape_maxpooling_2D')(hlayer_inputs)
            return hlayer_inputs
    # Decoder
    hlayer=decoder(inputs=inputs,depth=depth,width=width,latent=latent,skip_conn=skip_conn,residual_module_par=residual_module_par,name=name,nstack_idx=nstack_idx)
    # Reimaging of output
    if inputs.shape[1]==hlayer.shape[1]:
        hlayer=constantPadding_Layer(pad=[[0, 0], [1, 1], [1, 1],[0, 0]],layer_name=name+'_CNV2D_DEC_pad_output_layer_0')(hlayer)
        hlayer=tf.keras.layers.Conv2D(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/1), residual_module_par['Kernel_Size'], strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'_CNV2D_DEC_output_layer_0')(hlayer)
    else:
        hlayer=tf.keras.layers.Conv2D(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/1), residual_module_par['Kernel_Size'], strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'_CNV2D_DEC_output_layer_0')(hlayer)

    # Skip layer from input to output
    context_embed=tf.keras.layers.Conv2D(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/1), 1, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'_CNV2D_Context_Embed')(inputs)
    # context_embed=tfa.layers.MultiHeadAttention(head_size=8, num_heads=1,name=name+'_CNV2D_DEC_layer_attention'+str(1))([hlayer,context_embed]) 
    hlayer=tf.keras.layers.Add()([hlayer,context_embed])
    #hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2D_Context_Embed_activation_output_layer_0')(hlayer)
  
    #Dense Layer 
    for di in range(1):
        # Adds attention from input
        hlayer=tf.keras.layers.Dense(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/1), activation=residual_module_par['Activation_Func'], kernel_initializer=residual_module_par['Kernel_Init'], kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],name=name+'CNV2D_DEC_Dense_output_int_layer_0'+str(di+1))(hlayer)
        #hlayer=tf.keras.layers.Conv2D(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/2), residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'CNV2D_DEC_output_int_layer_0'+str(di+1))(hlayer)

    if residual_module_par['Skip_Connections']['Add_Input'][nstack_idx-1]:
        # Add permeability input
        #inputs_k=split_Layer(no_splits=inputs.shape[-1],axis=-1,layer_name='input_split_permx')(inputs)[-2]
        hlayer=tf.keras.layers.Conv2D(int(filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac']), 1, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,activation=None, kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'], name=name+'CNV2D_DEC_CNN_output_layer_0')(hlayer)
        skip_conn_proj_1=pad_stack_skip_conn(skip_conn=skip_conn[1], conv2d_hlayer=hlayer, name=name+'l1_output', depth_idx=1,residual_module_par=residual_module_par)
        #skip_conn_proj_1=pad_stack_skip_conn(skip_conn=inputs_k, conv2d_hlayer=hlayer, name=name+'l1_output', depth_idx=1,residual_module_par=residual_module_par)
        
        hlayer=tf.keras.layers.Add()([skip_conn_proj_1,hlayer])
        hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'CNV2D_DEC_Dense_activation_output_layer_0')(hlayer)

    #hlayer=tf.keras.layers.Conv2D(int((filter_list_enc[0]*residual_module_par['Decoder_Filter_Fac'])/2), 3, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=residual_module_par['Activation_Func'],name=name+'CNV2D_DEC_CNN_output_layer_1')(hlayer)
    olayer=tf.keras.layers.Conv2D(1, 1, strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Output_Layer'],activity_regularizer=residual_module_par['Kernel_Regu']['Output_Layer'],activation=None,name=name+'_CNV2D_DEC_final_output_layer')
    hlayer=olayer(hlayer)

    # Outer layer activation
    hlayer=tf.keras.layers.Activation(residual_module_par['Out_Activation_Func'],name=name+'_CNV2D_DEC_final_output_activation_layer',dtype=dt_type)(hlayer)
    hlayer,hlayer_w=hlayer,olayer.get_weights()
    return hlayer

def encoder_decoder_attn(inputs=None, depth=None,width=None,residual_module_par={},name=None,nstack_idx=None):
    filter_list_enc=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=depth,growth_rate=width['Growth_Rate'],growth_type='smooth',network_type='plain')              # Use of network_width_list function--ngens is equal to depth; growth type is set to smooth; network type as plain
    # Encoder
    if residual_module_par['Skip_Connections']['Add'][nstack_idx-1]:
        skip_conn={}
    for i in range(0,depth):
        hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=1, padding='same',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],activation=None, name=name+'_CNV2D_ENC_layer_'+str(i+1))(inputs)
        if residual_module_par['Skip_Connections']['Add'][nstack_idx-1] and residual_module_par['Skip_Connections']['Layers'][nstack_idx-1][i] not in [None,0]:
            skip_conn[i+1]=hlayer
        hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2D_ENC_activation_layer_'+str(i+1))(hlayer)
        
        if residual_module_par['Dropout']['Add'] in [True,'encoder'] and residual_module_par['Dropout']['Layer'][i]==1:
            hlayer=tf.keras.layers.Dropout(residual_module_par['Dropout']['Rate'], noise_shape=None, seed=None)(hlayer) 
    # Decoder
    


def plain_network(inputs=None,depth=None,width=None,activation_func=None,kernel_init=None,kernel_regu=None,nstack_idx=None,name=None,residual_module_par={}):
    # FLatten the 2D if used 
    #if dnn_arch=='plain2D':
    #    hlayer=tf.unstack(inputs, axis=-1,name=name+'_unstack_layer_'+str(i+1))

    width_list=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=width['No_Gens'][nstack_idx],growth_rate=width['Growth_Rate'],growth_type=width['Growth_Type'],network_type='plain')
    # breakpoint()
    for i in range(0,depth):
        if i==0:
            hlayer=tf.keras.layers.Dense(width_list[i], activation=None, kernel_initializer=kernel_init,kernel_regularizer=kernel_regu, name=name+'_hlayer_'+str(i+1))(inputs)
        else:
            if i!=depth-1:
                hlayer=tf.keras.layers.Dense(width_list[i], activation=None, kernel_initializer=kernel_init,kernel_regularizer=kernel_regu, name=name+'_hlayer_'+str(i+1))(hlayer)
            else:
                hlayer=tf.keras.layers.Dense(width_list[i], activation=None, kernel_initializer=kernel_init, kernel_regularizer=kernel_regu,name=name+'_hlayer_'+str(i+1))(hlayer)
        # Adds a Batch Normalization Layer:
        if residual_module_par['Batch_Norm']['Use_Batch_Norm']:
            if residual_module_par['Batch_Norm']['Before_Activation']:
                hlayer=tf.keras.layers.BatchNormalization(axis=1, momentum=residual_module_par['Batch_Norm']['Momentum'],epsilon=1e-6, name=name+'_batch_norm_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.Activation(activation_func,name=name+'_activation_'+str(i+1))(hlayer)
            else:
                hlayer=tf.keras.layers.Activation(activation_func,name=name+'_activation_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.BatchNormalization(axis=1, momentum=residual_module_par['Batch_Norm']['Momentum'], epsilon=1e-6, name=name+'_batch_norm_'+str(i+1))(hlayer)
        else:
            hlayer=tf.keras.layers.Activation(activation_func,name=name+'_activation_'+str(i+1))(hlayer)
        # Adds a Drop Out Layer (if active)
        if residual_module_par['Dropout']['Add'] and residual_module_par['Dropout']['Layer'][i] in [True,1] and len(residual_module_par['Dropout']['Layer'])>=depth:
            hlayer=tf.keras.layers.Dropout(residual_module_par['Dropout']['Rate'], noise_shape=None, seed=None)(hlayer)
        
    return hlayer
def dnn_sub_block(inputs=None,width={},depth=None,activation_func=None,olayer_act_func=None,kernel_init='glorot_normal',kernel_regu=None, name=None,residual_module_par={},layer_LSTM={'Add':False},layer_CNN={'Add':False}):
    # Residual module is/not contained in the dnn sub-block
    nstack_name=['pre_sat','pre','gsat','osat']
    if name[:-6] in nstack_name:
        nstack_idx=nstack_name.index(name[:-6])
    else:
        nstack_idx=1  # Set to pressure index
        
    if depth<=0:
        hlayer = inputs
    else:
        if residual_module_par['Network_Type']=='RESN':
            hlayers_=dnn1.RNN_Layer(depth=depth,width=width,res_par=residual_module_par,layer_name=name,gaussian_process=None,out_act_func=olayer_act_func,idx=nstack_idx,batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
            hlayer=hlayers_(inputs,output_layer=True)
        elif residual_module_par['Network_Type']=='LSTM':
            # Check if Dense layers are to be added inbetween the LSTM
            stack_list=LSTM_pad_list(depth=depth,dense_layers_inbetween=residual_module_par['Add_Dense_Inbetween_LSTM'])
            width_list=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=0,growth_rate=width['Growth_Rate'],growth_type=width['Growth_Type'],network_type='plain')  # Plain Network Type still used for LSTM
            for i in range(0,depth):
                if i==0:
                    if stack_list[i]!=0:
                        hlayer=tf.keras.layers.LSTM(width_list[i], activation=residual_module_par['Activation_Func'], recurrent_activation=residual_module_par['LSTM_Activation_Func'],use_bias=True, kernel_initializer=residual_module_par['Kernel_Init'],name=name+'_LSTM_layer_'+str(i+1))(tf.expand_dims(inputs,-1))
                    else:
                        hlayer=tf.keras.layers.Dense(width_list[i], activation=residual_module_par['Activation_Func'], kernel_initializer=residual_module_par['Kernel_Init'], name=name+'_Dense_layer_LSTM_'+str(i+1))(inputs)
                else:
                    if stack_list[i]!=0:
                        hlayer=tf.keras.layers.LSTM(width_list[i], activation=residual_module_par['Activation_Func'], recurrent_activation=residual_module_par['LSTM_Activation_Func'],use_bias=True, kernel_initializer=residual_module_par['Kernel_Init'],name=name+'_LSTM_layer_'+str(i+1))(tf.expand_dims(hlayer,-1))
                    else:
                        hlayer=tf.keras.layers.Dense(width_list[i], activation=residual_module_par['Activation_Func'], kernel_initializer=residual_module_par['Kernel_Init'], name=name+'_Dense_layer_LSTM_'+str(i+1))(hlayer)
        elif residual_module_par['Network_Type']=='CNN2D':
            stack_list=LSTM_pad_list(depth=depth,dense_layers_inbetween=False)
            #width_list=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=0,growth_rate=width['Growth_Rate'],growth_type=width['Growth_Type'],network_type='plain')  # Plain Network Type still used for LSTM
            hlayer=encoder_decoder(inputs=inputs,depth=depth,width=width,name=name,nstack_idx=nstack_idx,residual_module_par=residual_module_par)
        elif residual_module_par['Network_Type']=='ATTN':
            hlayer=encoder_decoder_attn(inputs=inputs,depth=depth,width=width,name=name,nstack_idx=nstack_idx,residual_module_par=residual_module_par)
        else:  #Plain Network
            hlayer=plain_network(inputs=inputs,depth=depth,width=width,activation_func=activation_func,kernel_init=kernel_init,kernel_regu=kernel_regu,nstack_idx=nstack_idx,name=name,residual_module_par=residual_module_par)
        # ===============================================================================================================================
        # Add a Memory/CNN layer before the output
        if layer_LSTM['Add']:
            hlayer=tf.keras.layers.Reshape((1,hlayer.shape[-1]))(hlayer)
            #batch_input_shape=(841, hlayer.shape[1],hlayer.shape[2])
            hlayer=tf.keras.layers.LSTM(layer_LSTM['Units'], activation=layer_LSTM['Activation_Func'], recurrent_activation=layer_LSTM['Recur_Activation_Func'],use_bias=layer_LSTM['Bias'], kernel_initializer=kernel_init,stateful=layer_LSTM['Stateful'],name=name+'_LSTM_layer_'+str(i+1))(hlayer)
        if layer_CNN['Add'] and layer_CNN['Type']=='1D':
            hlayer=tf.keras.layers.Reshape((hlayer.shape[-1],1))(hlayer)
            hlayer=tf.keras.layers.Conv1D(layer_CNN['Filter'], layer_CNN['Kernel_Size'], strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=kernel_init,activation=layer_CNN['Activation_Func'],name=name+'_CNV1D_layer_'+str(i+1))(hlayer)
            hlayer=tf.keras.layers.Flatten()(hlayer)
        #hlayer=tf.keras.layers.GaussianNoise(0.005)(hlayer)
        #===============================================================================================================================
    return hlayer
def CNN_dense(inputs=None, depth=None,width=None,residual_module_par={},name=None):
    depth-=1
    filter_list_enc=network_width_list(depth=depth,width=width['Bottom_Size'],ngens=depth,growth_rate=width['Growth_Rate'],growth_type='smooth',network_type='plain')
    for i in range(0,depth):
        if i==0:
            hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=1, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],activation=None, name=name+'_CNV2D_layer_'+str(i+1))(inputs)
        else:
            if i<depth-1:
                # Pads the tensor
                hlayer=tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1)), data_format=None, name=name+'_CNV2D_pad_layer_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size']+2, strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_ENC_layer_'+str(i+1))(hlayer)
            else:
                hlayer=tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1)), data_format=None, name=name+'_CNV2D_pad_layer_'+str(i+1))(hlayer)
                hlayer=tf.keras.layers.Conv2D(filter_list_enc[i], residual_module_par['Kernel_Size'], strides=2, padding='valid',data_format='channels_last', dilation_rate=1, groups=1,kernel_initializer=residual_module_par['Kernel_Init'],kernel_regularizer=residual_module_par['Kernel_Regu']['Hidden_Layer'],activation=None,name=name+'_CNV2D_ENC_layer_'+str(i+1))(hlayer)
            # Add convolution layer
        hlayer=tf.keras.layers.Activation(residual_module_par['Activation_Func'],name=name+'_CNV2D_ENC_activation_layer_'+str(i+1))(hlayer)
    else:
        return hlayer         
def sparse_pad_list(depth=None,nstacks=None):
    if nstacks>depth:
        nstacks=depth
    elif nstacks==0:
        nstacks=1
    no_per_stack=int(depth/nstacks)
    rem_stack=int(depth%nstacks)
    new_list=[]
    for i in range(int(nstacks)):
        if i==0:
            stack=[no_per_stack+rem_stack]+(no_per_stack+rem_stack-1)*[0]
        else:
            stack=[no_per_stack]+(no_per_stack-1)*[0]
        new_list=new_list+stack
    return new_list

def LSTM_pad_list(depth=None,dense_layers_inbetween=True):
    pad_list=depth*[1]
    if dense_layers_inbetween:
        for i in range(1,depth,2):
            pad_list[i]=0
    pad_list[depth-1]=0
    return pad_list
            
def network_width_list(width=None,depth=None,ngens=None,network_type='plain',growth_type='smooth',growth_rate=0.5):
    import numpy as np
    # network_type: 'plain' | 'resn'
    # growth_type: 'smooth' | 'cross'
    # resnet usually dont have a cross growth
    def create_even(num):
        return int(num/2)*2
    if ngens==None or ngens==0:
        ngens=1
    elif ngens>depth:
        ngens=depth

    no_per_gen=int(depth/ngens)
    rem_gen=int(depth%ngens)
    new_list=[] 
    for i in range(int(ngens)):
        if i==0:
            if network_type=='plain':
                gen=(no_per_gen+(rem_gen//2))*[(1*growth_rate**(i))]
            else:
                gen=[(1*growth_rate**(i))]+(no_per_gen+(rem_gen)-1)*[0]
        else:
            if network_type=='plain':
                if growth_type=='smooth':
                    gen=(no_per_gen)*[(1*growth_rate**(i))]
                else:
                    if i<=int(ngens/2):
                        gen=(no_per_gen)*[(1*growth_rate**(i))] 
                    else:
                        gen=(no_per_gen)*[(1*(growth_rate)**(2*int(ngens/2)-i))]
                if i==int(ngens)-1:
                    gen=gen+((rem_gen//2)+(rem_gen%2))*[(1*growth_rate**(0))]
            else:
                if growth_type=='smooth':
                    gen=[(1*growth_rate**(i))]+(no_per_gen-1)*[0]
                else:
                    if i<=int(ngens/2):
                        gen=[(1*growth_rate**(i))]+(no_per_gen-1)*[0]
                    else:
                        gen=[(1*(growth_rate)**(2*int(ngens/2)-i))]+(no_per_gen-1)*[0]

        new_list=new_list+gen
        
    new_list=[int(create_even(np.ceil(i*width))) for i in new_list]
    return new_list
#==============================================================================================================================================================================================================================
# Randomly Initialize variables
def make_rand_variables(initializer='Random_Normal',var_shape=(),dist_mean=1.0,dist_std=0.2,norm_output=True,var_trainable=False):
    if initializer=='Random_Normal':
        rand_var=tf.random_normal_initializer(mean=tf.cast(dist_mean,dt_type),stddev=dist_std)(shape=var_shape,dtype=dt_type)
    else:
        rand_var=dist_mean
    if norm_output==True:
        nrand_var=tf.linalg.normalize(rand_var,ord=1)[0]
    else:
        nrand_var=rand_var
    nrand_var=np.nan_to_num(nrand_var)
    var=tf.Variable(nrand_var,dtype=dt_type,trainable=var_trainable)
    return var

class Truncation_Error(tf.keras.layers.Layer):
    def __init__(self, initial_value=0.00012,units=1,**kwargs):
        super(Truncation_Error, self).__init__(**kwargs)
        self.initial_value=initial_value
        self.units=units
            
    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.keras.initializers.Constant(self.initial_value)
        b_init = tf.zeros_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),dtype=self.dtype), trainable=True)
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype=self.dtype), trainable=True)
        
        #self.w = self.add_weight(shape=(input_shape[-1], self.units),initializer=tf.constant_initializer(self.initial_value),trainable=True)
        #self.b = self.add_weight(shape=(self.units,),initializer='zeros',trainable=True)
  
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
# Create the model class--inherits the keras.model class
class pinn_Model(tf.keras.Model):
    def __init__(self,ts,cfd_type,*args, **kwargs):
        super().__init__( *args, **kwargs)
        # ts is a 2D tensor representing the training statistics. 
        # ts rows is defined by: ['x_coord', 'y_coord', 'z_coord', 'Time', 'Poro', 'PermX', 'PermZ', 'Mobility', 
                                    # 'Tcompressibility', 'GasFVF','Gas Production Rate (Mscf/D)']
        # ts columns is defined by: ['min','max','mean','std','count']
        
        # Get the CFD type
        self.cfd_type=cfd_type                  # Dictionary of the 'Type':'PINN', 'noPINN'; 'Dimension': '2D', '3D'
        self.ts=tf.Variable(tf.cast(ts[0],dtype=self.dtype),trainable=False)           # Training statistics--if value is None, no normalization is done
        self.ts_idx_keys=[ts[1],ts[2]]
        #=========================================================================================================================================
        self.phi_0=self.ts[4,2]                  # Mean porosity of the training data      
        # Compute the rock/formation compressibility using Newman, G. H. (1973) correlation for consolidated sandstones
        self.cf=97.32e-6/(1+55.8721*self.phi_0**1.428586)
        self.grad=[]
        self.loss_func={}   #Loss functions 
        # Create a variable for the cumulative rate
        self.cum={key:tf.Variable(0, dtype=self.dtype, trainable=False) for key in ['Gas_Pred','Gas_Obs','Cum_N']}
        self.batch_seed_no={'numpy':0} #'tf':tf.Variable(0,trainable=False,dtype=self.dtype)
        self.eps_pss={'Average':tf.keras.metrics.Mean(name="mean_eps",dtype=self.dtype),'Value':0.}
        self.alpha=tf.Variable(1., dtype=self.dtype, trainable=False)
        self.timestep=tf.Variable(1., dtype=self.dtype, trainable=False)
        self.timesteps=tf.constant(0.,shape=[],dtype=self.dtype)
        #self.kr_gas_oil=relative_permeability(endpoints=cfd_type['SCAL']['End_Points'],corey_exp=cfd_type['SCAL']['Corey_Exp'])
        #self.rnd_gen=tf.random.Generator.from_seed(cfd_type['Seed']).split(count=2)
        #self.mbc_wt=tf.Variable(1., dtype=self.dtype, trainable=False)
        #self.tstep=tf.Variable(cfd_type['Timestep'],trainable=False,dtype=self.dtype)
        #self.q_n1_ij=tf.Variable(0.,shape=tf.TensorShape(None),trainable=False,dtype=self.dtype,validate_shape=False)
        #====Truncation Error -- Time
        #self.inp_M=tf.keras.Input(shape=self.cfd_type['Dimension']['Dim'], name='Error_Input')
        #self.layer_Mt=Truncation_Error(initial_value=0.00015,name='Truncation_Error_Mt')
        #self.layer_Mt(self.inp_M)
        #inpt=tf.keras.Input(shape=(1,), name='Error_Input')
        #self.layer_MBC=tf.keras.layers.Dense()
        
        #self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(25.), trainable=True,name='tpss',dtype=self.dtype)
        
        # Create a ID index stitch for the gradients to used with dynamic stitch. Initialize the gradients accumulator
        self.idx=bl.convert_1D(self).idx
        if self.cfd_type['Accum_Grad']['Add']==True:
            self.n_grad_steps=tf.constant(0, dtype=tf.int32)
            self.accum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=dt_type), trainable=False) for v in self.trainable_variables]
        #=========================================================================================================================================
        # Create a dictionary to hold the best model weight-biases based on a selected epoch obseravation range--using callback; also the train time
        self.wbl_epoch=[]
        self.wblt_epoch_ens=[]                          # Best tuning parameters for each realization/perturbation
        self.history_ens=[]
        self.best_ens=0
        #=========================================================================================================================================
        # Define the loss metrics for the non-physics training
        if self.cfd_type['Data_Arr']!=3:
            self.dom_loss = tf.keras.metrics.Mean(name="dom_loss")
            self.dbc_loss = tf.keras.metrics.Mean(name="dbc_loss")
            self.nbc_loss = tf.keras.metrics.Mean(name="nbc_loss")
            self.ibc_loss = tf.keras.metrics.Mean(name="ibc_loss")
            self.ic_loss = tf.keras.metrics.Mean(name="ic_loss")
        self.total_loss = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.val_loss_tracker = tf.keras.metrics.MeanSquaredError(name="val_loss")
        self.val_mae_metric = tf.keras.metrics.MeanAbsoluteError(name="val_mae")
        
        # Use the mean metric to hold norm values
        self.grad_norm=tf.keras.metrics.Mean(name='gradient_norm')
        #=========================================================================================================================================
        # Define a weighting parameter (lm) for each loss and make it trainable. Starting value is given by: lm(i)=1/(no. of losses)
        if self.cfd_type['Type']=='PINN':
            # [DOM, DBC, NBC, IBC, IC, TD]
            self.mbc_loss = tf.keras.metrics.Mean(name="mbc_loss")
            self.cmbc_loss = tf.keras.metrics.Mean(name="cmbc_loss")
            self.loss_func['Batch_Loss']=bl.pinn_batch_sse_grad
            if self.cfd_type['DNN_Arch'] in ['resn','plain','']:
                self.loss_func['Squeeze_Out']=tf.squeeze
                self.loss_func['Reshape']=lambda x:tf.reshape(x,self.cfd_type['Dimension']['Reshape'])
                self.loss_func['Reduce_Axis']=[1,2,3]
            else:
                self.loss_func['Squeeze_Out']=lambda x:x # Output is not squeezed
                self.loss_func['Reshape']=lambda x:x
                self.loss_func['Reduce_Axis']=[1,2,3,4]
            self.lbl_idx=15
            if self.cfd_type['Fluid_Type'] in ['dry-gas','DG']:
                self.loss_func['Physics_Error']=bl.physics_error_gas_2D
                self.nwt=make_rand_variables(initializer='Random_Normal',var_shape=[9],dist_mean=cfd_type['Solu_Weights'],dist_std=7*[0.]+[0.,0.],norm_output=True,var_trainable=False)        # DOM, DBC, NBC, IBC, IC
                self.nwt_test=make_rand_variables(initializer='Random_Normal',var_shape=[2],dist_mean=[1.,0.],dist_std=0.,norm_output=True,var_trainable=False)        # DOM, DBC, NBC, IBC, IC
                self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td_loss_sg")}
            else:
                self.loss_func['Physics_Error']=bl.physics_error_gas_oil_2D
                self.nwt=make_rand_variables(initializer='Random_Normal',var_shape=[10],dist_mean=cfd_type['Solu_Weights'],dist_std=7*[0.]+[0.,0.,0.],norm_output=True,var_trainable=False)        # DOM, DBC, NBC, IBC, IC
                self.nwt_test=make_rand_variables(initializer='Random_Normal',var_shape=[3],dist_mean=[1.,0.,0.],dist_std=0.,norm_output=True,var_trainable=False)        # DOM, DBC, NBC, IBC, IC
                self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td_loss_sg"),'so':tf.keras.metrics.Mean(name="td_loss_so")}            
        else:
            self.loss_func['Batch_Loss']=bl.pinn_batch_sse_grad    
            self.loss_func['Physics_Error']=bl.zeros_like_pinn_error
            if self.cfd_type['Data_Arr']==0:
                if self.cfd_type['Fluid_Type'] in ['dry-gas','DG']:
                    # Set a label index for the input--18 for dry gas and 21 for gas-condensate data
                    self.lbl_idx=16
                    # Constitutive relationship and QoIs--pressure, gas saturation
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[1],dist_mean=1.0,dist_std=0.0,norm_output=True,var_trainable=False)  
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td_loss_sg")}
                else:
                    self.lbl_idx=17
                    # Constitutive relationship and QoIs--pressure, gas saturation, oil saturation
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[3],dist_mean=1.0,dist_std=0.0,norm_output=True,var_trainable=False)  
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td _loss_sg"),'so':tf.keras.metrics.Mean(name="td_loss_so")}
            elif self.cfd_type['Data_Arr']==1:
                if self.cfd_type['DNN_Arch']=='cnn2d':
                    self.loss_func['Reduce_Axis']=[1,2,3,4]
                    self.loss_func['Reshape']=lambda x:x
                    self.loss_func['Squeeze_Out']=lambda x:x
                    #self.loss_func['Batch_Loss']=bl.nopinn_batch_sse_grad_1
                else:
                    self.loss_func['Reduce_Axis']=[1,]
                    self.loss_func['Reshape']=lambda x:tf.reshape(x,[-1])
                    self.loss_func['Squeeze_Out']=tf.squeeze
                    #self.loss_func['Batch_Loss']=bl.nopinn_batch_sse_grad_1
                self.lbl_idx=15
                if self.cfd_type['Fluid_Type'] in ['dry-gas','DG']:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[2],dist_mean=cfd_type['Solu_Weights'][7:],dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td_loss_sg")}
                    if self.cfd_type['Aux_Layer']['Bu']['Use']:
                        nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[4],dist_mean=1*[1.]+1*[1.]+2*[0.],dist_std=0.0,norm_output=True,var_trainable=False)
                        self.td_loss.update({'Bg':tf.keras.metrics.Mean(name="td_loss_Bg"),'ug':tf.keras.metrics.Mean(name="td_loss_ug")})
                else:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[3],dist_mean=cfd_type['Solu_Weights'][7:],dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sg':tf.keras.metrics.Mean(name="td_loss_sg"),'so':tf.keras.metrics.Mean(name="td_loss_so")}
                    if self.cfd_type['Aux_Layer']['Bu']['Use']:
                        nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[7],dist_mean=1*[1.]+2*[1.]+4*[0.],dist_std=0.0,norm_output=True,var_trainable=False)
                        self.td_loss.update({'Bg':tf.keras.metrics.Mean(name="td_loss_Bg"),'Bo':tf.keras.metrics.Mean(name="td_loss_Bo"),\
                                   'ug':tf.keras.metrics.Mean(name="td_loss_ug"),'uo':tf.keras.metrics.Mean(name="td_loss_uo")})

            elif self.cfd_type['Data_Arr']==2:
                self.loss_func['Batch_Loss']=bl.nopinn_batch_sse_grad_1
                self.lbl_idx=15
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[2],dist_mean=1.0,dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sbu_g':tf.keras.metrics.Mean(name="td_loss_sbu_g")}
                else:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[3],dist_mean=1.0,dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'p':tf.keras.metrics.Mean(name="td_loss_p"),'sbu_g':tf.keras.metrics.Mean(name="td_loss_sbu_g"),'sbu_o':tf.keras.metrics.Mean(name="td_loss_sbu_o")}

            elif self.cfd_type['Data_Arr']==3:                      # Special Kind of arrangement for pretraining the PVT data
                self.loss_func['Batch_Loss']=bl.nopinn_batch_sse_grad_pvt
                self.lbl_idx=15
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[2],dist_mean=cfd_type['Solu_Weights'],dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'Bg':tf.keras.metrics.Mean(name="td_loss_Bg"),'ug':tf.keras.metrics.Mean(name="td_loss_ug")}
                else:
                    nrwt=make_rand_variables(initializer='Random_Normal',var_shape=[6],dist_mean=cfd_type['Solu_Weights'],dist_std=0.0,norm_output=True,var_trainable=False)
                    # Create additional metrics for other labels 
                    self.td_loss={'Bg':tf.keras.metrics.Mean(name="td_loss_Bg"),'Bo':tf.keras.metrics.Mean(name="td_loss_Bo"),\
                                   'ug':tf.keras.metrics.Mean(name="td_loss_ug"),'uo':tf.keras.metrics.Mean(name="td_loss_uo"),\
                                    'Rs':tf.keras.metrics.Mean(name="td_loss_Rs"),'Rv':tf.keras.metrics.Mean(name="td_loss_Rv")}
              
            # Pad with zero-based vector of shape=5 to represent the empty DOM, DBC, NBC, IBC, IC solution sets.
            self.nwt=tf.concat([tf.zeros([7],dtype=dt_type),nrwt],axis=0)
        
        #=========================================================================================================================================
        # Solution label index settings        
        self.solu_idx={'DOM':0,'DBC':1,'NBC':2,'IBC':3,'IC':4,'TD':5,'TDA':6} 
        if self.cfd_type['Fluid_Type']in ['dry-gas','DG']:        
            self.nT=2                   # Number of predictor terms
            self.nT_list=[0,1]
            self.aux_models={'sg':None}
        else:
            self.nT=3
            self.nT_list=[0,1,2]
            self.models={'sg':None,'so':None}
        
        global_seed_gen = tf.random.get_global_generator()
        global_seed_gen.reset_from_seed(self.batch_seed_no['numpy'])
            
    # Override the __train_step__ method
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        # Set the batch seed number and set for any random distribution -- adding noising effect
        #tf.random.set_seed(self.batch_seed_no['numpy'])
                
        # Perform batch PINN calculation using the pinn_batch_sse_gradfunction
        #batch_sse_grad=tf.cond(tf.math.equal(self.cfd_type['Type'],'PINN'),lambda: bl.pinn_batch_sse_grad(self,x,y),lambda: bl.nopinn_batch_sse_grad_0(self,x,y))
        batch_sse_grad=self.loss_func['Batch_Loss'](self,x,y)
        
        # Update the batch seed number
        #self.batch_seed_no['tf'].assign_add((self.cfd_type['Seed'])*0.5)
    
        # The return value using the nopinn_batch_sse_grad function is a list with indexing:
        # [0]: Weighted SSE loss list=[batch,DOM,DBC,NBC,IBC,IC,TD]
        # [1]: Weighted SSE gradient list=[batch,DOM,DBC,NBC,IBC,IC,TD]
        # [2]: Error count list=[batch,DOM,DBC,NBC,IBC,IC,TD]
        # [3]: Weighted MSE loss list=[batch,DOM,DBC,NBC,IBC,IC,TD]
        # [4]: Model's output
        
        # Compute the individual solution gradient based on MSE
        #Training data is at the nth list
        no_terms=len(batch_sse_grad[2])
               
        # Compute the PINN (if any) MSE gradients
        _wmse_grad=[[gradients/bl.zeros_to_ones(batch_sse_grad[2][i+1]) for gradients in batch_sse_grad[1][i+1]] for i in range(no_terms-1)]
        """
        if self.cfd_type['Data_Arr']!=3:
            batch_wmse_grad=[(a+b+c+d+e+f+g) for a,b,c,d,e,f,g in zip(_wmse_grad[0],_wmse_grad[1],_wmse_grad[2],_wmse_grad[3],_wmse_grad[4],_wmse_grad[5],_wmse_grad[6])] 
        else:
            batch_wmse_grad=_wmse_grad[0]"""
        #dom_norm=[tf.norm(t,ord='euclidean') for t in _wmse_grad[0]]
        #_wmse_grad[-3]=[tf.clip_by_norm(_wmse_grad[-3][t],1000.) for t in range(len(_wmse_grad[-3]))]
        #mbc_norm=[tf.norm(t,ord='euclidean') for t in _wmse_grad[-3]]
        #_wmse_grad[0]=[tf.clip_by_norm(gradient,tf.reduce_max(mbc_norm)) for gradient in _wmse_grad[0]]
        
        if self.cfd_type['Type'].upper()=='PINN':
            batch_wmse_grad=[tf.math.reduce_mean(i,axis=0) for i in zip(*_wmse_grad)] 
        else:
             batch_wmse_grad=[tf.math.reduce_sum(i,axis=0) for i in zip(*_wmse_grad)] 
            
        aggregate_grads_outside_optimizer = (
        self.optimizer._HAS_AGGREGATE_GRAD and  # pylint: disable=protected-access
        not isinstance(self.distribute_strategy.extended,tf.distribute.StrategyExtended))
        
        #if aggregate_grads_outside_optimizer:
            # We aggregate gradients before unscaling them, in case a subclass of
            # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
            # done on scaled gradients, not unscaled gradients, for numeric stability.
            #batch_rgl_mse_grad = self.optimizer._aggregate_gradients(zip(batch_rgl_mse_grad, self.trainable_variables))    # pylint: disable=protected-access    
        
        #if isinstance(self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer):
            #batch_rgl_mse_grad = self.optimizer.get_unscaled_gradients(batch_rgl_mse_grad)
        
        # Clips the gradient
        #_norm=tf.math.divide(tf.norm(tf.dynamic_stitch(self.idx,batch_wmse_grad),ord='euclidean'),tf.math.sqrt(tf.cast(len(batch_wmse_grad),dtype=dt_type)))
        
        #batch_wmse_grad=tf.cond(tf.math.greater_equal(_norm, self.cfd_type['Gradient_Norm']),\
                                  #lambda: [tf.clip_by_norm(gradients,self.cfd_type['Gradient_Norm']) for gradients in batch_wmse_grad],lambda: batch_wmse_grad)
        #batch_rgl_mse_grad = self.optimizer._clip_gradients(batch_rgl_mse_grad)  
        batch_wmse_grad_norm=[tf.norm(t,ord='euclidean') for t in batch_wmse_grad]
        
        #batch_wmse_grad_norm_norm=tf.linalg.normalize(batch_wmse_grad_norm) 

        #global_norm = tf.math.sqrt(tf.reduce_sum([tn**2 for tn in batch_wmse_grad_norm]))
        
        #batch_wmse_grad=[batch_wmse_grad[i]*tf.math.minimum(batch_wmse_grad_norm[i],(self.cfd_type['Gradient_Norm']))/batch_wmse_grad_norm[i] for i in range(len(batch_wmse_grad))]
        #batch_wmse_grad_norm=[t*self.cfd_type['Gradient_Norm']/tf.math.maximum(global_norm, self.cfd_type['Gradient_Norm']) for t in batch_wmse_grad]
        #batch_wmse_grad=[tf.math.divide_no_nan(batch_wmse_grad[i],batch_wmse_grad_norm[i])*batch_wmse_grad_norm_norm[0][i] for i in range(len(batch_wmse_grad))]

        #batch_wmse_grad=[tf.where(tf.math.is_nan(batch_wmse_grad[i]),tf.zeros_like(batch_wmse_grad[i]),batch_wmse_grad[i]) for i in range(len(batch_wmse_grad))]
        #[((tf.cast(batch_wmse_grad[i]<=100000.,self.dtype)*batch_wmse_grad[i])+(tf.cast(batch_wmse_grad[i]>100000.,self.dtype)*100000.)) for i in range(len(batch_wmse_grad))]
        
        max_mbc=[tf.reduce_max(t) for t in _wmse_grad[-3]]
        
        #tf.print('\nMax_dom\n',max_dom,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/dom.out" )
        #tf.print('\nMax_mbc\n',max_mbc,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/mbc.out" )

        #tf.print('Min_T',output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
        #tf.print(min_t,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
        #tf.print('\n',output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )

        
        # Accumulate batch gradients
        if self.cfd_type['Accum_Grad']['Add']==True:
            self.accum_grad(batch_wmse_grad)
        else:
            self.optimizer.apply_gradients(zip(batch_wmse_grad, self.trainable_variables)) 

        # Model's output --pressure and/or saturation 
        y_pred=batch_sse_grad[4]

        self.total_loss.update_state(batch_sse_grad[3][0])
        # Compute the global norm
        #self.grad_norm.update_state(tf.math.divide(tf.norm(tf.dynamic_stitch(self.idx,batch_wmse_grad),ord='euclidean'),tf.math.sqrt(tf.cast(len(batch_wmse_grad),dtype=dt_type))))
        self.grad_norm.update_state(tf.math.divide(tf.linalg.global_norm(batch_wmse_grad),tf.math.sqrt(tf.cast(len(batch_wmse_grad),dtype=dt_type))))

        # MAE on the pressure and saturation (QoIs) and [permeability (Constitutive relationship)]
        dict_map={"loss": self.total_loss.result(), "mae": self.mae_metric.result()} 
        
        # Compute PINN-defined metrics--MSE        
        if self.cfd_type['Data_Arr']!=3:
            self.dom_loss.update_state(batch_sse_grad[3][1])
            self.dbc_loss.update_state(batch_sse_grad[3][2])
            self.nbc_loss.update_state(batch_sse_grad[3][3])
            self.ibc_loss.update_state(batch_sse_grad[3][4])
            self.ic_loss.update_state(batch_sse_grad[3][5])
            dict_map.update({"dom_loss": self.dom_loss.result(),"dbc_loss": self.dbc_loss.result(),\
                "nbc_loss": self.nbc_loss.result(),"ibc_loss": self.ibc_loss.result(),"ic_loss": self.ic_loss.result()})
            
        if self.cfd_type['Type']=='PINN' or self.cfd_type['Data_Arr']==0:
            # OUTPUTS: p,sg and/or so
            self.mbc_loss.update_state(batch_sse_grad[3][6])
            self.cmbc_loss.update_state(batch_sse_grad[3][7])
            dict_map.update({"mbc_loss":self.mbc_loss.result(),"cmbc_loss":self.cmbc_loss.result()})
            if self.cfd_type['Fluid_Type']in ['dry-gas','DG']: 
                self.mae_metric.update_state([y[0],y[1]],[y_pred[0],y_pred[1]],sample_weight=self.nwt[7:9])
                self.td_loss['p'].update_state(batch_sse_grad[3][8][0])
                self.td_loss['sg'].update_state(batch_sse_grad[3][8][1])
                dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"grad_norm":self.grad_norm.result()})
            else: 
                self.mae_metric.update_state([y[0],y[1],y[2]],[y_pred[0],y_pred[1],y_pred[2]],sample_weight=self.nwt[7:10])  #To be updated
                self.td_loss['p'].update_state(batch_sse_grad[3][8][0])
                self.td_loss['sg'].update_state(batch_sse_grad[3][8][1])
                self.td_loss['so'].update_state(batch_sse_grad[3][8][2])
                dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"td_loss_so": self.td_loss['so'].result(),"grad_norm":self.grad_norm.result()})
        else:
            if self.cfd_type['Data_Arr']==1:
                nT=tf.shape(y_pred)[0]
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']: 
                    # OUTPUTS: p,sg,Bg,ug ...[qg]
                    self.td_loss['p'].update_state(batch_sse_grad[3][8][0])
                    self.td_loss['sg'].update_state(batch_sse_grad[3][8][1])
                    self.mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[5:5+nT])
                    #self.mae_metric.update_state([y[0],y[1],y[2],y[3]],[y_pred[0],y_pred[1],y_pred[2],y_pred[3]],sample_weight=self.nwt[5:nT])
                    if self.cfd_type['Aux_Layer']['Bu']['Use']:
                        self.td_loss['Bg'].update_state(batch_sse_grad[3][8][2])
                        self.td_loss['ug'].update_state(batch_sse_grad[3][8][3])
                        dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"td_loss_Bg": self.td_loss['Bg'].result(),\
                                     "td_loss_ug": self.td_loss['ug'].result(),"grad_norm":self.grad_norm.result()})
                    else:
                        dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"grad_norm":self.grad_norm.result()})
                else:
                    # OUTPUTS: p,so,sg,Bg,Bo,ug,uo, ...[qg,qo]
                    self.td_loss['p'].update_state(batch_sse_grad[3][8][0])
                    self.td_loss['sg'].update_state(batch_sse_grad[3][8][1])
                    self.td_loss['so'].update_state(batch_sse_grad[3][8][2])
                    self.mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[7:7+nT])
                    if self.cfd_type['Aux_Layer']['Bu']['Use']:
                        self.td_loss['Bg'].update_state(batch_sse_grad[3][8][3])
                        self.td_loss['Bo'].update_state(batch_sse_grad[3][8][4])
                        self.td_loss['ug'].update_state(batch_sse_grad[3][8][5])
                        self.td_loss['uo'].update_state(batch_sse_grad[3][8][6])
                        dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"td_loss_so": self.td_loss['so'].result(),"td_loss_Bg": self.td_loss['Bg'].result(),\
                                     "td_loss_Bo": self.td_loss['Bo'].result(),"td_loss_ug": self.td_loss['ug'].result(),"td_loss_uo": self.td_loss['uo'].result(),"grad_norm":self.grad_norm.result()})
                    else:
                        dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sg": self.td_loss['sg'].result(),"td_loss_so": self.td_loss['so'].result(),"grad_norm":self.grad_norm.result()})

            elif self.cfd_type['Data_Arr']==2:
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']:  
                    # OUTPUTS: p,sbu_g
                    self.mae_metric.update_state([y[0],y[1]],[y_pred[0],y_pred[1]],sample_weight=self.nwt[5:7])
                    self.td_loss['p'].update_state(batch_sse_grad[3][6][0])
                    self.td_loss['sbu_g'].update_state(batch_sse_grad[3][6][1])
                    dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sbu_g": self.td_loss['sbu_g'].result(),"grad_norm":self.grad_norm.result()})
                else:
                    # OUTPUTS: p,sbu_g,sbu_o
                    self.mae_metric.update_state([y[0],y[1],y[2]],[y_pred[0],y_pred[1],y_pred[2]],sample_weight=self.nwt[5:8])
                    self.td_loss['p'].update_state(batch_sse_grad[3][6][0])
                    self.td_loss['sbu_g'].update_state(batch_sse_grad[3][6][1])
                    self.td_loss['sbu_o'].update_state(batch_sse_grad[3][6][2])
                    dict_map.update({"td_loss_p": self.td_loss['p'].result(),"td_loss_sbu_g": self.td_loss['sbu_g'].result(),"td_loss_sbu_o": self.td_loss['sbu_o'].result(),"grad_norm":self.grad_norm.result()})
            elif self.cfd_type['Data_Arr']==3:               # Special form for PVT pretraining
                nT=tf.shape(y_pred)[0]
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']: 
                    # OUTPUTS: Bg,ug
                    self.td_loss['Bg'].update_state(batch_sse_grad[3][1][0])
                    self.td_loss['ug'].update_state(batch_sse_grad[3][1][1])
                    self.mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[5:5+nT])
                    dict_map.update({"td_loss_Bg": self.td_loss['Bg'].result(),"td_loss_ug": self.td_loss['ug'].result(),"grad_norm":self.grad_norm.result()})
                else:
                    # OUTPUTS: Bg,Bo,ug,uo
                    self.td_loss['Bg'].update_state(batch_sse_grad[3][1][0])
                    self.td_loss['Bo'].update_state(batch_sse_grad[3][1][1])
                    self.td_loss['ug'].update_state(batch_sse_grad[3][1][2])
                    self.td_loss['uo'].update_state(batch_sse_grad[3][1][3])
                    self.td_loss['Rs'].update_state(batch_sse_grad[3][1][4])
                    self.td_loss['Rv'].update_state(batch_sse_grad[3][1][5])
                    self.mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[5:5+nT])
                    dict_map.update({"td_loss_Bg": self.td_loss['Bg'].result(),"td_loss_Bo": self.td_loss['Bo'].result(),"td_loss_ug": self.td_loss['ug'].result(),\
                                     "td_loss_uo": self.td_loss['uo'].result(),"td_loss_Rs": self.td_loss['Rs'].result(),"td_loss_Rv": self.td_loss['Rv'].result(),"grad_norm":self.grad_norm.result()})
   
        # Return a dict mapping metric names to current value--i.e., Mean
        return dict_map
    
    def accum_grad(self, grad):
        self.accum_step.assign_add(1)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(grad[i])
        
        # If accum_step has reached the n_grad_steps, then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.math.equal(self.accum_step, self.n_grad_steps), self.apply_accu_gradients, lambda: None)
        
    def apply_accu_gradients(self):
        # Apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))
        # Reset the step and accumulated gradient values, and other cumulative variables 
        self.accum_step.assign(0)
        [self.cum[key].assign(0) for key in ['Gas_Pred','Gas_Obs','Cum_N']]

        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=dt_type))    
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        
        # Updates the metrics tracking the loss--could be scaled further/cleaner
        if self.cfd_type['Type']=='PINN' or self.cfd_type['Data_Arr']==0:
            if self.cfd_type['Fluid_Type'] in ['dry-gas','DG']:
                nT=2#tf.shape(self.outputs)[0]._inferred_value[0]-5  # Exclude gsat,1/Bg,1/ug,qg,s1,qg_opt,pwf--tf.shape is zero indexed
            else:
                nT=3#tf.shape(self.outputs)[0]._inferred_value[0]-10  # Exclude gsat,osat,1/Bg,1/Bo,1/ug,1/uo,qg,qo,s1,s2,dPVT...
            
            if self.cfd_type['Val_Data_Label']:
                y_pred = self(x, training=False)
                self.val_loss_tracker.update_state(tf.stack(y)[0:nT],tf.stack(y_pred[0:nT],0),sample_weight=self.nwt_test[0:nT])
                self.val_mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred[0:nT],0),sample_weight=self.nwt_test[0:nT])
            else:
                #=====================================IC Solution======================================================= 
                # Forward pass is used as it still the model solution but at initial condition (t=0) 
                xn_t0=list(x)
                xn_t0[3]=self.cfd_type['Norm_Limits'][0]*tf.ones_like(xn_t0[0])
                y_pred=self(xn_t0, training=False)
               
                pi=tf.ones_like(tf.stack(y_pred[0:nT],0))*self.cfd_type['Pi']
                self.val_loss_tracker.update_state(pi,tf.stack(y_pred[0:nT],0),sample_weight=self.nwt_test[0:nT])
                self.val_mae_metric.update_state(pi,tf.stack(y_pred[0:nT],0),sample_weight=self.nwt_test[0:nT]) 
        else:
            y_pred = self(x, training=False)
            if self.cfd_type['Data_Arr'] in [1,3]:
                nT=tf.shape(y_pred)[0]
                self.val_loss_tracker.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[7:7+nT])
                self.val_mae_metric.update_state(tf.stack(y)[0:nT],tf.stack(y_pred,0),sample_weight=self.nwt[7:7+nT])
            elif self.cfd_type['Data_Arr']==2:
                if self.cfd_type['Fluid_Type']in ['dry-gas','DG']: 
                    self.val_loss_tracker.update_state([y[0],y[1]],[y_pred[0],y_pred[1]],sample_weight=self.nwt[7:9])
                    self.val_mae_metric.update_state([y[0],y[1]],[y_pred[0],y_pred[1]],sample_weight=self.nwt[7:9])
                else:
                    self.val_loss_tracker.update_state([y[0],y[1],y[2]],[y_pred[0],y_pred[1],y_pred[2]],sample_weight=self.nwt[7:10])
                    self.val_mae_metric.update_state([y[0],y[1],y[2]],[y_pred[0],y_pred[1],y_pred[2]],sample_weight=self.nwt[7:10])
        # Return a dict mapping metric names to current value--i.e., Mean
        return {"loss": self.val_loss_tracker.result(), "mae": self.val_mae_metric.result()}
        
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        if self.cfd_type['Type']=='PINN' or self.cfd_type['Data_Arr']==0:
            data_reset=[self.total_loss,self.dom_loss,self.dbc_loss,self.nbc_loss,self.ibc_loss,self.ic_loss,self.mbc_loss,self.cmbc_loss,self.td_loss['p'],self.td_loss['sg'],self.mae_metric,self.val_loss_tracker,self.val_mae_metric,self.grad_norm]
            if self.cfd_type['Fluid_Type'] not in ['dry-gas','DG']:
                data_reset=data_reset+[self.td_loss['so']]
        else:
            if self.cfd_type['Data_Arr']==1:
                data_reset=[self.total_loss,self.dom_loss,self.dbc_loss,self.nbc_loss,self.ibc_loss,self.ic_loss,self.td_loss['p'],self.td_loss['sg'],self.mae_metric,self.val_loss_tracker,self.val_mae_metric,self.grad_norm]
                if self.cfd_type['Aux_Layer']['Bu']['Use']:
                    data_reset=data_reset+[self.td_loss['Bg'],self.td_loss['ug']]
                    if self.cfd_type['Fluid_Type'] not in ['dry-gas','DG']:
                        data_reset=data_reset+[self.td_loss['so'],self.td_loss['Bo'],self.td_loss['uo']]      
                               
            elif self.cfd_type['Data_Arr']==2:
                data_reset=[self.total_loss,self.dom_loss,self.dbc_loss,self.nbc_loss,self.ibc_loss,self.ic_loss,self.td_loss['p'],self.td_loss['sbu_g'],self.mae_metric,self.val_loss_tracker,self.val_mae_metric,self.grad_norm]
                if self.cfd_type['Fluid_Type'] not in ['dry-gas','DG']:
                    data_reset=data_reset+[self.td_loss['sbu_o']]
            elif self.cfd_type['Data_Arr']==3:       # Special data arrangement for PVT pretraining
                data_reset=[self.total_loss,self.mae_metric,self.val_loss_tracker,self.val_mae_metric,self.grad_norm,self.td_loss['Bg'],self.td_loss['ug']]
                if self.cfd_type['Fluid_Type'] not in ['dry-gas','DG']:
                    data_reset=data_reset+[self.td_loss['Bo'],self.td_loss['uo'],self.td_loss['Rs'],self.td_loss['Rv']] 
        return data_reset

# Custom Layer class for the viscosity-compressibility function
class viscosity_FVF_inverse(tf.keras.layers.Layer):
    # Order of function: invBg,invug,d/dp(invBg),d/dp(invug) + invBo,invuo,d/dp(invBo),d/dp(invuo)
    # +...+(Fx**5)+(Ex**4)+(Dx**3)+(Cx**2)+(Bx**1)+(A**0)
    def __init__(self,units=1,pweights=[],cname=''):
        super(viscosity_FVF_inverse, self).__init__(name=cname)
        self.units=units
        self.pweights=pweights
        self.cname=cname
        #self.custom_function=True

    def build(self,input_shape):
        '''
        if len(self.pweights)==0:
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(initial_value=w_init(shape=(tf.shape(self.pweights)[0],self.units)),trainable=False,)
            self.b= tf.Variable(initial_value=w_init(shape=(tf.shape(self.pweights)[0],self.units)),trainable=False,}'''
           
        self.w = self.add_weight(name=self.cname+'_weights',shape=(tf.shape(self.pweights)[0],self.units),initializer=tf.constant_initializer(value=self.pweights),trainable=False)
        self.b = self.add_weight(name=self.cname+'_bias',shape=(self.units,),initializer='zeros',trainable=False)
        
    def call(self, inputs):
        inputs=tf.squeeze(inputs,-1)
        inv_inputs=tf.squeeze(tf.transpose([tf.map_fn(lambda j: tf.math.pow(tf.cast(inputs,inputs.dtype),j),tf.cast(tf.map_fn(lambda i:i,tf.range(tf.shape(self.pweights)[0]-1,-1,-1)),inputs.dtype))]),-1)
        #X_invu=tf.squeeze(tf.transpose([tf.map_fn(lambda j: tf.math.pow(tf.cast(inputs,dt_type),j),tf.cast(tf.map_fn(lambda i:i,tf.range(self.poly_order[1],-1,-1)),dt_type))]),-1)

        # Derivative--invB*, invu*  *: could be for gas or condensate
        #X_dinvB=tf.stack([(tf.math.pow(tf.cast(inputs,dt_type),tf.cast(j,dt_type))*tf.cast(j+1,dt_type)) for j in tf.map_fn(lambda i:i,tf.range(self.poly_order[0]-1,-1,-1))],axis=1)
        #X_dinvu=tf.stack([(tf.math.pow(tf.cast(inputs,dt_type),tf.cast(j,dt_type))*tf.cast(j+1,dt_type)) for j in tf.map_fn(lambda i:i,tf.range(self.poly_order[1]-1,-1,-1))],axis=1)

        #X_dinvB=tf.squeeze(tf.transpose([tf.map_fn(lambda j: tf.math.pow(tf.cast(inputs,dt_type),j)*(j+1.),tf.cast(tf.map_fn(lambda i:i,tf.range(self.poly_order[0]-1,-1,-1)),dt_type))]),-1)
        #X_dinvu=tf.squeeze(tf.transpose([tf.map_fn(lambda j: tf.math.pow(tf.cast(inputs,dt_type),j)*(j+1.),tf.cast(tf.map_fn(lambda i:i,tf.range(self.poly_order[1]-1,-1,-1)),dt_type))]),-1)

        #idx=tf.repeat(tf.shape(self.w),[1,tf.math.abs(tf.rank(inv_inputs)-tf.rank(self.w))+1])
        #self.w=tf.reshape(self.w,idx)
        out=((tf.matmul(inv_inputs,self.w)+self.b))             #transpose
        #out=tf.reduce_sum(self.w*inv_inputs,axis=[0])+self.b
        return tf.transpose(out,perm=[1,0,2])

class identity_Layer(tf.keras.layers.Layer):
    def __init__(self,constant=1.,trainable=False,layer_name=''):
        super(identity_Layer, self).__init__(name=layer_name)
        self.constant=tf.Variable(constant,trainable=trainable)
    def call(self, inputs):
        return self.constant*(inputs)  

class constant_Layer(tf.keras.layers.Layer):
    def __init__(self,constant=1.,trainable=False,layer_name='',mean=0.,stddev=0.):
        super(constant_Layer, self).__init__(name=layer_name)
        self.constant=tf.Variable(constant,trainable=trainable)
        self.mean=mean
        self.stddev=stddev
    def call(self, inputs):
        return self.constant*(tf.ones_like(inputs))#+tf.random.normal(tf.shape(inputs),mean=self.mean,stddev=self.stddev)

class constantDiff_Layer(tf.keras.layers.Layer):
    def __init__(self,constant=1.,trainable=False,layer_name=''):
        super(constantDiff_Layer, self).__init__(name=layer_name)
        self.constant=tf.Variable(constant,trainable=trainable)
    def call(self, inputs):
        return self.constant-inputs 

class transpose_Layer(tf.keras.layers.Layer):
    def __init__(self,perm=[],layer_name=''):
        super(transpose_Layer, self).__init__(name=layer_name)
        self.perm=perm
    def call(self, inputs):
        return tf.transpose(inputs,perm=self.perm) 
    
class multiply_Layer(tf.keras.layers.Layer):
    def __init__(self,constant=1.,trainable=False,layer_name=''):
        super(multiply_Layer, self).__init__(name=layer_name)
        self.constant=tf.Variable(constant,trainable=trainable)
        #self.trainable_kernel=trainable
    def call(self, x,y=1.,constraint=None):

        y=tf.convert_to_tensor(y,dtype=x.dtype)
        output=self.constant*tf.multiply(x,y) 
        #if constraint is not None:
        constraint=[tf.convert_to_tensor(constraint[i],dtype=x.dtype) for i in [0,1]]
        output=(tf.cast(output<constraint[0],x.dtype)*constraint[0])+(tf.cast((output>=constraint[0])&(output<=constraint[1]),x.dtype)*output)+\
            (tf.cast(output>constraint[1],x.dtype)*constraint[1])
        return output
    
class expand_dims_Layer(tf.keras.layers.Layer):
    def __init__(self,conn_idx=None,qoi=None,dim=None,layer_name=''):
        super(expand_dims_Layer, self).__init__(name=layer_name)
        self.conn_idx=conn_idx
        self.qoi=qoi
        self.dim=dim
    def call(self, inputs):
        inputs_0=tf.split(inputs,num_or_size_splits=inputs.shape[-1],axis=-1)[0]
        return tf.cast(tf.expand_dims(tf.scatter_nd(self.conn_idx, self.qoi, self.dim),0),inputs.dtype)*tf.ones_like(inputs_0)    


class constantPadding_Layer(tf.keras.layers.Layer):
    def __init__(self,pad=[],mode='SYMMETRIC',layer_name=''):
        super(constantPadding_Layer, self).__init__(name=layer_name)
        self.pad=pad
        self.mode=mode
    def call(self, inputs):
        return tf.pad(inputs,self.pad,mode=self.mode)
  
class split_Layer(tf.keras.layers.Layer):
    def __init__(self,no_splits=None,axis=-1,layer_name='',reshape=None):
        super(split_Layer, self).__init__(name=layer_name)
        self.no_splits=no_splits
        self.axis=axis
        self.no_splits_list=list(range(no_splits))
        self.reshape=lambda x:tf.reshape(x,[-1,*reshape[1:]])
        if reshape is None:
            self.reshape=lambda x:x
    def call(self, inputs):
        return [self.reshape(tf.split(inputs,num_or_size_splits=self.no_splits, axis=self.axis)[i]) for i in self.no_splits_list] 
    
class restep_Layer(tf.keras.layers.Layer):
    def __init__(self,limits=[0,1],values=[0,1],layer_name=''):
        super(restep_Layer, self).__init__(name=layer_name)
        self.limits=limits
        self.values=values
    def call(self, inputs):
        c=inputs[0]
        p=inputs[1]
        out=(tf.cast(c<=self.limits[0],self.dtype)*self.values[0])+(tf.cast((c>self.limits[0])&(c<=self.limits[1]),self.dtype)*p)+\
            (tf.cast((c>self.limits[1]),self.dtype)*self.values[1])
        return out
    
class binary_Layer(tf.keras.layers.Layer):
    def __init__(self,limits=[0.,],values=[None,1.],layer_name='',alpha=0.,trainable=False,lower_limit=0.,upper_limit=0.99):
        super(binary_Layer, self).__init__(name=layer_name)
        self.limits=limits
        self.values=values
        self.alpha=0.005
        if self.values[0] in [None,'',0.,'None']:
            self.values[0]=lambda x:tf.zeros_like(x)
        
        self.init_value=tf.constant_initializer(value=alpha)
        self.alpha=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
                            max_value=upper_limit, rate=1., axis=0),trainable=trainable)


    def call(self, inputs):
        c=inputs
        d=tf.nn.relu(c)
        e=(d**(1.+self.alpha))-d+tf.ones_like(d)
        out=(tf.cast(c<=self.limits[0],self.dtype)*self.values[0](c))+\
            (tf.cast(c>self.limits[0],self.dtype)*(d+0.))
        return out
    
class tsoftplus_Layer(tf.keras.layers.Layer):
    def __init__(self,layer_name='',alpha=0.05,trainable=True,lower_limit=0.,upper_limit=0.05):
        super(tsoftplus_Layer, self).__init__(name=layer_name)
        self.init_value=tf.constant_initializer(value=alpha)
        self.alpha=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
                            max_value=upper_limit, rate=1., axis=0),trainable=trainable)

    def call(self, inputs):
        return tf.nn.relu(tf.math.log(tf.math.exp(tf.math.exp(1.)*inputs)+self.alpha))
    
class sigmoid_Layer(tf.keras.layers.Layer):
    def __init__(self,layer_name='',alpha=[1.,1],trainable=[False,True],lower_limit=[10.,1.],upper_limit=[100.,10],rate=[1.,1.]):
        super(sigmoid_Layer, self).__init__(name=layer_name)
        self.trainable_kernel=trainable
        self.len=range(len(alpha))
        self.init_value=[tf.constant_initializer(value=alpha[i]) for i in self.len]
        self.alpha=[tf.Variable(initial_value=self.init_value[i]([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit[i],\
                            max_value=upper_limit[i], rate=rate[i], axis=0),trainable=trainable) for i in self.len]

    def call(self, inputs):
        out=(tf.math.divide_no_nan(1.,(1. + tf.math.exp(-self.alpha[0]*inputs))))**(self.alpha[1])
        return out

class scaled_relu_Layer(tf.keras.layers.Layer):
    def __init__(self,limits=[0,1],act_func=[tf.nn.sigmoid],layer_name='',alpha=1.,trainable=False,lower_limit=0.1,upper_limit=0.99,dew_point=4048.4854,beta=0.):
        super(scaled_relu_Layer, self).__init__(name=layer_name)
        self.limits=limits
        self.trainable_kernel=trainable
        self.init_value=tf.constant_initializer(value=alpha)
        self.alpha=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
                            max_value=upper_limit, rate=1., axis=0),trainable=trainable)
        self.act_func=act_func
        self.pdew=dew_point
        self.beta=beta
        self.logs=[]
        
    def call(self, inputs):
        out=self.alpha*self.act_func[0](inputs)
        out=(tf.cast(out<self.limits[0],self.dtype)*self.limits[0])+(tf.cast((out>=self.limits[0])&(out<=self.limits[1]),self.dtype)*out)+\
             (tf.cast(out>self.limits[1],self.dtype)*(self.limits[1]+self.beta*out))

        #out=(tf.cast(pre>=self.pdew,self.dtype)*0.)+(tf.cast((pre<self.pdew),self.dtype)*self.act_func[0](out)*self.alpha) 
        return out

class trainable_activation_Layer(tf.keras.layers.Layer):    
    def __init__(self,layer_name='',alpha=1.,trainable=False,lower_limit=0.1,upper_limit=10.,act_func=lambda x:tf.math.exp(-x)):
        super(trainable_activation_Layer, self).__init__(name=layer_name)
        self.trainable_kernel=trainable
        self.init_value=tf.constant_initializer(value=alpha)
        self.alpha=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
                            max_value=upper_limit, rate=1., axis=0),trainable=trainable)
        self.act_func=act_func  
    def call(self, inputs):
        #return self.alpha*tf.math.log(1.+tf.math.abs(inputs))
        return self.act_func(inputs*self.alpha)
    
class gaussian_blur_Layer(tf.keras.layers.Layer):    
    def __init__(self,layer_name='',sigma=0.01,trainable=False,lower_limit=0.,upper_limit=0.5,model=None,f_idx=5,mean=3.):
        super(gaussian_blur_Layer, self).__init__(name=layer_name)
        self.model=model #uninitialized model wrapper
        self.trainable_kernel=trainable
        # self.init_value=tf.constant_initializer(value=bl.normalize_diff(self.model,sigma,stat_idx=f_idx,compute=True,x0=mean).numpy())
        # self.sigma=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
        #                     max_value=bl.normalize_diff(self.model,upper_limit,stat_idx=f_idx,compute=True,x0=mean), rate=1., axis=0),trainable=trainable)
        self.init_value=tf.constant_initializer(sigma)
        self.sigma=tf.Variable(initial_value=self.init_value([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=lower_limit,\
                             max_value=upper_limit, rate=1., axis=0),trainable=trainable)
            
    def call(self, inputs):
        return tfa.image.sharpness(inputs,self.sigma)
        #return tfa.image.gaussian_filter2d(inputs, (3, 3),0.05,'REFLECT')

    
class timestep_Hardlayer(tf.keras.layers.Layer):    
    def __init__(self,layer_name='',norm_limits=[[-1,1],[-1,1]],init_value=1.,alpha={'Value':0.1,'Trainable':False,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.},\
                 tshift=5.,model=None,cfd_type=None,activation_func=tf.nn.softplus):
        super(timestep_Hardlayer, self).__init__(name=layer_name)
        self.norm_limits=norm_limits
        self.init_value=init_value
        self.alpha=alpha
        self.tshift=tshift
        self.model=model
        self.init=tf.constant_initializer(value=alpha['Value'])
        self.alpha=tf.Variable(initial_value=self.init([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=alpha['Lower_Limit'],\
                            max_value=alpha['Upper_Limit'], rate=alpha['Rate'], axis=0),trainable=alpha['Trainable']) 
        self.activation_func=activation_func
        
        
    def call(self, inputs,):
        tn1=inputs[0][0]
        tn0=inputs[0][1]
        dt_n=inputs[1]
        
        t1=bl.nonormalize(self.model,(tn1),stat_idx=3,compute=True)
        t0=bl.nonormalize(self.model,(tn0),stat_idx=3,compute=True)
        #alpha=(tnorm-self.norm_limits[0][0])/(self.norm_limits[0][1]-self.norm_limits[0][0])
        tstep_sc=tf.math.maximum(tf.math.minimum(self.alpha*(dt_n),tf.ones_like(dt_n)*10.),tf.ones_like(dt_n)*0.1)
        welopens_idx=tf.reduce_mean([tf.reduce_mean([self.wel_opens(limits=self.model.cfd_type['Connection_Shutins']['Days'][c][cidx],time_n0=t0,time_n1=t1,dtype=self.dtype,tshift=self.tshift) for cidx in self.model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in self.model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)
        tstep=(tf.cast(welopens_idx>0.,self.dtype)*tstep_sc)+(tf.cast(welopens_idx<=0.,self.dtype)*dt_n)
        
        return tstep
    def wel_opens(self,limits=None,time_n0=None,time_n1=None,dtype=None,tshift=1.):
        return (tf.cast((((time_n0>(limits[0]-tshift))&(time_n0<(limits[0]+tshift)))|((time_n0>(limits[1]-tshift))&(time_n0<(limits[1]+tshift))))&(tf.reduce_sum(limits)>0),dtype))        
        #return (tf.cast((time_n0==time_n1)&(((time_n0>=(limits[0]-tshift))&(time_n0<=(limits[0]+tshift)))|((time_n0>=(limits[1]-tshift))&(time_n0<=(limits[1]+tshift))))&(tf.reduce_sum(limits)>0),dtype)) 

class timestep_pressure_Layer(tf.keras.layers.Layer):
    def __init__(self,value=None,layer_name=''):
        super(timestep_pressure_Layer, self).__init__(name=layer_name)
    def call(self, inputs,pre):
        p_idx=tf.math.softmax(tf.math.abs(pre-tf.reduce_mean(pre,axis=[1,2,3],keepdims=True)),axis=1)
        tstep=tf.reduce_sum(inputs*p_idx,axis=[1,2,3],keepdims=True)
        
        return tstep

        

class replace_nan_Layer(tf.keras.layers.Layer):
    def __init__(self,value=None,layer_name=''):
        super(replace_nan_Layer, self).__init__(name=layer_name)
        self.value=value
    def call(self, inputs):
        out=tf.where(tf.math.logical_or(tf.math.is_nan(inputs),tf.math.is_inf(inputs)),tf.ones_like(inputs)*self.value,inputs)
        return out

# class update_variables_Layer(tf.keras.layers.Layer):
#     def __init__(self,shape=[None,39,39,1],layer_name=''):
#         super(update_variables_Layer, self).__init__(name=layer_name)
#         self.dt_n_1=tf.keras.Input(shape=shape[1:], name='previous_timestep')
#         self.dt_n_1*=tf.zeros(shape=[1,*shape[1:]])        

#     def call(self, inputs):
#         out=self.dt_n_1
#         self.dt_n_1=(-self.dt_n_1+inputs)
#         return out
        
class normalization_Layer(tf.keras.layers.Layer):
    def __init__(self,stat_limits=[5000.,14.7],norm_limits=[-1,1],norm_type='linear',layer_name=''):
        super(normalization_Layer, self).__init__(name=layer_name)
        self.stat_limits=stat_limits
        self.norm_limits=norm_limits
        self.norm_type=norm_type
    def call(self, inputs):
        #out=(((inputs-self.stat_limits[0])/(self.stat_limits[1]-self.stat_limits[0]))*(self.norm_limits[1]-self.norm_limits[0]))+self.norm_limits[0]
        out=(-((self.stat_limits[0]-inputs)/(self.stat_limits[0]-self.stat_limits[1]))*(self.norm_limits[0]-self.norm_limits[1]))+self.norm_limits[0]
        return out

class spline_Interpolation(tf.keras.layers.Layer):
    def __init__(self,points_x=None,values_y=None,order=1,layer_name=''):
        super(spline_Interpolation, self).__init__(name=layer_name)
        self.points_x=points_x
        self.values_y=values_y
        self.order=order
    def call(self, inputs):
        return tfa.image.interpolate_spline(self.points_x,self.values_y,inputs,self.order)

def scaled_lisht(x,y,act_func=tfa.activations.lisht):   
    def func(z):
        b=0.25
        f=act_func(z)
        u=(1.-b)*y
        l=(1.+b)*x 
        mu=y+0.1*f# ((1.-tf.math.exp(-3.142*f/y))*y)#
        ml=x+0.1*f #tf.math.exp(f)*x 
        return tf.where(f>u,tf.math.minimum(f,mu),tf.where(f>l,f,tf.math.maximum(f,ml)))
        #return tf.where(f>y,tf.ones_like(f)*y,tf.where(f>x,f,tf.ones_like(f)*x))
    return func
class gas_oil_Hardlayer(tf.keras.layers.Layer):
    def __init__(self,norm_limits=[[-1,1],[-1,1]],kernel_activation=[None,tf.nn.relu],init_value=None,kernel_exponent=[{'Value':0.5773,'Trainable':True,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.}],input_activation=tf.nn.sigmoid,layer_name='',**kwargs):
        super(gas_oil_Hardlayer, self).__init__(**kwargs,name=layer_name)
        self.norm_limits=norm_limits
        self.init_value=init_value
        self.input_activation=input_activation
        self.trainable=True
        self.kernel_activation=kernel_activation
        self.init=[tf.constant_initializer(value=kernel_exponent[i]['Value']) for i in range(len(kernel_exponent))] #0.57721
        self.kernel_exponent=[tf.Variable(initial_value=self.init[i]([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=kernel_exponent[i]['Lower_Limit'],\
                            max_value=kernel_exponent[i]['Upper_Limit'], rate=kernel_exponent[i]['Rate'], axis=0),trainable=kernel_exponent[i]['Trainable']) for i in range(len(kernel_exponent))]
        
        self._kernel_initializer=tf.keras.initializers.get('glorot_normal')
        self.rbf_dim=50
        softplus1=lambda x: lambda z: tf.math.minimum(tf.nn.softplus(z),x)
        #self.dense_layer_k=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(32, activation=tf.nn.swish,**self._get_common_kwargs_for_layer())) #
        self.rbf_k=tf.keras.layers.experimental.RandomFourierFeatures(output_dim=self.rbf_dim,scale=0.5,trainable=True,dtype=self.dtype,**self._get_common_kwargs_for_layer())
        self.dense_output_k=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,**self._get_common_kwargs_for_layer()))  #lambda x:1.+tf.math.exp(-x) tfa.layers.SpectralNormalization 
        self.p_dew=4048.45
        self.p_maxl=3000.
        #self.layer_norm=tf.keras.layers.LayerNormalization(axis=[1,2,3])
        if self.kernel_activation[0] in [None,'']:
            self.kernel_activation[0]=lambda x:x
        if self.kernel_activation[1] in [None,'']:
            self.kernel_activation[1]=lambda x:x
        
        if input_activation is None:
            self.input_activation=lambda x:x
            
        p1=-1.4602E-08; p2=1.6856E-04; p3=-6.4895E-01; p4=8.3343E+02
        self._Sg_CCE=lambda x:((p1*x**3)+(p2*x**2)+(p3*x**1)+(p4*x**0))

        
    def call(self, inputs):
        t=inputs[0][0]
        k=inputs[0][1]
        p=inputs[0][2]
        sg=inputs[1]
        alpha=self.kernel_activation[0]((t-self.norm_limits[0][0])/(self.norm_limits[0][1]-self.norm_limits[0][0]))
        beta=(self.kernel_activation[1]((p-self.norm_limits[1][0])/(self.norm_limits[1][1]-self.norm_limits[1][0])))#*self.kernel_exponent[0][0]
        beta_a=(self.kernel_activation[1]((p-self.norm_limits[1][0])/(self.norm_limits[1][1]-self.norm_limits[1][0])))
        
        beta_dp=tf.nn.relu((p-self.norm_limits[1][0])/(self.p_maxl-self.norm_limits[1][0]))#*alpha
        gamma_0=tf.math.maximum(tf.math.minimum((p-self.norm_limits[1][0])/(self.p_maxl-self.norm_limits[1][0]),5.),0.)
        gamma_1=tf.math.maximum(tf.math.minimum(((p-self.p_maxl)/(self.norm_limits[1][1]-self.p_maxl))+1.,2.),1.)
        #gamma=tf.where(p>=self.p_dew,0.,tf.where((p>=self.p_maxl),gamma_0,gamma_1))
        
        beta_m0=tf.where((p>self.p_maxl),1.,0.)
        beta_m1=tf.where(p==14.7,tf.ones_like(self.input_activation(sg)),tf.zeros_like(self.input_activation(sg))) #tf.where(p<=4048.,tf.ones_like(p),tf.zeros_like(p))#
        
        tp=tf.multiply(1.,beta)
        Sg_m=tf.reshape(self.rbf_k(tf.reshape(tp,(-1,1))),tf.concat([tf.shape(p)[0:-1],[tf.shape(p)[-1]*self.rbf_dim]],0))
        Sg_m=self.dense_output_k(Sg_m)
        alpha_pb=tf.nn.relu(tf.math.divide_no_nan((alpha-Sg_m),(1.-Sg_m)))


        #out=(self.init_value)-((beta)*(1.+alpha)*self.input_activation(sg)) 
        #beta_mp=(1.-beta_m0)*(1.-beta_m1)
        #out=(self.init_value)-(beta_mp*beta*(1.+alpha)*self.input_activation(sg))-beta_m0*(self.init_value-0.3598)-beta_m1*(self.init_value-0.1593)
        #out=(self.init_va lue)-(beta*(1+alpha)*(sg)) #beta*(1+alpha)  
        #out=tf.where((out>=(self.init_value-0.35))&(out<=self.init_value),out,tf.math.sigmoid(out))  #1.*(tf.math.exp(out)-1.) tf.ones_like(out)*self.init_value
        #out=(tf.cast(out<(self.init_value-0.35),self.dtype)*(tf.ones_like(out)*0.35))+(tf.cast((out>=(self.init_value-0.35))&(out<=self.init_value),self.dtype)*out)+(tf.cast(out>self.init_value,self.dtype)*tf.ones_like(out)*self.init_value)
        #out=(self.init_value)-(beta_m0*0.35)-(beta*(1.+alpha)*(1.-beta_m0)*self.input_activation(sg))
        so_maxl=0.36
        clip=lambda l,m,a: lambda z: tf.where((z<=m),z,tf.math.divide_no_nan(m,tf.math.exp(-a*m))*tf.math.exp(-a*z)) #2*m-z m*tf.math.exp(-z**2/2) ,m*0.5*(1-tf.math.sin(z))
        clip1=lambda l,m: lambda z:z-(1+0.05)*tf.nn.relu(z-m) #0.75*(1-tf.math.tanh(z))*(z+0.01)**m  #2*m*(1.-tf.math.sigmoid(z*tf.math.tanh(z)))
        eta=(gamma_0**(1.-gamma_0))
        stb=lambda n: lambda x: tf.math.divide_no_nan(x,tf.math.maximum(x,1e-6)**(1.-n)) 
        sigmoid1=tfp.bijectors.Sigmoid(low=0.,high=0.35, validate_args=False, name='sigmoid')
        #out=(self.init_value)-tf.math.sign(beta)*((0.5*((1.-beta)*beta**0.5))+(alpha*(beta+1)))*(tf.nn.sigmoid(sg/100))
        #out=(self.init_value)-(0.2*tf.math.sign(beta)+(beta*alpha)*(sg))#tf.math.exp(-(beta*(0.+alpha))**2)

        ab=alpha*beta+alpha#tf.math.maximum((beta)*(alpha),1e-6)
        sb=tf.math.sign(beta)
        ab_dp=so_maxl*tf.math.maximum(beta_dp,1e-6)**(1.-beta_dp)
        
        so_a=((alpha)*(tf.nn.sigmoid((sg/1)**1)))#sb*(alpha*tf.math.exp(-alpha/tf.math.sigmoid(beta))+(1.-0.5*tf.math.tanh(alpha))*0.2)*sg #
        so_b=so_maxl-(so_a-so_maxl)*0.125 #-0.125*beta+0.355#so_maxl*tf.math.exp(-(0.01)*(beta))
        so=tf.where(so_a<=so_maxl,so_a,so_b)        
        #out=(self.init_value)-(ab_dp)
        out=(self.init_value)-sb*clip1(0.,0.36)(so_a)
        #out=(self.init_value)-sb*self.init_value*(beta**(0.25)*(1-beta)+(0.15*beta))
        #out=(self.init_value)-tf.math.minimum(sb*((1.-alpha)*alpha**0.5)*(sg/100)**1,0.35)           
        return out
     
    def _get_common_kwargs_for_layer(self):
        common_kwargs = dict()
        # Create new clone of kernel/bias initializer, so that we don't reuse the initializer instance, which could lead to same init value since initializer is stateless.
        kernel_init = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_init
        return common_kwargs

class output_Hardlayer(tf.keras.layers.Layer):
    def __init__(self,norm_limits=[[-1,1],[-1,1]],co_var='no_correlate',init_value=None,kernel_activation=[None,None],input_activation=None,layer_name='',\
        kernel_exponent=[{'Value':1.,'Trainable':False,'Lower_Limit':0.,'Upper_Limit':0.99,'Rate':1.},{'Value':1.,'Trainable':False,'Lower_Limit':0.,'Upper_Limit':0.99,'Rate':1.}],\
        rbf_activation_function=tf.nn.sigmoid,regularization=0.,weights=[0.],model=None,cfd_type=None,**kwargs):
        super(output_Hardlayer, self).__init__(**kwargs,name=layer_name)
        self.norm_limits=norm_limits
        self.init_value=init_value
        self.regularization=regularization
        self.rweights=weights
        
        self.trainable_kernel=True
        self.init=[tf.constant_initializer(value=kernel_exponent[i]['Value']) for i in range(len(kernel_exponent))] #0.57721
        self.kernel_exponent=[tf.Variable(initial_value=self.init[i]([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=kernel_exponent[i]['Lower_Limit'],\
                            max_value=kernel_exponent[i]['Upper_Limit'], rate=kernel_exponent[i]['Rate'], axis=0),trainable=kernel_exponent[i]['Trainable']) for i in range(len(kernel_exponent))]
        self.kernel_activation=kernel_activation
        self.input_activation=input_activation
        self.model=model
        #self.cfd_type=cfd_type
        self.tn0=norm_limits[0]
        self.co_var=co_var
        self._kernel_initializer=tf.keras.initializers.get('glorot_normal')
        
        if self.model is not None:
            self.tmax=bl.nonormalize(self.model,1.,stat_idx=3,compute=True)

        if self.kernel_activation[0] in [None,'']:
            self.kernel_activation[0]=lambda x:x
        if self.kernel_activation[1] in [None,'']:
            self.kernel_activation[1]=lambda x:x
        
        if input_activation is None:
            self.input_activation=lambda x:x
            
        # self.depth_dense=1
        #self.trn_act_func=trainable_activation_Layer(layer_name='k_act_func',alpha=1.,trainable=True,lower_limit=0.1,upper_limit=10.,act_func=lambda x:tf.math.exp(-x))
        self.rbf_dim=50
        #self.dense_layer_k=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(32, activation=tf.nn.swish,**self._get_common_kwargs_for_layer())) 
        self.rbf_k=tf.keras.layers.experimental.RandomFourierFeatures(output_dim=self.rbf_dim,scale=0.5,trainable=True,dtype=self.dtype,**self._get_common_kwargs_for_layer())
        self.dense_output_k=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(1, activation=rbf_activation_function,**self._get_common_kwargs_for_layer()))  #lambda x:1.+tf.math.exp(-x) tfa.layers.SpectralNormalization 
        #self.dense_output_k=tf.keras.layers.Conv2D(1, 3, strides=1, padding='same',data_format='channels_last', activation=tf.nn.swish,**self._get_common_kwargs_for_layer())       
        #tfa.layers.SpectralNormalization
        #self.dense_output_tk=tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)  #lambda x:tf.minimum(tf.nn.softplus(x),1.)
        # self.dense_output_tk=tfa.layers.SpectralNormalization(tf.keras.layers.Dense(1, activation=tf.nn.softplus,**self._get_common_kwargs_for_layer()))  #lambda x:1.+tf.math.exp(-x)        
        # self.rbf_tk=tf.keras.layers.experimental.RandomFourierFeatures(output_dim=self.rbf_dim,scale=0.5,trainable=True,dtype=dtype,**self._get_common_kwargs_for_layer())
        # self.mha=tf.keras.layers.MultiHeadAttention(key_dim=1,num_heads=2,**kwargs)  tfa.layers.SpectralNormalization

    def call(self, inputs,t_n0=-1.):
        tn1=inputs[0][0]
        tn2=inputs[0][1]
        tn3=inputs[0][2]
        p=inputs[1]
        
        t1=bl.nonormalize(self.model,(tn1),stat_idx=3,compute=True)
        t0=bl.nonormalize(self.model,(t_n0),stat_idx=3,compute=True)
        treg_fac=tf.reduce_mean([tf.reduce_mean([self.shut_days(limits=self.model.cfd_type['Connection_Shutins']['Days'][c][cidx],time_n0=t0,time_n1=t1,dtype=self.dtype,rate=.5) for cidx in self.model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in self.model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)
        
        # if init_layer is not None:
        #     self.init_value=init_layer
        beta=((tn3-self.norm_limits[1][0])/(self.norm_limits[1][1]-self.norm_limits[1][0]))
        lmd=(tf.cast(beta<=0.,tn3.dtype)*tf.zeros_like(beta))+(tf.cast(beta>0.,tn3.dtype)*tf.ones_like(beta))
        #lmd=(tf.cast(beta<=1.,tn2.dtype)*tf.zeros_like(beta))+(tf.cast(beta>1.,tn3.dtype)*tf.ones_like(beta))
        beta=(self.kernel_activation[1](beta))**(self.kernel_exponent[1][0])
        #beta=self.dense_layer(beta)
        #beta=self.dense_output(beta)beta=tf.where(b>=0.,beta,tf.zeros_like(beta))
        
        beta=tf.where(tf.math.equal(self.co_var,'correlate'),beta,1.)
        lmd=tf.where(tf.math.equal(self.co_var,'correlate'),lmd,1.)
        
        # Permeability
        tn2_=tf.multiply(1.,tn2)
        
        #tn2_=self.input_proj(tf.concat([tn1,tn2],-1))
        alpha_k=tf.reshape(self.rbf_k(tf.reshape(tn2_,(-1,1))),tf.concat([tf.shape(p)[0:-1],[tf.shape(p)[-1]*self.rbf_dim]],0))
        #alpha_k=self.dense_layer(tn2_)
        alpha_k=self.dense_output_k(alpha_k)

        #Time - Permeability
        # alpha_tk=tf.reshape(self.rbf_tk(tf.reshape(tn2,(-1,1))),tf.concat([tf.shape(p)[0:-1],[tf.shape(p)[-1]*self.rbf_dim]],0))
        # alpha_tk=self.dense_output_tk(alpha_tk)
        #stb=lambda n: lambda x: tf.math.divide_no_nan(x,tf.math.maximum(x,1.e-6)**(1.-n))       
        alpha_t=((tn1-self.norm_limits[0][0])/(self.norm_limits[0][1]-self.norm_limits[0][0]))
        alpha=self.kernel_activation[0](alpha_t)
        alpha=((alpha)**(self.kernel_exponent[0][0]))*(1./alpha_k)
        #alpha=(alpha)**(self.kernel_exponent[0][0])              
        
        # shp=tf.concat([tf.constant([-1]),tf.ones(tf.rank(t),dtype=tf.int32)],0)
        # diff_t=tf.reduce_min(tf.math.maximum(tf.expand_dims(t,0)-tf.reshape(self.tn0,shp),0.),axis=0)
        # alpha=(self.kernel_activation((t-self.norm_limits[0])/(self.norm_limits[1]-self.norm_limits[0]+diff_t)))**(self.kernel_exponent)
        # output_hard=self.init_value*(1-alpha)-(alpha)*p
        # regu=self.regularization*tf.linalg.global_norm(self.rweights)**2
        output_hard=(self.init_value)-(alpha*self.input_activation(p))#-(0.5*alpha**2)*regu
        
        #output_hard=((self.init_value)*(1.-alpha_t))-(alpha_t*self.input_activation(p))#-(0.5*alpha**2)*regu

        return output_hard
    
    def shut_days(self,limits=None,time_n0=None,time_n1=None,dtype=None,rate=0.5):
        out=(tf.cast((time_n0<limits[0])&(limits[0]>0.),dtype)*1.)+(tf.cast((time_n0>=limits[0])&(time_n0<=limits[1])&((limits[0]+limits[1])>0.),dtype)*1.)+\
            (tf.cast((time_n0>limits[1])&(limits[1]>0.),dtype)*2.)
        # out=(tf.cast((time_n0<limits[0])&(limits[0]>0.),dtype)*tf.nn.relu((time_n1-0.)/(limits[0]-0.)))+(tf.cast((time_n0>=limits[0])&(time_n0<=limits[1])&((limits[0]+limits[1])>0.),dtype)*tf.nn.relu((time_n1-limits[0])/(limits[1]-limits[0])))+\
        #     (tf.cast((time_n0>limits[1])&(limits[1]>0.),dtype)*tf.nn.relu((time_n1-limits[1])/(self.tmax-limits[1])))

        out=tf.where(out==0.,1.,out)
        return (out)
    def _get_common_kwargs_for_layer(self):
        common_kwargs = dict()
        # Create new clone of kernel/bias initializer, so that we don't reuse the initializer instance, which could lead to same init value since initializer is stateless.
        kernel_init = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_init
        return common_kwargs


class self_attention_Layer(tf.keras.layers.Layer):
    def __init__(self,num_heads=2,layer_name='',shape=(1,39,39,1),filter_factor=1,**kwargs):
        super(self_attention_Layer, self).__init__(name=layer_name) 
        self.num_filters=[shape[-1]//filter_factor,shape[-1]//filter_factor,shape[-1]]
        self.output_shapes=[shape[-1],int(tf.reduce_prod(shape[1:-1]))]
        self.mha=[tf.keras.layers.MultiHeadAttention(key_dim=self.num_filters[0],num_heads=num_heads,output_shape=self.output_shapes[i],**kwargs) for i in range(2)]
        self.cnn2d=[tf.keras.layers.Conv2D(self.num_filters[i], 1, strides=1, padding='same',data_format='channels_last', activation=None,name=layer_name+'_CNV2D_channels_attn_qkv'+str(i),) for i in range(3)]
        self.linproj=[tf.keras.layers.Conv2D(shape[-1], 1, strides=1, padding='same',data_format='channels_last', activation=None,name=layer_name+'_CNV2D_linproj'+str(i),) for i in range(3)]
        self.dense=tf.keras.layers.Dense(shape[-1],activation=None)
        self.add=tf.keras.layers.Add()
        self.layer_norm=tf.keras.layers.LayerNormalization(axis=-1)
        self.layer_name=layer_name
        self.init=tf.constant_initializer(value=1.) #tf.zeros_initializer()#0.57721
        self.gamma=tf.Variable(initial_value=self.init([1]),dtype=self.dtype,constraint=tf.keras.constraints.MinMaxNorm(min_value=0., max_value=1., rate=1., axis=0),trainable=True)
        self.trainable_kernel=True
    def call(self, inputs,values):
        # proj_query=self.cnn2d[0](inputs)
        # proj_key=self.cnn2d[1](inputs)
        # proj_value=self.cnn2d[2](inputs)
        # qk_reshape=(-1,proj_query.shape[-1],tf.reduce_prod(proj_query.shape[1:-1]))
        # v_reshape=(-1,proj_value.shape[-1],tf.reduce_prod(proj_value.shape[1:-1]))
        # proj_query=tf.transpose(tf.reshape(tf.transpose(proj_query,perm=[0,3,1,2]),qk_reshape),perm=[0,2,1]) # Bx(HxW)xC
        # proj_key=tf.reshape(tf.transpose(proj_key,perm=[0,3,1,2]),qk_reshape) #BxCx(HxW)
        # proj_value=tf.reshape(tf.transpose(proj_value,perm=[0,3,1,2]),v_reshape) # BxCx(HxW)
        
        # #attn_output,_=self.mha(query=self.cnn2d[0](inputs),key=self.cnn2d[1](inputs),value=self.cnn2d[2](inputs),return_attention_scores=True)
        # attn_map=tf.nn.softmax(tf.linalg.matmul(proj_query,proj_key),axis=-1)
        # attn_output=tf.linalg.matmul(proj_value,tf.transpose(attn_map,perm=[0,2,1]))
        # attn_output=tf.transpose(tf.reshape(attn_output,tf.concat([tf.constant([-1]),[tf.shape(inputs)[-1]],(tf.shape(inputs)[1:-1])],0)),[0,2,3,1])
        # #attn_output=tf.reshape(tf.transpose(attn_output,perm=[0,2,1]),tf.concat([tf.constant([-1]),tf.shape(inputs)[1:]],0))
        # attn_output=self.gamma*attn_output+inputs
        
        # Spatial Attention
        if values is None:
            values=inputs
        attn_output_sp,_=self.mha[0](query=inputs,key=values,value=values,return_attention_scores=True)
        attn_output=self.add([inputs,attn_output_sp])
        # attn_output=self.dense(attn_output_sp)
        
        # Channel Attention
        # qk_inputs=tf.reshape(tf.transpose(inputs,perm=[0,3,1,2]),(-1,inputs.shape[-1],tf.reduce_prod(inputs.shape[1:-1])))
        # v_inputs=tf.reshape(tf.transpose(values,perm=[0,3,1,2]),(-1,values.shape[-1],tf.reduce_prod(values.shape[1:-1])))
        # attn_output_ch,_=self.mha[1](query=qk_inputs,key=v_inputs,value=v_inputs,return_attention_scores=True)
        # attn_output_ch=tf.transpose(tf.reshape(attn_output_ch,tf.concat([tf.constant([-1]),[inputs.shape[-1]],inputs.shape[1:-1]],0)),perm=[0,2,3,1])
        # attn_output=self.add([inputs,self.gamma*attn_output_sp,(1-self.gamma)*attn_output_ch]) 
        # attn_output=self.dense(attn_output)
        return attn_output

    def check_attn_out(self,inputs,outputs):
        proj_output=outputs
        if (inputs.shape[-1])!=(outputs.shape[-1]):
              proj_output=self.linproj[0](outputs)                                        
        return proj_output
    
class spatial_attention_Layer(tf.keras.layers.Layer):
    def __init__(self,layer_name='',shape=(1,39,39,1),activation_func=None,kernel_size=3,**kwargs):
        super(spatial_attention_Layer, self).__init__(name=layer_name) 
        self.cnn2d=[tf.keras.layers.Conv2D(1, kernel_size, strides=1, padding='same',data_format='channels_last', activation='sigmoid',name=layer_name+'_CNV2D_spatial_attention'+str(i),) for i in range(1)]
        self.add=tf.keras.layers.Add()
        self.multiply=tf.keras.layers.Multiply()
        self.layer_name=layer_name
    def call(self, inputs,value=None):
        concat_max_avg=self.max_pool_max_avg(inputs)
        attn_map=self.cnn2d[0](concat_max_avg)
        attn_output=self.multiply([inputs,attn_map])
        attn_output=self.add([inputs,attn_output])
        return attn_output

    def max_pool_max_avg(self,inputs):
        max_pool=tf.reduce_max(inputs,axis=-1,keepdims=True)
        avg_pool=tf.reduce_mean(inputs,axis=-1,keepdims=True)
        outputs=tf.concat([max_pool,avg_pool],axis=-1)
                                   
        return outputs
    
class PVT_Model(tf.keras.layers.Layer):
    def __init__(self, config=None,cfd_type=None,poly_coeff_Bu=None,list_idx=[0,1],layer_name=''):
        super(PVT_Model, self).__init__(name=layer_name)
        self.config=config
        self.cfd_type=cfd_type
        self.poly_coeff_Bu=poly_coeff_Bu
        self.list_idx=list_idx
    def build(self, input_shape):
        return
    def call(self, inputs):
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape1:
            tape1.watch(inputs)
            out_PVT=layers_PVT(inputs=inputs,config=self.config,cfd_type=self.cfd_type,poly_coeff_Bu=self.poly_coeff_Bu)
            
        #d_dp_invBg=tape1.gradient(out_PVT[0],inputs, unconnected_gradients='zero') 
        #d_dp_invBo=tape1.gradient(out_PVT[1],inputs, unconnected_gradients='zero') 
        #d_dp_invug=tape1.gradient(out_PVT[2],inputs, unconnected_gradients='zero') 
        #d_dp_invuo=tape1.gradient(out_PVT[3],inputs, unconnected_gradients='zero') 
        #d_dp_Rs=tape1.gradient(out_PVT[4],inputs, unconnected_gradients='zero') 
        #d_dp_Rv=tape1.gradient(out_PVT[5],inputs, unconnected_gradients='zero') 
        #d_dp_PVT=tape1.jacobian(out_PVT,inputs,unconnected_gradients='zero',experimental_use_pfor=True)
        d_dp_PVT=[tape1.gradient(out_PVT[i],inputs, unconnected_gradients='zero') for i in self.list_idx]
        del tape1        
        #d_dp_PVT=[d_dp_invBg,d_dp_invBo,d_dp_invug,d_dp_invuo,d_dp_Rs,d_dp_Rv]
        d_dp_PVT=tf.where(tf.logical_or(tf.math.is_nan(d_dp_PVT), tf.math.is_inf(d_dp_PVT)),tf.zeros_like(d_dp_PVT), d_dp_PVT)
        d_dp_PVT=tf.where(tf.math.equal(self.cfd_type['Fit_Fluid_Derivatives'],True),d_dp_PVT,tf.zeros_like(d_dp_PVT))
        #d_dp_PVT=tf.transpose(d_dp_PVT,perm=[1,0,2,3,4])
        # inputs_pad=tf.pad(inputs,[[0,0],[1,1],[1,1],[0,0]],mode='SYMMETRIC')
        # _ij=inputs_pad[...,1:-1,1:-1,:]; 
        # _i1=inputs_pad[...,1:-1,2:,:]; _i_1=inputs_pad[...,1:-1,:-2,:]
        # _j1=inputs_pad[...,2:,1:-1,:]; _j_1=inputs_pad[...,:-2,1:-1,:] 
        
        # _ih=(_i1+_ij)*0.5; _i_h=(_ij+_i_1)*0.5; 
        # _jh=(_j1+_ij)*0.5; _j_h=(_ij+_j_1)*0.5; 
        # avg_inputs=[_ih,_jh,_i_h,_j_h]
        # avg_PVT=[layers_PVT(inputs=avg_inputs[i],config=self.config,cfd_type=self.cfd_type,poly_coeff_Bu=self.poly_coeff_Bu)[0] for i in [0,1,2,3]]
        return tf.stack(out_PVT,axis=0),d_dp_PVT
    
def layers_Gas_Oil_Saturation(inputs=None,pre=None,config=None,model=None,width=None,depth=None,hlayer_activation_func=None,olayer_activation_func=None,kernel=None,residual_module_par=None,layer_name='',solu_type=''):
    # model: uninitialized model wrapper
    residual_module_par_ps=residual_module_par
    # if solu_type.upper()=='PRESSURE_DEPENDENT':
    #     inputs_pre=normalization_Layer(stat_limits=[14.7,model.cfd_type['Pi']],norm_limits=model.cfd_type['Norm_Limits'],layer_name='pre_normalize')(pre)

    Sgi=1-model.cfd_type['SCAL']['End_Points']['Swmin']

    inputs_list=split_Layer(no_splits=inputs.shape[-1],axis=-1,layer_name='input_split_gas_oil')(inputs)

    if config['ps'][2] is not None:
        residual_module_par_ps['Network_Type']=config['ps'][2].upper()
        width['ps']['Growth_Rate']=1.5 
        if model.cfd_type['Type'].upper()=='PINN' and config['ps'][2]=='cnn2d':
            width['ps']['Bottom_Size']*=0.5;
            residual_module_par_ps['Kernel_Init']='glorot_normal'
            residual_module_par_ps['Latent_Layer']['Width']*=0.5
            residual_module_par_ps['Latent_Layer']['Skip_Conn']=True
        inputs_pre=normalization_Layer(stat_limits=[model.cfd_type['Pi'],3000.],norm_limits=model.cfd_type['Norm_Limits'],layer_name='pre_normalize')(pre)
        inputs_pre=tf.keras.layers.Concatenate(axis=-1)(inputs_list+[inputs_pre])
        sg_hlayer=dnn_sub_block(inputs=inputs,width=width['ps'],depth=depth['ps'][2],activation_func=hlayer_activation_func['ps'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['sg'],name='gsat_split',olayer_act_func=olayer_activation_func['s'],residual_module_par=residual_module_par_ps)

        if config['ps'][2] not in ['cnn2d','resn']:
            sg_int=tf.keras.layers.Dense(1, activation=olayer_activation_func['s'], kernel_initializer=kernel['Init'], name='gsaturation')(sg_hlayer)
            sg=identity_Layer(layer_name='identity_layer_gsaturation')(sg_int)
        else:
            sg=sg_hlayer
            #sg=tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(sg)
            #sg=tf.keras.activations.softmax(sg,axis=1)
            #sg=tf.keras.layers.Activation(tf.nn.sigmoid,name='gsat_activation')(sg)
        #Add a hard layer for the gas saturation 
        if model.cfd_type['Type'].upper()=='PINN':
            #so=restep_Layer(limits=[model.cfd_type['Dew_Point']],layer_name='pre_oil_saturation_comp_restep_layer')([pre,sg])
            pdew=model.cfd_type['Dew_Point']
            pi=model.cfd_type['Pi']
            lisht=lambda x,y: lambda z: tf.math.maximum(tf.math.minimum(tfa.activations.lisht(z),y),x);
            linear=lambda x,y: lambda z: tf.math.maximum(tf.math.minimum(z,y),x);
            prelu=lambda x: lambda z: tf.nn.relu(x)**(x)
            softplus1=lambda x: lambda z: tf.math.minimum(tf.nn.softplus(z),x)
            scale_elu=lambda x: lambda z: tf.minimum(tf.keras.layers.ELU(alpha=x)(tf.math.tanh(z)))
            softplus_relu=lambda x: tf.nn.softplus(tf.nn.relu(x))
            binary_func=binary_Layer(limits=[0.,],values=[0.,1.],layer_name='binary_activation_layer')
            tsoftplus_func=tsoftplus_Layer(layer_name='tsoftplus_activation_layer')
            sigmoid1=tfp.bijectors.Sigmoid(low=0.,high=0.4, validate_args=False, name='sigmoid')
            sigmoid2=lambda x: lambda z: tf.math.divide_no_nan(1. ,(1. + tf.math.exp(-z)))**x
            scaled_sigmoid=lambda x,y: lambda z: (y-x)*tf.nn.sigmoid(z)+x;

            sigmoidt=sigmoid_Layer(layer_name='gsat_trainable_sigmoid_activation_layer',alpha=[1.,1.],trainable=[False,False],lower_limit=[1.,10.],upper_limit=[10.,100.])
            hardlayer_act_func=tfp.bijectors.Sigmoid(low=0.,high=0.4, validate_args=True, name='rbf_sigmoid_activation')
            rbf_func=tfp.bijectors.Sigmoid(low=1.,high=5., validate_args=True, name='rbf_sigmoid_activation')
            #sg=output_Hardlayer(norm_limits=[[pdew,14.7],[-1,1]],init_value=Sgi,kernel_activation=[tf.nn.relu,None],input_activation=hardlayer_act_func,layer_name='gsaturation_hardlayer',regularization=0.00,weights=[],kernel_exponent=[{'Value':0.5,'Trainable':True,'Lower_Limit':0.,'Upper_Limit':0.99,'Rate':1.},])([[inputs,],sg])
            #sg=output_Hardlayer(norm_limits=[[-1,1],[pdew,14.7]],co_var='no_correlate',init_value=Sgi,kernel_activation=[None,tf.nn.relu],input_activation=hardlayer_act_func,layer_name='gsaturation_hardlayer',regularization=0.00,weights=[],kernel_exponent=[{'Value':0.5773/3,'Trainable':True,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.},{'Value':1.,'Trainable':False,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.}],model=model,rbf_activation_function=rbf_func)([[*inputs_list[-2:],pre],sg])
            #sg=output_Hardlayer(norm_limits=[[-1,1],[pdew,14.7]],co_var='correlate',init_value=Sgi,kernel_activation=[None,tf.nn.relu],input_activation=hardlayer_act_func,layer_name='gsaturation_hardlayer',regularization=0.00,weights=[],kernel_exponent=[{'Value':0.5773/3,'Trainable':True,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.},{'Value':1.,'Trainable':False,'Lower_Limit':0.1,'Upper_Limit':1.,'Rate':1.}],model=model)([[*inputs_list[-2:],pre],sg])
            sg=gas_oil_Hardlayer(norm_limits=[[-1,1],[pdew,14.7]],kernel_activation=[None,binary_func],init_value=Sgi,input_activation=tf.nn.sigmoid,layer_name='gsaturation_hardlayer')([[inputs_list[-2],inputs_list[-1],pre],sg])           #sg=multiply_Layer(layer_name='gas_saturation_hardlayer_scaled')(sg,y=1.,constraint=[(Sgi-0.3),Sgi])
    else:
        #sg=constant_Layer(constant=Sgi,layer_name='unmodelled_layer_gsaturation',mean=0.,stddev=0.)(inputs)
        sg=constant_Layer(constant=Sgi,layer_name='unmodelled_layer_gsaturation',mean=0.,stddev=0.)(inputs_list[-2])
    out_sat=[sg] 

    # Oil Saturation
    if model.cfd_type['Fluid_Type'] in ['gas_cond','GC']:
        if config['ps'][3] is not None:
            residual_module_par_ps['Network_Type']=config['ps'][3].upper()
            so_hlayer=dnn_sub_block(inputs=inputs,width=width['ps'],depth=depth['ps'][3],activation_func=hlayer_activation_func['ps'],kernel_init=kernel['Init'],kernel_regu=kernel['Regu']['so'],name='osat_split',residual_module_par=residual_module_par_ps)
            if config['ps'][3]!='cnn2d':
                so_int=tf.keras.layers.Dense(1, activation=olayer_activation_func['s'], kernel_initializer=kernel['Init'], name='osaturation')(so_hlayer)
                #so=tf.keras.layers.Lambda(lambda x: iden_fun(x),name='identity_layer_osaturation')(so_int)
                so=identity_Layer(layer_name='identity_layer_osaturation')(so_int)
            else:
                so=so_hlayer           
        else:
            #so_hlayer=None
            #so=tf.keras.layers.Lambda(lambda x: 1-x-cfd_type['SCAL']['End_Points']['Swmin'],name='unmodelled_layer_osaturation')(sg)
            so=(constantDiff_Layer(constant=(1-model.cfd_type['SCAL']['End_Points']['Swmin']),layer_name='unmodelled_layer_osaturation')((sg)))
        out_sat+=[so]
    return out_sat


class saturation_Gas_Oil_Model(tf.keras.layers.Layer):
    def __init__(self, config=None,cfd_type=None,width=None,depth=None,olayer_activation_func=None,kernel=None,residual_module_par=None,pressure_dependence=True,hard_enforcement=False,compute_derivatives=True,reshape_output=None,layer_name=''):
        super(saturation_Gas_Oil_Model, self).__init__(name=layer_name)
        self.config=config
        self.cfd_type=cfd_type
        self.width=width
        self.depth=depth
        self.olayer_activation_func=olayer_activation_func
        self.kernel=kernel
        self.res_par=residual_module_par
        self.pres_dep=pressure_dependence
        self.hard_enforcement=hard_enforcement
        self.compute_derivatives=compute_derivatives
        self.reshape_output=reshape_output
        self.trainable_kernel=True
        self.idx_list=[0]
        self.hardlayer_act_func=lambda x,y: lambda z: tf.math.maximum(tf.math.minimum(tfa.activations.lisht(z),y),x)
        self.Sgi=(1-self.cfd_type['SCAL']['End_Points']['Swmin'])
        if self.pres_dep:
            self.layer_pre_norm=normalization_Layer(stat_limits=[1000.,cfd_type['Pi']],norm_limits=cfd_type['Norm_Limits'],layer_name='pre_normalize')
            
        # Gas Saturation
        if self.config[2] is not None:
            self.res_par['Network_Type']=self.config[2].upper()
            if self.cfd_type['Type'].upper()=='PINN': 
                self.width['Bottom_Size']*=1.
            self.layers_gsat=dnn1.RNN_Layer(depth=self.depth[2],width=self.width,kernel_size=3,res_par=self.res_par,layer_name='gas_saturation',gaussian_process=None,out_act_func=self.olayer_activation_func['s'],batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
            
            #Add a hard layer for the gas saturation
            if self.cfd_type['Type'].upper()=='PINN' and self.hard_enforcement:
                #tf.keras.layers.ELU(alpha=0.5)  #scaled_relu_Layer(limits=[0.,0.4],layer_name='gsat_scaled_relu_int_output')
                self.layers_gsat=dnn1.RNN_Layer(depth=self.depth[2],width=self.width,kernel_size=3,res_par=self.res_par,layer_name='gas_saturation',gaussian_process=None,out_act_func=None,batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
                self.hardlayer_gsat=output_Hardlayer(norm_limits=[[self.cfd_type['Dew_Point'],14.7],],init_value=self.Sgi,kernel_activation=[tf.nn.relu,],input_activation=None, layer_name='gsaturation_hardlayer',regularization=0.00,weights=[],kernel_exponent=[{'Value':0.5777/3,'Trainable':True,'Lower_Limit':0.1,'Upper_Limit':0.99,'Rate':1.},])
                #self.hardlayer_gsat=output_Hardlayer(norm_limits=[-1,1],init_value=1-self.cfd_type['SCAL']['End_Points']['Swmin'],kernel_activation=None,input_activation=self.hardlayer_act_func(0.001,0.5), layer_name='gsaturation_hardlayer',regularization=0.00,weights=[],kernel_exponent={'Value':0.5,'Trainable':True})
        else:
            self.layers_gsat=constant_Layer(constant=self.Sgi,layer_name='unmodelled_layer_gsaturation',mean=0.,stddev=0.)

        # Oil Saturation
        if self.cfd_type['Fluid_Type'] in ['gas_cond','GC']:
            if self.config[3] is not None:
                self.res_par['Network_Type']=self.config[2].upper()
                self.layers_osat=dnn1.RNN_Layer(depth=self.depth[3],width=self.width,kernel_size=3,res_par=self.res_par,layer_name='oil_saturation',gaussian_process=None,out_act_func=self.olayer_activation_func['s'],batch_norm={'Use_Batch_Norm':False,},dropout={'Add':False,},attn={})
            else:
                self.layers_osat=constantDiff_Layer(constant=self.Sgi,layer_name='unmodelled_layer_osaturation') 
            self.idx_list=[0,1]
                
    def build(self, input_shape):
        return
  
    def call(self, inputs,):
        #inputs=tf.reshape(inputs,(-1,1))
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape1:
            tape1.watch(inputs)
        #     out_S=layers_Gas_Oil_Saturation(inputs=inputs,config=self.config,cfd_type=self.cfd_type,width=self.width,depth=self.depth,hlayer_activation_func=self.hlayer_activation_func,\
        #                                     olayer_activation_func=self.olayer_activation_func,kernel=self.kernel,residual_module_par=self.residual_module_par,layer_name='')
            hlayer=self.compute_saturation(inputs)
            outputs=hlayer
        if self.compute_derivatives:
            d_dp_S=[tape1.gradient(hlayer[i],inputs, unconnected_gradients='zero') for i in self.idx_list]
            del tape1 
            if self.reshape_output is not None:
                d_dp_S=[tf.reshape(i,(-1,*self.reshape_output)) for i in d_dp_S]
            outputs=tf.stack(hlayer),tf.stack(d_dp_S)
        return outputs
    
    @tf.function(jit_compile=True)
    def compute_saturation(self,inputs):
        pre_hlayer=inputs
        if self.config[2] is not None:
            #pre_hlayer=self.layer_pre_norm(pre_hlayer)
            hlayer_gsat=self.layers_gsat(pre_hlayer,output_layer=True) 
            # Gas Saturation
            if self.cfd_type['Type'].upper()=='PINN' and self.hard_enforcement:
                #hlayer_gsat_ad=self.layers_gsat_ad(pre_hlayer)
                #hlayer_gsat_bd=hlayer_gsat
                #hlayer_gsat=(tf.cast(inputs>self.cfd_type['Dew_Point'],self.dtype)*tf.zeros_like(inputs))+(tf.cast(inputs<=self.cfd_type['Dew_Point'],self.dtype)*hlayer_gsat)
                hlayer_gsat=self.hardlayer_gsat([[inputs,],hlayer_gsat])
        else:
            hlayer_gsat=self.layers_gsat(pre_hlayer) 
    
        hlayer=[hlayer_gsat]        
        # Oil Saturation
        if self.cfd_type['Fluid_Type'] in ['gas_cond','GC']:
            if self.config[3] is not None:
                hlayer_osat=self.layers_osat(pre_hlayer,output_layer=True)
            else:
                hlayer_osat=self.layers_osat(hlayer_gsat)  #Difference layer with the gas saturation 
            hlayer=[hlayer_gsat,hlayer_osat]   
        
        if self.reshape_output is not None:
            hlayer=[tf.reshape(i,(-1,*self.reshape_output)) for i in hlayer]
        
        return hlayer       


    
# Layer class activation function wrapper
def act_func_name_wrapper(act_func):
    if not hasattr(act_func,'__name__') and hasattr(act_func,'_instrumented_keras_layer_class'):
        act_func.__name__=act_func.name[:3]+str(act_func.alpha).replace('.','.')
    return act_func
#============================================================================================================================================
# Swish activation function with beta adjustment
def swish_beta(beta=None):
    @tf.custom_gradient
    def swish(features):
        features=tf.convert_to_tensor(features, name='features')
        beta_=tf.convert_to_tensor(beta, dtype=features.dtype, name='beta') 
        def grad(dy):
            "Gradient for the swish activation function"
            with tf.control_dependencies([dy]):
                sigmoid_features = tf.math.divide(1.0,1.0+tf.math.exp(-beta_*features))
                activation_grad = (sigmoid_features * (1.0 + (features*beta_) * (1.0 - sigmoid_features)))
            return dy * activation_grad

        return tf.math.divide(features,1.0+tf.math.exp(-beta_*features)),grad
    
    swish.__name__='swish'+str(beta).replace('.','_')
    return swish

# Hard limit activation function
def hard_limit_func(lower_limit=14.7,upper_limit=5000., alpha=0.0):
    #@tf.custom_gradient
    def hard_limit(features):
        features=tf.convert_to_tensor(features, name='features')
        lower_limit_=tf.convert_to_tensor(lower_limit, dtype=features.dtype, name='lower_limit') 
        upper_limit_=tf.convert_to_tensor(upper_limit, dtype=features.dtype, name='upper_limit') 
        alpha_=tf.convert_to_tensor(alpha, dtype=features.dtype, name='alpha') 
        #breakpoint()
        def f1(): return lower_limit_
        def f2(): return upper_limit_
        def f3(): return features

        def grad(dy):
            "Gradient for the swish activation function"
            with tf.control_dependencies([dy]):
                #activation_grad = tf.case([(tf.less(features, lower_limit_), lambda: tf.zeros_like(features)), (tf.greater(features, upper_limit_), lambda: tf.zeros_like(features))],
                #default=lambda: tf.ones_like(features), exclusive=False)
                activation_grad=tf.where(tf.less(features, lower_limit_),tf.ones_like(features)*alpha_,tf.where(tf.greater(features, upper_limit_),tf.ones_like(features)*alpha,tf.ones_like(features)))
            return dy * activation_grad
        
        #tf.case([(tf.less(features, lower_limit_), f1), (tf.greater(features, upper_limit_), f2)],default=f3, exclusive=False),grad
        r1=tf.where(tf.less(features, lower_limit_),lower_limit_+tf.math.abs(features-lower_limit_)*alpha_,tf.where(tf.greater(features, upper_limit_),upper_limit_+(features-upper_limit_)*alpha_,features))
        return r1#,grad
    
    hard_limit.__name__='hard_limit'+str(lower_limit)+'_'+str(upper_limit).replace('.','_')
    return hard_limit

# Sigmoid activation function with pressure adjustment
def sigmoidp_func(lower_limit=14.7,upper_limit=5000.):
    @tf.custom_gradient
    def sigmoidp(features):
        features=tf.convert_to_tensor(features, name='features')
        lower_limit_=tf.convert_to_tensor(lower_limit, dtype=features.dtype, name='lower_limit') 
        upper_limit_=tf.convert_to_tensor(upper_limit, dtype=features.dtype, name='upper_limit') 
        def grad(dy):
            "Gradient for the swish activation function"
            with tf.control_dependencies([dy]):
                sigmoid_features = tf.math.divide(1.0,1.0+tf.math.exp(-features))
                activation_grad = (upper_limit_-lower_limit_)*(sigmoid_features * (1.0 - sigmoid_features))
            return dy * activation_grad

        return lower_limit_+(upper_limit_-lower_limit_)*tf.math.divide(1.0,1.0+tf.math.exp(-features)),grad
    
    sigmoidp.__name__='sigmoidp'+str(lower_limit)+'_'+str(upper_limit).replace('.','_')
    return sigmoidp

# Sine activation function with pressure adjustment
def sinep_func(lower_limit=14.7,upper_limit=5000.):
    @tf.custom_gradient
    def sinep(features):
        features=tf.convert_to_tensor(features, name='features')
        lower_limit_=tf.convert_to_tensor(lower_limit, dtype=features.dtype, name='lower_limit') 
        upper_limit_=tf.convert_to_tensor(upper_limit, dtype=features.dtype, name='upper_limit') 
        def grad(dy):
            "Gradient for the swish activation function"
            with tf.control_dependencies([dy]):
                activation_grad = (upper_limit_-lower_limit_)*tf.math.cos(features)*(0.5)
            return dy * activation_grad

        return lower_limit_+((upper_limit_-lower_limit_)*(tf.math.sin(features)+1)*0.5),grad
    
    sinep.__name__='sinep'+str(lower_limit)+'_'+str(upper_limit).replace('.','_')
    return sinep

# tanh activation function with pressure adjustment
def tanhp_func(lower_limit=14.7,upper_limit=5000.):
    @tf.custom_gradient
    def tanhp(features):
        features=tf.convert_to_tensor(features, name='features')
        lower_limit_=tf.convert_to_tensor(lower_limit, dtype=features.dtype, name='lower_limit') 
        upper_limit_=tf.convert_to_tensor(upper_limit, dtype=features.dtype, name='upper_limit') 
        def grad(dy):
            "Gradient for the swish activation function"
            with tf.control_dependencies([dy]):
                activation_grad = (upper_limit_-lower_limit_)*(1-(tf.math.tanh(features))**2)*(0.5)
            return dy * activation_grad

        return lower_limit_+((upper_limit_-lower_limit_)*(tf.math.tanh(features)+1)*0.5),grad
    
    tanhp.__name__='tanhp'+str(lower_limit)+'_'+str(upper_limit).replace('.','_')
    return tanhp

def emish(x) -> tf.Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function."""

    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.math.exp(x))

def softplusb(beta=1.):
    """Softplus with beta"""
    def softplus(features):
        features=tf.convert_to_tensor(features, name='features')
        beta_=tf.convert_to_tensor(beta, name='beta')
        return (1/beta_)*tf.math.log(1+tf.math.exp(beta_*features))
    return softplus

@tf.function
def normalize_diff(model,diff,stat_idx=0,compute=False):
    # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
    #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
    #                           Nonnormalized function: Linear scaling (a,b)= (xmax-xmin)*((x_norm-a)/(b-a))+xmin
    #                           Nonnormalized function: z-score= (x_norm*xstd)+xmean
    diff=tf.convert_to_tensor(diff, dtype=model.dtype, name='diff')
    
    def _lnk_linear_scaling():
        lin_scale_no_log=((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]))*diff
        lin_scale_log=((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/tf.math.log(model.ts[stat_idx,1]/model.ts[stat_idx,0]))*tf.math.log(diff)

        return tf.cond(tf.logical_and(tf.math.not_equal(stat_idx,5),tf.math.not_equal(stat_idx,6)),lambda: lin_scale_no_log, lambda: lin_scale_log)

    def _linear_scaling():
        return ((model.cfd_type['Norm_Limits'][1]-model.cfd_type['Norm_Limits'][0])/(model.ts[stat_idx,1]-model.ts[stat_idx,0]))*diff
    
    def _z_score():
        return (1/model.ts[stat_idx,3])*diff
    
    norm=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: diff)
    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm

@tf.function
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
    
    norm=tf.cond(tf.math.equal(compute,True),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'linear-scaling'),lambda: _linear_scaling(),lambda: tf.cond(tf.math.equal(model.cfd_type['Input_Normalization'],'lnk-linear-scaling'),lambda: _lnk_linear_scaling(),lambda: _z_score())),lambda: nonorm_input)
    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    norm=tf.where(tf.logical_or(tf.math.is_nan(norm), tf.math.is_inf(norm)),tf.zeros_like(norm), norm)
    return norm

#@tf.function
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
    if compute:
        if model.cfd_type['Input_Normalization']=='lnk-linear-scaling':
            nonorm=_linear_scaling()
        elif model.cfd_type['Input_Normalization']=='linear-scaling':
            nonorm=_lnk_linear_scaling()
        else:
            nonorm=_z_score()
    else:
        nonorm=norm_input    
    # Dropsout the derivative in an event of a nan number--when the min and max statistics are constant or standard deviation is zero
    return tf.where(tf.logical_or(tf.math.is_nan(nonorm), tf.math.is_inf(nonorm)),tf.zeros_like(nonorm), nonorm)

def relative_permeability(endpoints={'kro_Somax':0.90,'krg_Sorg':0.80,'krg_Swmin':0.90,'Swmin':0.22,'Sorg':0.2,'Sgc':0.05},corey_exp={'nog':3.,'ng':6.,'nw':2.},sat_threshold=0.,socr=0.2,so_max=0.28,dtype=tf.float32):
    # Relative permeability using Corey functions
    kro_somax=tf.convert_to_tensor(endpoints['kro_Somax'],dtype=dtype)
    krg_sorg=tf.convert_to_tensor(endpoints['krg_Sorg'],dtype=dtype)
    krg_swmin=tf.convert_to_tensor(endpoints['krg_Swmin'],dtype=dtype)
    
    swmin=tf.convert_to_tensor(endpoints['Swmin'],dtype=dtype)
    sorg=tf.convert_to_tensor(endpoints['Sorg'],dtype=dtype)
    sgc=tf.convert_to_tensor(endpoints['Sgc'],dtype=dtype)  # Useful in modelling gas condensates
    nog=tf.convert_to_tensor(corey_exp['nog'],dtype=dtype)
    ng=tf.convert_to_tensor(corey_exp['ng'],dtype=dtype)
    nw=tf.convert_to_tensor(corey_exp['nw'],dtype=dtype)
    sth=tf.convert_to_tensor(sat_threshold,dtype=dtype)
    socr=tf.math.maximum(tf.convert_to_tensor(socr,dtype=dtype),sorg)
    so_max=tf.convert_to_tensor(so_max,dtype=dtype)
    def compute_krog_krgo(sg):
        #tf.Assert(krg_swmin>=krg_sorg,[krg_swmin])
        #sg=(tf.cast(sg<0.,dtype)*0.)+(tf.cast((sg>=0.)&(sg<=1.),dtype)*sg)+(tf.cast(sg>1.,dtype)*1.)
        
        krog=kro_somax*(((1-sg-swmin)-sorg)/(1.-swmin-sorg))**nog  #(1-sg)-swmin
        krgo=krg_sorg*((sg-sgc)/(1.-sgc-swmin-sorg))**ng
        #krog_max=kro_somax*((so-sorg)/(1.-swmin-sorg))**nog
        
        # Endpoints saturation checks
        krog=(tf.cast((1-sg)<=(swmin+sorg),dtype)*0.)+(tf.cast((1-sg)>(swmin+sorg),dtype)*krog)
        #krog=(tf.cast((1-sg)<=(swmin+socr),dtype)*0.)+(tf.cast((1-sg)>(swmin+socr),dtype)*krog)
        krgo=(tf.cast(sg>(1-(swmin+sorg)),dtype)*krg_swmin)+(tf.cast(sg<=(1-(swmin+sorg)),dtype)*krgo)
        
        # Checks for Rel Perm values greater than the endpoints
        krog_ul=tf.math.minimum(krog,kro_somax)
        krog=(tf.cast(krog_ul>sth,dtype)*krog_ul)+(tf.cast(krog_ul<=sth,dtype)*0.)
        #krog=tf.math.maximum(tf.math.minimum(krog,kro_somax),0.)
        krgo=tf.math.maximum(tf.math.minimum(krgo,krg_swmin),0.)
        return krog,krgo
    return compute_krog_krgo
#=============================================================================================================================================================
def cost_func_Sg(_invBgug_n0_ij,_invBouo_n0_ij,_Rs_n0_ij,_Rv_n0_ij,Mo_Mg_n1_ij,_q_well_idx=None,rel_perm_model=None,_eps=1.0e-16,return_gradient=False):
    def a(Sg_n0_ij):
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tapeS:
            tapeS.watch(Sg_n0_ij)
            _krog_n0_ij,_krgo_n0_ij=rel_perm_model(Sg_n0_ij)
            _tmg_n0_ij=(_krgo_n0_ij*_invBgug_n0_ij)+(_krog_n0_ij*_invBouo_n0_ij)*_Rs_n0_ij
            _tmo_n0_ij=(_krog_n0_ij*_invBouo_n0_ij)+(_krgo_n0_ij*_invBgug_n0_ij)*_Rv_n0_ij
            cost=(_q_well_idx*(tf.math.divide(_tmo_n0_ij,_tmg_n0_ij+_eps)-Mo_Mg_n1_ij))
        d_Sg=tapeS.gradient(cost,Sg_n0_ij, unconnected_gradients='zero')
        del tapeS
        output=cost
        if return_gradient:
            output=cost,d_Sg
        return output  
    return a

def kro_krg_Sg(rel_perm_model=None,num_points=50,PVT=None):

    # #kro_krg=[tf.math.log(rel_perm_model(Sg[i])[0]/rel_perm_model(Sg[i])[1]+1e-9) for i in range(tf.shape(Sg)[0])]
    # kro_krg=[(rel_perm_model(Sg[i])[0]/rel_perm_model(Sg[i])[1]) for i in range(tf.shape(Sg)[0])]
    # #kro_krg=tf.where(tf.math.is_inf(kro_krg),tf.zeros_like(kro_krg),kro_krg)
    # reshape=(1,num_points,1)
    # Sg=tf.reshape(Sg,reshape)
    # kro_krg=tf.reshape(kro_krg,reshape)
    # kro_krg_Sg_model=spline_Interpolation(points_x=kro_krg,values_y=Sg,order=2,layer_name='')
    
    import find_root_chandrupatla as chp
    import find_root_levenberg_marquardt as lm
    import scipy
    rel_perm_model=model.cfd_type['Kr_gas_oil']
    #Sg=tf.reshape(tf.linspace(0.,1.,num_points),(-1,))
    PVT_0=model.PVT(tf.constant([2000.,3850.,2150.],shape=(3,1)))
    #krog,krgo=rel_perm_model(_Sgi)
    invBgug=PVT_0[0][0]*PVT_0[0][2]
    invBouo=PVT_0[0][1]*PVT_0[0][3]               
    Rs=PVT_0[0][4] 
    Rv=PVT_0[0][5] 
    mo_mg=tf.constant([0.67569095,0.8572296,0.67003226],shape=(3,1,1))

    krog,krgo=rel_perm_model(tf.constant([0.2,0.4,0.1],shape=(3,1,1)))
    tmg=(krgo*invBgug)+(krog*invBouo)*Rs
    tmo=(krog*invBouo)+(krgo*invBgug)*Rv
    _mo_mg=tf.math.divide_no_nan(tmo,tmg)
    cost=(tf.math.divide_no_nan(tmo,tmg)-mo_mg)        
    
    cf=cost_func_Sg(invBgug,invBouo,Rs,Rv,mo_mg,_q_well_idx=1,rel_perm_model=rel_perm_model,return_gradient=False)
    a=Newton_Raphson_ld(obj_func=cf,no_iter=5,init_like=tf.zeros_like(Rs),lower=0.,upper=1.,ld=0.001)
    a=chp.find_root_chandrupatla(cf,low=tf.zeros_like(Rs),high=tf.ones_like(Rs),position_tolerance=1e-08,value_tolerance=0.0,max_iterations=5,stopping_policy_fn=tf.reduce_all,validate_args=False,name='find_root_chandrupatla')
    a=tfp.math.secant_root(cf,tf.ones_like(Rs)*1.)
    a=lm.minimize(cf,tf.ones_like(Rs)*0.,10)
    scipy.optimize.fsolve(cf, np.ones_like(Rs)*0.5, fprime=None, args=(), )
    return 

kro_krg_values=tf.constant(1.5,shape=[35,39,39,1],dtype=tf.float32)

def obj_func_rel_perm(rel_perm_model=None,values=kro_krg_values,eps=1e-16):
     def func(Sg):
         with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape1:
             tape1.watch(Sg)
             c1=(tf.math.divide(rel_perm_model(Sg)[0],rel_perm_model(Sg)[1]+1e-16))
             cost=(c1-(values))
         d_Sg=tape1.gradient(cost,Sg, unconnected_gradients='zero')
         del tape1
         return cost,d_Sg
     func.shape=values.shape
     return func   

#@tf.function
def Newton_Raphson_Bisection(obj_func=None,no_iter=[0,1,2,3,4],init_val=0.5,lower=0.,upper=1.):
    lower=(tf.ones_like(init_val))*lower
    upper=(tf.ones_like(init_val))*upper
    sign_lower=tf.sign(lower)
    sign_upper=tf.sign(upper)

    init_val=(lower+upper)*0.5
    i, result = tf.constant(0), tf.convert_to_tensor(init_val)

    #while i < no_iter: # AutoGraph converts while-loop to tf.while_loop().
    for i in range(no_iter):
        result1=result-tf.math.divide_no_nan(obj_func(result)[0],obj_func(result)[1])
        print(obj_func(result)[1])
        c1=tf.cast(((result1<lower)|(result1>upper)),init_val.dtype)
        c1_1=c1*tf.cast(((tf.sign(obj_func(result)[0])*sign_lower)>=0.),init_val.dtype)
        c1_2=c1*tf.cast(((tf.sign(obj_func(result)[0])*sign_lower)<0.),init_val.dtype)
        c2=tf.cast(((result1>=lower)&(result1<=upper)),init_val.dtype)
        
        lower=result*c1_1+(1.-c1_1)*lower
        upper=result*c1_2+(1.-c1_2)*upper
        lower=c3*lower+(1-c3)*0.5
        upper=c3*upper+(1-c3)*0.5
        result=c2*result1+c1*(upper+lower)*0.5
        #i+=1
    return result

def Newton_Raphson_ld(obj_func=None,no_iter=5,init_like=0.5,lower=0.,upper=1.,ld=0.001,tol=0.001,_back_prop=False):
    lower=(tf.ones_like(init_like))*lower
    upper=(tf.ones_like(init_like))*upper
    ld=tf.ones_like(init_like)*ld
    sign_lower=tf.sign(lower)
    sign_upper=tf.sign(upper)
    
    init_like=(lower+upper)*0.5
    
    i, x_init,ld = tf.constant(0), tf.convert_to_tensor(init_like), tf.convert_to_tensor(ld)
    c = lambda i,_: tf.less(i,no_iter)
    
    #Reinitialize by bisection method - lower step rate - better convergence
    fa,dfa=tfp.math.value_and_gradient(lambda :obj_func(lower),lower,use_gradient_tape=True)
    fm,dfm=tfp.math.value_and_gradient(lambda :obj_func(x_init),x_init,use_gradient_tape=True)
    f_df=ld*tf.math.divide_no_nan(fm,dfm)
    
    upper=tf.where(fa*fm<0.,init_like,upper)
    lower=tf.where(fa*fm>=0.,init_like,lower)
    #ld=tf.where(tf.sign(dfa)*tf.sign(dfm)>=0.,ld*0.001,ld)

    #x_init-=f_df
    x_init=tf.math.maximum((upper+lower)*0.5,x_init-f_df)

    def b(i,x):
        f,df=tfp.math.value_and_gradient(lambda :obj_func(x[0]),x[0],use_gradient_tape=True)
        f_df=tf.math.divide_no_nan(f,df)
        
        c1=tf.cast(tf.math.abs(f_df)<(tol),init_like.dtype)
        x[3]=1.#c1*(ld/1)+(1-c1)*(ld*1)
        x_new=x[0]-x[3]*f_df
        
        #c1=tf.cast(((x_new<x[1])|(x_new>x[2])),init_like.dtype)
        ##c1=tf.cast(tf.math.abs(f_df)<(0.01*(x[2]-x[1])),init_like.dtype)
        ##c1_l=tf.cast(x_new<x[1],init_like.dtype)
        ##c1_u=tf.cast(x_new>x[2],init_like.dtype)
        # c1_1=c1*tf.cast(((tf.sign(obj_func(x[0])[0])*sign_lower)>=0.),init_like.dtype)
        # c1_2=c1*tf.cast(((tf.sign(obj_func(x[0])[0])*sign_lower)<0.),init_like.dtype)

        ##c2=tf.cast(((x_new>=x[1])&(x_new<=x[2])),init_like.dtype)
    
        # x[1]=x[0]*c1_1+(1.-c1_1)*x[1]
        # x[2]=x[0]*c1_2+(1.-c1_2)*x[2]
        # x[0]=c2*x_new+c1*(x[2]+x[1])*0.5
        

        x[0]=x_new
        
        ##c_end=tf.cast(i==(no_iter-1),init_like.dtype)
        ##x[0]=(c_end*((c1_l*x[1])+(c1_u*x[2])+(c2*x[0])))+(1.-c_end)*x[0]
        return i+1,x
    out=tf.while_loop(c, b, (i, [x_init,lower,upper,ld]),maximum_iterations=no_iter,back_prop=_back_prop)[1][0]
    #out=chp.find_root_chandrupatla(cf,low=out,high=tf.ones_like(Rs),position_tolerance=1e-08,value_tolerance=0.0,max_iterations=5,stopping_policy_fn=tf.reduce_all,validate_args=False,name='find_root_chandrupatla')

    return out

def Newton_Raphson_Bisection(obj_func=None,no_iter=5,init_like=0.5,lower=0.,upper=0.77,eps=1e-16,back_prop=False):
    lower=(tf.ones_like(init_like))*lower
    upper=(tf.ones_like(init_like))*upper
    
    sign_lower=tf.sign(lower)
    sign_upper=tf.sign(upper)
    init_val=(lower+upper)*0.5
    i, x_init = tf.constant(0), tf.convert_to_tensor(init_val)
    c = lambda i,_: tf.less(i,no_iter)

    def b(i,x):
        x_new=x[0]-tf.math.divide_no_nan(obj_func(x[0])[0],obj_func(x[0])[1])
        c1=tf.cast(((x_new<x[1])|(x_new>x[2])),init_val.dtype)
        c1_1=c1*tf.cast(((tf.sign(obj_func(x[0])[0])*sign_lower)>=0.),init_val.dtype)
        c1_2=c1*tf.cast(((tf.sign(obj_func(x[0])[0])*sign_lower)<0.),init_val.dtype)
        c2=tf.cast(((x_new>=x[1])&(x_new<=x[2])),init_val.dtype)

        x[1]=x[0]*c1_1+(1.-c1_1)*x[1]
        x[2]=x[0]*c1_2+(1.-c1_2)*x[2]
        x[0]=c2*x_new+c1*(x[2]+x[1])*0.5

        return i+1,x
    return tf.while_loop(c, b, (i, [x_init,lower,upper]),maximum_iterations=no_iter,back_prop=back_prop)[1][0]


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = tf.cast(gamma,self.dtype)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='glorot_normal',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs,-1) - self.mu
        l2 = tf.reduce_sum(tf.math.pow(diff, 2), axis=1)
        res = tf.math.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
