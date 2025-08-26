import os

import copy
import torch
import joblib
import random
import json
import math
import sys
import logging
import argparse
import numpy as np
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold 
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from mae_model_graph import fusion_model_mae_2
from util import Logger, get_patients_information,get_all_ci,get_val_ci,adjust_learning_rate,get_patients_information_new,get_new_dict,get_changed_form_dict
from mae_utils_graph import generate_mask
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_same_ones(arr1, arr2):
    count = 0
    for i in range(len(arr1)):
        if arr1[i] == 1 and arr2[i] == 1:
            count += 1
    return count






def prediction(all_data,v_model,val_id,patient_and_time,patient_sur_type,args):
    v_model.eval()
    
    img_feature_val = torch.empty((0)).to(device)
    rna_feature_val = torch.empty((0)).to(device)
    cli_feature_val = torch.empty((0)).to(device)

    x_img_feature_val = torch.empty((0)).to(device)
    x_rna_feature_val = torch.empty((0)).to(device)
    x_cli_feature_val = torch.empty((0)).to(device)

    

    val_pre_time_graph = {}
    lbl_pred_all = None
    status_all = []
    survtime_all = []
    val_pre_time = {}
    val_pre_time_img = {}
    val_pre_time_rna = {}
    val_pre_time_cli = {}
    iter = 0
    
    with torch.no_grad():
        for i_batch, id in enumerate(val_id):

            graph = all_data[id].to(device)
            if args.train_use_type != None:
                use_type_eopch = args.train_use_type
            else:
                use_type_eopch = graph.data_type
            out_pre,out_fea,out_att,dict,x_feature = v_model(graph,args.train_use_type,use_type_eopch,mix=args.mix)
            lbl_pred = out_pre[0]
            img_feature_val = torch.cat((img_feature_val,dict['img'].unsqueeze(0)),0)
            rna_feature_val = torch.cat((rna_feature_val,dict['rna'].unsqueeze(0)),0)
            cli_feature_val = torch.cat((cli_feature_val,dict['cli'].unsqueeze(0)),0)
            
            x_img_feature_val = torch.cat((x_img_feature_val,dict['mae_labels'][0].unsqueeze(0)),0)
            x_rna_feature_val = torch.cat((x_rna_feature_val,dict['mae_labels'][1].unsqueeze(0)),0)
            x_cli_feature_val = torch.cat((x_cli_feature_val,dict['mae_labels'][2].unsqueeze(0)),0)


            survtime_all.append(patient_and_time[id])
            status_all.append(patient_sur_type[id])

            val_pre_time[id] = lbl_pred.cpu().detach().numpy()[0]
            if iter == 0 or lbl_pred_all == None:
                lbl_pred_all = lbl_pred
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])

            iter += 1
            
            if 'img' in use_type_eopch:
                val_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
            if 'rna' in use_type_eopch:
                val_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()            
            if 'cli' in use_type_eopch:
                val_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy() 


        img_feature_flag = torch.ones(img_feature_val.shape[0]).to(device)
        rna_feature_flag = torch.ones(rna_feature_val.shape[0]).to(device)
        cli_feature_flag = torch.ones(cli_feature_val.shape[0]).to(device)

        risk_val = v_model.inference(img_feature_val,img_feature_flag,rna_feature_val,rna_feature_flag,cli_feature_val,cli_feature_flag,x_img_feature_val,x_rna_feature_val,x_cli_feature_val)
        
        
        for i in range(len(val_id)):
            val_pre_time_graph[val_id[i]] = risk_val.squeeze()[i].cpu().detach().numpy()
        
        

    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)
#     print(lbl_pred_all,survtime_all,status_all)
    loss_surv = _neg_partial_log(risk_val, survtime_all, status_all)
    loss = loss_surv

    val_ci_ = get_val_ci(val_pre_time_graph,patient_and_time,patient_sur_type)
    val_ci_img_ = 0 
    val_ci_rna_ = 0 
    val_ci_cli_ = 0

    if 'img' in args.train_use_type :
        val_ci_img_ = get_val_ci(val_pre_time_img,patient_and_time,patient_sur_type)
    if 'rna' in args.train_use_type :
        val_ci_rna_ = get_val_ci(val_pre_time_rna,patient_and_time,patient_sur_type)
    if 'cli' in args.train_use_type :
        val_ci_cli_ = get_val_ci(val_pre_time_cli,patient_and_time,patient_sur_type)
    return loss.item(), val_ci_, val_ci_img_, val_ci_rna_, val_ci_cli_
    
        
def _neg_partial_log(prediction, T, E):

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(device)

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn 


def _neg_partial_log_(prediction, T, E,DELETE):

   

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(device)
    train_delete = torch.tensor(np.array(DELETE),dtype=torch.float).to(device)

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus*train_delete)

    return loss_nn 



def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def train_a_epoch(model,train_data,all_data,patient_and_time,patient_sur_type,batch_size,optimizer,epoch,format_of_coxloss,delete_list,args):
  

    
    
    img_feature_all = torch.empty((0)).to(device)
    rna_feature_all = torch.empty((0)).to(device)
    cli_feature_all = torch.empty((0)).to(device)

    x_img_feature_all = torch.empty((0)).to(device)
    x_rna_feature_all = torch.empty((0)).to(device)
    x_cli_feature_all = torch.empty((0)).to(device)
    
    model.train() 
    # for name, parms in model.named_parameters():	
    #             logging.info('-->name{}'.format(name))
                # logging.info('-->para{}'.format(parms))
                # logging.info('-->grad_requirs{}'.format(parms.requires_grad))
                # logging.info('-->grad_value{}'.format(parms.grad))
                # logging.info("===") 

        

    train_statue_risk_time = {}
    

    lbl_pred_each = None
    lbl_pred_img_each = None
    lbl_pred_rna_each = None
    lbl_pred_cli_each = None
    batch_feature = None

    pred_risk_graph = torch.empty((0)).to(device)

    list_of_delete = []
    survtime_all = []
    status_all = []
    survtime_img = []
    status_img = []    
    survtime_rna = []
    status_rna = []      
    survtime_cli = []
    status_cli = []  
    
    iter = 0
    loss_nn_all = [] 
    train_pre_time = {}
    train_pre_time_graph = {}
    train_pre_time_img = {}
    train_pre_time_rna = {}
    train_pre_time_cli = {}
    count1 = 0
    all_loss = 0.0 
    mes_loss_of_mae = nn.MSELoss()

    
    mse_loss_of_mae = 0.0
    loss_surv = 0.0
    all_loss_surv = 0.0
    img_loss_surv = 0.0
    rna_loss_surv = 0.0
    cli_loss_surv = 0.0
    
    for i_batch,id in enumerate(train_data):
        
        count = 0
    
        iter += 1 
        
        num_of_model = len(all_data[id].data_type)

        # mask shape   (1, 1, 3)
        # mask value   [[[ True  True False]]] 其内容是随机的
        mask = generate_mask(num=3)
       
        
        if len(args.train_use_type) == 1:
            assert args.format_of_coxloss == 'one' and args.add_mse_loss_of_mae == False
            if args.train_use_type[0] in all_data[id].data_type:
                graph = all_data[id].to(device)

                # 为什么在model里面提供了两个完全一样的参数
                # return (one_x,multi_x),save_fea,(att_2,att_3),fea_dict,最主要的是(one_x,multi_x),one_x = torch.mean(multi_x,dim=0)
                out_pre,out_fea,out_att,fea_dict,x_feature = model(graph,args.train_use_type,args.train_use_type,mask,mix=args.mix) 
                
   
                lbl_pred = out_pre[0]
                
                use_type_eopch = args.train_use_type
                num_of_model = 1
        else:
            if args.train_use_type!=None:
                use_type_eopch = args.train_use_type
                num_of_model = len(use_type_eopch)                
            else:
                use_type_eopch = all_data[id].data_type

            #此处都会被执行
            graph = all_data[id].to(device)
            out_pre,out_fea,out_att,fea_dict,x_feature = model(graph,use_type_eopch,use_type_eopch,mask,mix=args.mix)
            lbl_pred = out_pre[0]
            

            


            img_feature_all = torch.cat((img_feature_all,fea_dict['img'].unsqueeze(0)),0)
            rna_feature_all = torch.cat((rna_feature_all,fea_dict['rna'].unsqueeze(0)),0)
            cli_feature_all = torch.cat((cli_feature_all,fea_dict['cli'].unsqueeze(0)),0)



            x_img_feature_all = torch.cat((x_img_feature_all,fea_dict['mae_labels'][0].unsqueeze(0)),0)
            x_rna_feature_all = torch.cat((x_rna_feature_all,fea_dict['mae_labels'][1].unsqueeze(0)),0)
            x_cli_feature_all = torch.cat((x_cli_feature_all,fea_dict['mae_labels'][2].unsqueeze(0)),0)


            

        if len(args.train_use_type) == 1 and args.train_use_type[0] not in all_data[id].data_type:
            pass
        else:
            train_pre_time[id] = lbl_pred.cpu().detach().numpy()
             
            # 应该是对应文章中的MAE loss
            if args.add_mse_loss_of_mae:
                #  mse_loss_of_mae += args.mse_loss_of_mae_factor * mes_loss_of_mae(input=fea_dict['mae_out'][mask[0]], target=fea_dict['mae_labels'][mask[0]])*delete_list[id]
                mse_loss_of_mae = 0
            

            survtime_all.append(patient_and_time[id])
            
            status_all.append(patient_sur_type[id])
            if iter == 0 or lbl_pred_each == None:
                lbl_pred_each = lbl_pred
            else:
                lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

            if 'img' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
                survtime_img.append(patient_and_time[id])
                status_img.append(patient_sur_type[id]) 
                list_of_delete.append(delete_list[id])           
                if lbl_pred_img_each == None :
                    lbl_pred_img_each = out_pre[1][use_type_eopch.index('img')]
                else:
                    lbl_pred_img_each = torch.cat([lbl_pred_img_each, out_pre[1][use_type_eopch.index('img')]])
            if 'rna' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()
                survtime_rna.append(patient_and_time[id])
                status_rna.append(patient_sur_type[id])            
                if lbl_pred_rna_each == None :
                    lbl_pred_rna_each = out_pre[1][use_type_eopch.index('rna')]
                else:
                    lbl_pred_rna_each = torch.cat([lbl_pred_rna_each, out_pre[1][use_type_eopch.index('rna')]])            
            if 'cli' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()
                survtime_cli.append(patient_and_time[id])
                status_cli.append(patient_sur_type[id])            
                if lbl_pred_cli_each == None :
                    lbl_pred_cli_each = out_pre[1][use_type_eopch.index('cli')]
                else:
                    lbl_pred_cli_each = torch.cat([lbl_pred_cli_each, out_pre[1][use_type_eopch.index('cli')]])
        
        train_statue_risk_time[id] = [patient_sur_type[id],train_pre_time[id][0],patient_and_time[id]]

        if iter % batch_size == 0 or i_batch == len(train_data)-1:

            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)
            img_feature_flag = torch.ones(img_feature_all.shape[0]).to(device)
            rna_feature_flag = torch.ones(rna_feature_all.shape[0]).to(device)
            cli_feature_flag = torch.ones(cli_feature_all.shape[0]).to(device)
            # print(img_feature_all.shape)
            risksss,un_loss,ce_loss = model.graph(img_feature_all,img_feature_flag,rna_feature_all,rna_feature_flag,cli_feature_all,cli_feature_flag,status_all,x_img_feature_all,x_rna_feature_all,x_cli_feature_all)
            
            
            pred_risk_graph = torch.cat([pred_risk_graph,risksss.squeeze()],0)
            for k in range(iter):
                train_pre_time_graph[train_data[k]] = pred_risk_graph.squeeze()[k].cpu().detach().numpy()

            # print('risksss',risksss.squeeze().shape)
            # print('img',lbl_pred_cli_each.shape)

            if np.max(status_all) == 0:
                lbl_pred_each = None
                lbl_pred_img_each = None
                lbl_pred_rna_each = None
                lbl_pred_cli_each = None
                batch_feature = None
                con_loss_label = None
                con_time_label = None
                survtime_all = []
                status_all = []
                list_of_delete = []
                survtime_img = []
                status_img = []    
                survtime_rna = []
                status_rna = []      
                survtime_cli = []
                status_cli = [] 
                iter = 0
                mse_loss_of_mae = 0.0
                loss_surv = 0.0
                all_loss_surv = 0.0
                img_loss_surv = 0.0
                rna_loss_surv = 0.0
                cli_loss_surv = 0.0
                continue

            optimizer.zero_grad() 


            if format_of_coxloss == 'one':
                all_loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
                
                loss_surv = args.all_cox_loss_factor * all_loss_surv
            elif format_of_coxloss == 'multi':
                if lbl_pred_img_each != None:
                    
                    len_lbl_pred_img_each = -len(survtime_img)
                    graph_loss_surv = args.graph_cox_loss_factor * _neg_partial_log_(risksss.squeeze()[len_lbl_pred_img_each:], survtime_img, status_img,list_of_delete)
                    
                    img_loss_surv = args.img_cox_loss_factor * _neg_partial_log_(lbl_pred_img_each, survtime_img, status_img,list_of_delete)
                    
                    # logging.info('img_loss_surv {} '.format(img_loss_surv))
                    loss_surv += img_loss_surv
                    loss_surv += graph_loss_surv
                
                    count = count_same_ones(status_img,list_of_delete)
                    count1 = count1+count
                
                

                if lbl_pred_rna_each != None:


                    # print(lbl_pred_rna_each.shape)

                    rna_loss_surv = args.rna_cox_loss_factor * _neg_partial_log_(lbl_pred_rna_each, survtime_rna, status_rna,list_of_delete)
                    loss_surv += rna_loss_surv
                    # print(lbl_pred_rna_each.shape)

                if lbl_pred_cli_each != None:   
                    cli_loss_surv = args.cli_cox_loss_factor * _neg_partial_log_(lbl_pred_cli_each, survtime_cli, status_cli,list_of_delete)
                    loss_surv += cli_loss_surv 
            else:
                raise("Wrong format_of_coxloss")
            if epoch<args.warmup:
                # loss = loss_surv  + args.un_loss_factor*un_loss +args.ce_loss_factor*ce_loss
                loss = loss_surv  + args.un_loss_factor*un_loss
                
            elif epoch>=args.warmup:
                loss = loss_surv  + args.un_loss_factor*un_loss +args.ce_loss_factor*ce_loss
                # loss = loss_surv  + args.un_loss_factor*un_loss
                
            
           
    
  


            # if args.add_mse_loss_of_mae: 
            #     mse_loss_of_mae/=iter
            #     loss += mse_loss_of_mae 

            all_loss += loss.item()
            loss.backward() 
            if epoch == 0:
                print('*',end='')
            else:  
                optimizer.step()

            torch.cuda.empty_cache()
            lbl_pred_each = None
            lbl_pred_img_each = None
            lbl_pred_rna_each = None
            lbl_pred_cli_each = None
            batch_feature = None
            con_loss_label = None
            con_time_label = None
            survtime_all = []
            status_all = []
            survtime_img = []
            status_img = []    
            survtime_rna = []
            status_rna = []      
            survtime_cli = []
            status_cli = [] 
            list_of_delete = []
            loss_nn_all.append(loss.data.item())
            con_loss = 0.0
            mse_loss = 0.0
            mse_loss_of_mae = 0.0
            kl_loss = 0.0 
            loss_surv = 0.0
            all_loss_surv = 0.0
            img_loss_surv = 0.0
            rna_loss_surv = 0.0
            cli_loss_surv = 0.0
            iter = 0       

    

    total = 0
    for value in train_statue_risk_time.values():
        total+=value[0]
    logging.info('死亡样本个数{}'.format(count1))
    
    
    
    sorted_train_statue_risk_time = sorted(train_statue_risk_time.items(),key=lambda x: x[1][1],reverse=False)

    ranked_train_statue_risk_time = {}
    
    for rank, (id,staue_risk_time) in enumerate(sorted_train_statue_risk_time,start=1):
        ranked_train_statue_risk_time[id] = [staue_risk_time[0],staue_risk_time[1],staue_risk_time[2],rank]


    t_train_ci_img = 0
    t_train_ci_rna = 0
    t_train_ci_cli = 0
    all_loss = all_loss/len(train_data)*batch_size
    t_train_ci = get_val_ci(train_pre_time_graph,patient_and_time,patient_sur_type)
    if len(args.train_use_type) != 1:
        if 'img' in args.train_use_type :
            t_train_ci_img = get_val_ci(train_pre_time_img,patient_and_time,patient_sur_type)
        if 'rna' in args.train_use_type :
            t_train_ci_rna = get_val_ci(train_pre_time_rna,patient_and_time,patient_sur_type)
        if 'cli' in args.train_use_type :
            t_train_ci_cli = get_val_ci(train_pre_time_cli,patient_and_time,patient_sur_type)

    return all_loss,t_train_ci,t_train_ci_img,t_train_ci_rna,t_train_ci_cli,ranked_train_statue_risk_time


def main(args): 
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details


    #该字典是为了存储训练集每个样本confidence的
    dynamic_instace_confience = {}
    
    

    #融合方法？？？
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr
    
    label = "{} lr_={} lr_step_={} epoch_={} warmup_={} contributed_={}  lambda={} percent={} percent-clean={} mix={} graph_factor{} ce_loss_factor{}".format(cancer_type,lr,args.lr_step,args.epochs,args.warmup,args.contribute,args.lambda_,args.percent,args.percentclean,args.mix,args.graph_cox_loss_factor,args.ce_loss_factor) 
    
            # 创建一个Logger对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个处理器将日志消息写入文件
    file_path_saved = 'D:\\python-project\\HGCN\\HGCN_code\\logging-test\\'
    file_handler = logging.FileHandler(label+'.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建一个处理器将消息打印到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志消息的格式
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到Logger对象
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    if args.add_mse_loss_of_mae:
        label = label + " {}*mae_loss".format(args.mse_loss_of_mae_factor)

    if args.img_cox_loss_factor != 1:
        label = label + " img_ft_{}".format(args.img_cox_loss_factor)
    if args.rna_cox_loss_factor != 1:
        label = label + " rna_ft_{}".format(args.rna_cox_loss_factor)    
    if args.cli_cox_loss_factor != 1:
        label = label + " cli_ft_{}".format(args.cli_cox_loss_factor)    
    if args.mix:
        label = label + " mix"
    if args.train_use_type != None:
        label = label + ' use_'
        for x in args.train_use_type:
            label = label + x
        
    
   
    logging.info(label)   
  
    if cancer_type == 'lihc': 
        patients = joblib.load('D:\python-project\HGCN\data\LIHC\lihc_patients.pkl')
        sur_and_time = joblib.load('D:\python-project\HGCN\data\LIHC\lihc_sur_and_time.pkl')
        all_data=joblib.load('D:\python-project\HGCN\data\LIHC\lihc_data.pkl')
        seed_fit_split = joblib.load('D:\python-project\HGCN\data\LIHC\lihc_split.pkl')
    elif cancer_type == 'lusc': 
        patients = joblib.load('D:\python-project\HGCN\data\LUSC\lusc_patients.pkl')
        sur_and_time = joblib.load('D:\python-project\HGCN\data\LUSC\lusc_sur_and_time.pkl')
        all_data=joblib.load('D:\python-project\HGCN\data\LUSC\lusc_data.pkl')             
        seed_fit_split = joblib.load('D:\python-project\HGCN\data\LUSC\lusc_split.pkl')
    elif cancer_type == 'esca': 
        patients = joblib.load('D:\python-project\HGCN\data\ESCA\esca_patients.pkl')
        sur_and_time = joblib.load('D:\python-project\HGCN\data\ESCA\esca_sur_and_time_154.pkl')
        all_data=joblib.load('D:\python-project\HGCN\data\ESCA\esca_data.pkl')             
        seed_fit_split = joblib.load('D:\python-project\HGCN\data\ESCA\esca_split.pkl')
    elif cancer_type == 'luad':
        patients = joblib.load('D:\python-project\HGCN\data\LUAD\luad_patients.pkl')
        sur_and_time = joblib.load('D:\python-project\HGCN\data\LUAD\luad_sur_and_time.pkl')
        all_data=joblib.load('D:\python-project\HGCN\data\LUAD\luad_data.pkl')             
        seed_fit_split = joblib.load('D:\python-project\HGCN\data\LUAD\luad_split.pkl')
    elif cancer_type == 'kirc': 
        patients = joblib.load('D:\python-project\HGCN\data\KIRC\kirc_patients.pkl')
        sur_and_time = joblib.load('D:\python-project\HGCN\data\KIRC\kirc_sur_and_time.pkl')
        all_data=joblib.load('D:\python-project\HGCN\data\KIRC\kirc_data.pkl')             
        seed_fit_split = joblib.load('D:\python-project\HGCN\data\KIRC\kirc_split.pkl')
    elif cancer_type == 'ucec': 
        patients = joblib.load('D:/python-project/HGCN/data/UCEC/ucec_patients.pkl')
        sur_and_time = joblib.load('D:/python-project/HGCN/data/UCEC/ucec_sur_and_time.pkl')
        all_data=joblib.load('D:/python-project/HGCN/data/UCEC/ucec_data.pkl')             
        seed_fit_split = joblib.load('D:/python-project/HGCN/data/UCEC/ucec_split.pkl')

    patient_sur_type, patient_and_time, kf_label = get_patients_information(patients,sur_and_time)


    all_seed_patients = []

    all_fold_test_ci = []
    all_fold_test_ci_59 = []

    test_ci_all = [[0] * args.epochs for _ in range(5)]

    all_fold_test_ci_cli_3 = []


    all_all_ci = []
    all_gnn_time = []
    all_each_model_time = []
    all_fold_each_model_ci = []
    ##

    all_epoch_val_loss = []
    all_epoch_test_loss = []
    
    all_epoch_test_img_ci = []
    all_epoch_test_rna_ci = []
    all_epoch_test_cli_ci = []
    
    all_epoch_train_ci = []
    all_epoch_val_ci = []
    all_epoch_test_ci = []


    repeat = -1
    for seed in range(start_seed,start_seed+repeat_num):
        repeat+=1
        setup_seed(0)
            
        seed_patients = []
        gnn_feature = {}
        one_test_feature = {}
        gnn_time = {}
        each_model_time = {'img':{},'rna':{},'cli':{},'imgrna':{},'imgcli':{},'rnacli':{}}

        val_gnn_time = {}
        test_fold_ci = []
        test_fold_ci_59 = []
        val_fold_ci = []
        test_each_model_ci = {'img':[],'rna':[],'cli':[],'imgrna':[],'imgcli':[],'rnacli':[]}

        train_fold_ci=[]
        fold_att_1 = {}
        fold_att_2 = {}
        
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_test_loss = []
        epoch_train_ci = []
        epoch_val_ci = []
        epoch_test_ci = []
    

        n_fold = 0

        kf = StratifiedKFold(n_splits= 5,shuffle=True,random_state = seed)
        for train_index, test_index in kf.split(patients,kf_label):
            all_patients_ranked = {}
            
            fold_patients = []
            n_fold+=1
            logging.info('fold:{}'.format(n_fold))
            

             
            if fusion_model == 'fusion_model_mae_2':
                model = fusion_model_mae_2(in_feats=1024,
                               n_hidden=args.n_hidden,
                               out_classes=args.out_classes,
                               dropout=drop_out_ratio,
                               train_type_num = len(args.train_use_type)
                                      ).to(device)
               
            optimizer=Adam(model.parameters(),lr=lr,weight_decay=5e-4)

            
            if args.if_fit_split:
                train_data = seed_fit_split[n_fold-1][0]
                val_data = seed_fit_split[n_fold-1][1]
                test_data = seed_fit_split[n_fold-1][2]
            else:
                t_train_data = np.array(patients)[train_index]
                t_l = []
                for x in t_train_data:
                    t_l.append(patient_sur_type[x])
                train_data, val_data ,_ , _ = train_test_split(t_train_data,t_train_data,test_size=0.25,random_state=1,stratify=t_l)         
                test_data = np.array(patients)[test_index]

            logging.info('train_data {} val_data {} test_data{}'.format(len(train_data),len(val_data),len(test_data)))
            fold_patients.append(train_data)
            fold_patients.append(val_data)
            fold_patients.append(test_data)
            seed_patients.append(fold_patients)
   
            
            best_loss = 9999
            best_val_ci = 0
            tmp_train_ci=0



            for epoch in range(epochs):
                
                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=args.lr_step, lr_gamma=args.adjust_lr_ratio)
                
                # model,train_data,all_data,patient_and_time,patient_sur_type,batch_size,optimizer,epoch,format_of_coxloss,args
                if epoch<args.warmup:
                    delete_list = {}
                    for i in range(len(train_data)):
                        delete_list[train_data[i]] = 1
                    all_loss,t_train_ci,t_train_ci_img,t_train_ci_rna,t_train_ci_cli,train_statue_risk_time = train_a_epoch(model,train_data,all_data,patient_and_time,patient_sur_type,batch_size,optimizer,epoch, format_of_coxloss,delete_list, args)
                    train_statue_risk_time_original = copy.deepcopy(train_statue_risk_time)
                    if epoch==0:
                        #将样本的排序加入字典
                        all_patients_ranked = {key:[value[-1]] for key, value in train_statue_risk_time.items()}
                        #获取每个样本的确信度，epoch为0时，确信度设置为0
                        dynamic_instace_confience = {key: 0 for key, value in train_statue_risk_time.items()}
                    else:
                        #将样本的排序加入字典
                        for key in train_statue_risk_time.keys():
                            all_patients_ranked[key].append(train_statue_risk_time[key][-1])

                        #获取当前epoch每个样本的波动
                        new_dict_changed = get_changed_form_dict(all_patients_ranked)
                    

                        #累计样本之前的波动与现在的波动，当作该样本的确信度
                        for key in new_dict_changed.keys():         
                                dynamic_instace_confience[key] = args.lambda_*dynamic_instace_confience[key] + (1-args.lambda_)*new_dict_changed[key]
                    sorted_items = sorted(train_statue_risk_time.items(),key=lambda x:x[1][1])

                else:

                    delete_list = {}
                    for i in range(len(train_data)):
                        delete_list[train_data[i]] = 1
                    precent_pre_epoch = args.percent
                    precent_clean = args.percentclean
                    new_dict,top_keys,last_keys = get_new_dict(train_statue_risk_time,train_statue_risk_time_original,dynamic_instace_confience,precent_pre_epoch,precent_clean)

                    keys_original =  train_statue_risk_time_original.keys()

                    # for key in keys_original:
                    #     print(key,train_statue_risk_time_original[key][0],train_statue_risk_time_original[key][2],new_dict[key][0],new_dict[key][2])

                    #     logging.info('键 {}，原始的生存状态 {}，原始的生存时间 {}，当前的生存状态 {}，当前的生存时间 {}'.format(key,train_statue_risk_time_original[key][0],train_statue_risk_time_original[key][2],new_dict[key][0],new_dict[key][2]))
                        
                    for i in range(len(last_keys)):
                        if train_statue_risk_time_original[last_keys[i]][0] == 0:
                            delete_list[last_keys[i]] = 0
                        else:
                            continue
                        
                    patient_sur_type_new, patient_and_time_new = get_patients_information_new(new_dict)

                    all_loss,t_train_ci,t_train_ci_img,t_train_ci_rna,t_train_ci_cli,train_statue_risk_time = train_a_epoch(model,train_data,all_data,patient_and_time_new,patient_sur_type_new,batch_size,optimizer,epoch, format_of_coxloss,delete_list, args)
                    
                    for key in train_statue_risk_time.keys():
                            all_patients_ranked[key].append(train_statue_risk_time[key][-1])
                    

                    #获取当前epoch每个样本的波动
                    new_dict_changed = get_changed_form_dict(all_patients_ranked)

                    #累计样本之前的波动与现在的波动，当作该样本的确信度
                    for key in new_dict_changed.keys():         
                            dynamic_instace_confience[key] = args.lambda_*dynamic_instace_confience[key] + (1-args.lambda_)*new_dict_changed[key]


                    
                    
                t_test_loss,test_ci,test_img_ci,test_rna_ci,test_cli_ci = prediction(all_data,model,test_data,patient_and_time,patient_sur_type,args)  
                v_loss,val_ci,val_img_ci,val_rna_ci,val_cli_ci = prediction(all_data,model,val_data,patient_and_time,patient_sur_type,args)
              
                
                
                if val_ci >= best_val_ci and epoch>1 or epoch==args.epochs-1 :
                    best_val_ci = val_ci
                    tmp_train_ci = t_train_ci
                    logging.info(val_ci)
                    t_model = copy.deepcopy(model)
                    logging.info('更新了模型')
                if epoch==args.warmup-1:
                    test_fold_ci_59.append(test_ci)
                logging.info("fold:{}, epoch：{:2d}，train_loos：{:.4f},train_ci：{:.4f},val_loos：{:.4f},val_ci：{:.4f},test_loos：{:.4f},test_ci：{:.5f}".format(n_fold,epoch,all_loss,t_train_ci,v_loss,val_ci,t_test_loss,test_ci)) 
                
                test_ci_all[n_fold-1][epoch] = test_ci
    

            t_model.eval() 

            
            t_test_loss,test_ci,_,_,_ = prediction(all_data,model,test_data,patient_and_time,patient_sur_type,args)
            

            test_fold_ci.append(test_ci)
            
            val_fold_ci.append(best_val_ci)
            train_fold_ci.append(tmp_train_ci)

            one_model_res = [{},{},{}]
            two_model_res = [{},{},{}]
            fold_fusion_test_ci = {}
            with torch.no_grad():
                for id in test_data:  
                    data = all_data[id]
                    data.to(device)
                    (one_x,multi_x),fea,(att_1,att_2,att_each),_,x_feature = t_model(data,args.train_use_type,args.train_use_type,mix=args.mix)
                    gnn_time[id] = one_x.cpu().detach().numpy()[0]
                    fold_fusion_test_ci[id] = one_x.cpu().detach().numpy()[0]

                    logging.info('{} {} {}'.format(data.sur_type.cpu().detach().numpy()[0],one_x.cpu().detach().numpy()[0],patient_and_time[id]))
                    # 输出测试集合的ID、生存状态
                    
                    
                    

                    # mask = generate_mask(num=3)
                    # one_test_feature[id] = {}
                    # for i,type_name in enumerate(['img','rna','cli']):
                    #     print('开始单模态啦')
                    #     if type_name in data.data_type:
                    #         (one_,_),one_fea,(_,_),_ = t_model(data,args.train_use_type,use_type=[type_name],in_mask=mask,mix=args.mix)
                    #         one_model_res[i][id] = one_.cpu().detach().numpy()[0]
                    #         each_model_time[type_name][id] = one_.cpu().detach().numpy()[0]

                    # for i,two_type_name in enumerate([['img','rna'],['img','cli'],['rna','cli']]):
                    #     print('开始双模太')
                        
                    #     print(mask)
                    #     (one_,two_),one_fea,(_,_),_ = t_model(data,args.train_use_type,use_type=two_type_name,in_mask=mask,mix=args.mix)
                    #     two_model_res[i][id] = one_.cpu().detach().numpy()[0]
                    #     cat_name = two_type_name[0]+two_type_name[1]
                    #     each_model_time[cat_name][id] = one_.cpu().detach().numpy()[0]     

                    del data        
            # for i,type_name in enumerate(['img','rna','cli']): 
            #     t_ci = get_val_ci(one_model_res[i],patient_and_time,patient_sur_type)
            #     test_each_model_ci[type_name].append(t_ci)
            #     logging.info('{} {} ci: {}'.format(len(one_model_res[i]),type_name,t_ci))

                
            # for i,type_name in enumerate([['img','rna'],['img','cli'],['rna','cli']]): 
            #     t_ci = get_val_ci(two_model_res[i],patient_and_time,patient_sur_type)
            #     cat_name = type_name[0]+type_name[1]
            #     test_each_model_ci[cat_name].append(t_ci)
            #     logging.info('{} {} ci: {}'.format(len(two_model_res[i]),cat_name,t_ci))                
                
            test_ci = get_val_ci(fold_fusion_test_ci,patient_and_time,patient_sur_type)
            logging.info('all ci:{}'.format(test_ci))


            torch.save(t_model.state_dict(), 'esca_graph_120_{}.pth'.format(n_fold))
            del model, train_data, test_data, t_model
            

        logging.info('seed: {}'.format(seed))
        logging.info('test fold ci:')
        for x in test_fold_ci:
            logging.info(x)
          
        logging.info('all ci:')
        logging.info(get_all_ci(gnn_time,patient_and_time,patient_sur_type))
        
        logging.info('val fold ci:')
        for x in val_fold_ci:
            logging.info(x)

    
        all_fold_test_ci.append(test_fold_ci) 
        all_fold_test_ci_59.append(test_fold_ci_59)
        all_fold_each_model_ci.append(test_each_model_ci)
        all_all_ci.append(get_all_ci(gnn_time,patient_and_time,patient_sur_type))
        all_gnn_time.append(gnn_time)
        all_each_model_time.append(each_model_time)

    
    logging.info('summary :')
    logging.info(label)  
    
    logging.info('fusion test fold ci')
    for i,x in enumerate(all_fold_test_ci):       
        logging.info(x)
        logging.info('{} epoch average{}'.format(args.epochs,sum(x)/len(x)))
    
    logging.info('fusion test fold ci of {} epoch'.format(args.warmup))
    for i,x in enumerate(all_fold_test_ci_59):       
        logging.info(x)
        logging.info('{} epoch average{}'.format(args.warmup,sum(x)/len(x)))
        
        
    # for i,type_name in enumerate(['img','rna','cli','imgrna','imgcli','rnacli']): 

    #     logging.info(type_name+' ci:')
    #     for fold_ in all_fold_each_model_ci:
    #         logging.info(fold_[type_name])

    average = [sum(col) / len(col) for col in zip(*test_ci_all)]
    max_value=  max(average)
    max_index = average.index(max_value)
    logging.info('输出各个折epoch的均值{}'.format(average))
    logging.info('最好的epoch{} 结果{}'.format(max_index,max_value))

    # joblib.dump(all_gnn_time,'your path'+sys_time.strftime('%Y-%m-%d-%H-%M')+label+'.pkl')
    # joblib.dump(all_each_model_time,'your path'+sys_time.strftime('%Y-%m-%d-%H-%M')+label+'.pkl')



def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="esca", help="Cancer type")
    parser.add_argument("--img_cox_loss_factor", type=float, default=5, help="img_cox_loss_factor")
    parser.add_argument("--rna_cox_loss_factor", type=float, default=1, help="rna_cox_loss_factor")
    parser.add_argument("--cli_cox_loss_factor", type=float, default=5, help="cli_cox_loss_factor")
    parser.add_argument("--graph_cox_loss_factor", type=float, default=5, help="graph_cox_loss_factor")
    parser.add_argument("--un_loss_factor", type=float, default=1, help="graph_cox_loss_factor")
    parser.add_argument("--ce_loss_factor", type=float, default=20, help="graph_cox_loss_factor")
    
    parser.add_argument("--train_use_type", type=list, default=['img','rna','cli'], help='train_use_type,Please keep the relative order of img, rna, cli')
    parser.add_argument("--format_of_coxloss", type=str, default="multi", help="format_of_coxloss:multi,one")
    parser.add_argument("--add_mse_loss_of_mae", action='store_true', default=True, help="add_mse_loss_of_mae")
    parser.add_argument("--mse_loss_of_mae_factor", type=float, default=0, help="mae_loss_factor")
    parser.add_argument("--start_seed", type=int, default=2, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="fusion_model_mae_2", help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.3, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate of model training")
    parser.add_argument("--lr_step", type=float, default=40, help="")
    parser.add_argument("--epochs", type=int, default=3, help="Cycle times of model training")
    parser.add_argument("--warmup", type=int, default=1, help="warmup times of model training")
    parser.add_argument("--contribute", type=float, default=0.000, help="warmup times of model training")
    parser.add_argument("--top_k", type=int, default=10, help="top k")
    parser.add_argument("--batch_size", type=int, default=512, help="Data volume of model training once")
    parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")    
    parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension") 
    parser.add_argument("--mix", action='store_true', default=False, help="mix mae")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=True, help="fixed division/random division")
    parser.add_argument("--details", type=str, default='', help="Experimental details")
    parser.add_argument("--lambda_", type=float, default=0.5, help="without")
    parser.add_argument("--percent", type=float, default=0.5, help="without")
    parser.add_argument("--percentclean", type=float, default=0.9, help="without")
    
    args, _ = parser.parse_known_args()
    return args



if __name__ == '__main__':
    try:
        # occumpy_mem(cuda_device)
        args=get_params()
        main(args)
    except Exception as exception:
        raise
