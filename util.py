import os
import copy
import torch
import joblib
import random
import sys
import time as sys_time
import numpy as np
from scipy.stats import linregress
from types import MappingProxyType
from lifelines.utils import concordance_index as ci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_changed_form_dict(dict):
    new_dict = {}

    for key,value in dict.items():
        last_two_value = value[-2:]
        last_two_element_abs = abs(last_two_value[0] - last_two_value[1])
        if last_two_element_abs==0:
            last_two_element_abs=1
        new_dict[key] = 1/last_two_element_abs
    return new_dict




def get_new_dict(dict,orignal,dict_confidence,precent_pre_epoch,precent_clean):

    
    keys1 = list(dict.keys())
    # 使用列表推导式输出字典中每个值的第三个元素
    survival_time_elements = [value[2] for value in dict.values()]
    delete_dict = {}
    delete_list = []

    max_value = max(dict_confidence.values())

    dict_confidence_censor = {key:value for key,value in dict_confidence.items() if orignal[key][0]==0}

    dict_confidence_clean = {key:value for key,value in dict_confidence.items() if orignal[key][0]==1}


    sorted_dict = sorted(dict_confidence_censor.items(), key=lambda item: item[1], reverse=True)
    top_percent = int(len(sorted_dict)*precent_pre_epoch)

    sorted_dict_clean = sorted(dict_confidence_clean.items(), key=lambda item: item[1], reverse=True)
    top_percent_clean = int(len(sorted_dict_clean)*precent_clean)

    
    top_keys = [x[0] for x in sorted_dict[:top_percent]]
    # print(top_keys)
    last_keys = [x[0] for x in sorted_dict[top_percent:]]

    top_keys_clean = [x[0] for x in sorted_dict_clean[:top_percent_clean]]
    

    clean = [keys for keys in orignal if orignal[keys][0]==1]
    

    top_keys.extend(top_keys_clean)
    
    print('**********************',len(top_keys))

    filtered_dict = {key: dict[key] for key in top_keys if key in dict}
    filtered_orignal = {key: orignal[key] for key in top_keys if key in orignal}


    dict_new = get_new_stayue_and_time(filtered_dict,filtered_orignal)

    dict.update(dict_new)
    
    last_keys = list(set(delete_list) | set(last_keys))

    return dict,top_keys,last_keys


def custom_sort(arr,mid):

        left_part = [x for x in arr if x < mid]
        right_part = [x for x in arr if x > mid]

        left_part = sorted(left_part,reverse=True)
        right_part = sorted(right_part)

        sorted_arr = left_part+right_part

        return  sorted_arr,left_part,right_part


def greedy_search(dict, start,end,ith,round_list):

    
  
    best_cindex = 0
    best_value = None
    keys = list(dict.keys())
    round_list.append(ith)

    list_index = [x for x in range(ith,len(dict))]




    for possible_value in np.arange(start,end,1.1):
        # 将第6个值替换为可能的值
        dict[keys[ith]][2] = possible_value
        dict[keys[ith]][0] = 1
        #y_true 这里是以预测的风险为标准生成新的时间，所以y_true是预测的风险
        #event  是患者的生存状态，这里也是会变的
        #scores 是患者的得分，在这里是患者的生存时间


        event = [dict[keys[key]][0] for key in list_index]
        y_true = [dict[keys[key]][1] for key in list_index]
        scores =[dict[keys[key]][2] for key in list_index]


        # print('预测的风险作为真实风险',y_true)
        # print('状态',event)
        # print('时间',scores)
        # 计算新数据的C-index
        cindex = get_all_ci_from_list(scores,y_true,event)

        # 更新最佳值和C-index
        if cindex > best_cindex:
            best_cindex = cindex
            best_value = possible_value
    return best_value



def get_all_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in patient_and_time:
        ordered_time.append(patient_and_time[x])
        
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(ci(ordered_time, ordered_pred_time, ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)


def get_new_stayue_and_time(dict,orginal):


    for key in dict.keys():
        dict[key][0] = orginal[key][0]
        dict[key][2] = orginal[key][2]
        

    keys = list(dict.keys())
    for i in range(len(keys) - 1, -1, -1):
        if dict[keys[i]][0] == 1:
            continue
        else:
            distance = np.abs(np.arange(len(keys)) - i)
            distance[i] = 9999
            nearest_indice = np.argsort(distance)[:15]
            round, left_part, right_part = custom_sort(nearest_indice, i)

            round_list = []
            left_part_list = []
            right_part_list = []

            for j in range(len(round)): 
                round_list.append(orginal[keys[j]][2])

            for j in range(len(right_part)): 
                right_part_list.append(orginal[keys[j]][2])

            if len(right_part_list)==0:
                max_round = 0
            else:
                max_round = sum(right_part_list)/len(right_part_list)
            if max_round > dict[keys[i]][2]:
                start = int(dict[keys[i]][2])
                end = int(max_round)
                ith = i
                list_index = [x for x in range(ith, len(dict))]
                event = [dict[keys[key]][0] for key in list_index]

                if sum(event)>=2 and end>start:
                    
                    a = greedy_search(dict, start, end, ith, round)
                    
                    dict[keys[i]][2] = a
                else:
                    continue
            else:
                continue
                
    return dict







def cf(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in patient_and_time:
        ordered_time.append(patient_and_time[x])
        
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(ci(ordered_time, ordered_pred_time, ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)

def get_val_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in pre_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(len(ordered_time), len(ordered_pred_time), len(ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)



def get_all_ci_from_list(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in range(len(pre_time)):
        ordered_time.append(patient_and_time[x])
        
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(ci(ordered_time, ordered_pred_time, ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)



def harrell_c(y_true, scores, event):
    '''
    Compute Harrel C-index given true event/censoring times,
    model output, and event indicators.
    
    Args:
        y_true (array): array of true event times
        scores (array): model risk scores
        event (array): indicator, 1 if event occurred at that index, 0 for censorship
    Returns:
        result (float): C-index metric
    '''
    
    n = len(y_true)
    assert (len(scores) == n and len(event) == n)
    
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    
    result = 0.0
    
    
    
    # use double for loop to go through cases
    for i in range(n):
        # set lower bound on j to avoid double counting
        for j in range(i+1, n):
            
            # check if at most one is censored
            if event[i] == 1 or event[j] == 1:
                # check if neither are censored
                if event[i] == 1 and event[j] == 1:
                    
                    permissible += 1.0
                    
                    # check if scores are tied
                    if scores[i] == scores[j]:
                        ties += 1.0
                    # check for concordant
                    elif y_true[i] < y_true[j] and scores[i] > scores[j]:
                        concordant += 1.0
                    elif y_true[i] > y_true[j] and scores[i] < scores[j]:
                        concordant += 1.0
                
                # check if one is censored
                elif event[i] != event[j] :   
                    # get censored index
                    censored = j
                    uncensored = i
                    
                    if event[i] == 0:
                        censored = i
                        uncensored = j
                        
                    # check if permissible
                    # Note: in this case, we are assuming that censored at a time
                    # means that you did NOT die at that time. That is, if you
                    # live until time 30 and have event = 0, then you lived THROUGH
                    # time 30.
                    if y_true[uncensored] <= y_true[censored]:
                        permissible += 1.0
                        
                        # check if scores are tied
                        if scores[uncensored] == scores[censored]:
                            # update ties 
                            ties += 1.0
                            
                        # check if scores are concordant 
                        if scores[uncensored] > scores[censored]:
                            concordant += 1.0
    
    # set result to c-index computed from number of concordant pairs,
    # number of ties, and number of permissible pairs (REPLACE 0 with your code)  
    result = (concordant + 0.5*ties) / permissible
    
    
    
    return result


def get_patients_information(patients,sur_and_time):
    patient_sur_type = {}
    for x in patients: 
        
        patient_sur_type[x] = sur_and_time[x][0]
        
    time = []
    patient_and_time = {}
    for x in patients:
        time.append(sur_and_time[x][-1])
        patient_and_time[x] = sur_and_time[x][-1]
        
    kf_label = []
    for x in patients:
        kf_label.append(patient_sur_type[x])
    
    return patient_sur_type, patient_and_time, kf_label
    


def get_patients_information_new(train_statue_risk_time_updated):
    patient_sur_type_new = {}
    patient_and_time_new ={}
    for key,value in train_statue_risk_time_updated.items(): 
        patient_sur_type_new[key] = value[0]
        patient_and_time_new[key] = value[2]
    return patient_sur_type_new, patient_and_time_new

def find_closest_keys1(my_dict, A):
    # 选择第一个值为0的键对应的所有第二个值及其键名
    keys_and_values_for_zero = [(key, value[1]) for key, value in my_dict.items() if value[0] == 1]

    # 计算与目标元素的差值
    differences = [(abs(x - A), key) for key, x in keys_and_values_for_zero]

    # 根据差值进行排序
    differences.sort()
    closest_keys = [x[1] for x in differences[:48]]
    return closest_keys

def find_closest_keys(my_dict, A, B,top_k):
    # 选择第一个值为0的键对应的所有第二个值及其键名
    keys_and_values_for_zero = [(key, value[1],value[2]) for key, value in my_dict.items() if value[0] == 1]

    # 计算与目标元素的差值
    differences = [(abs(x - A), key) for key,x,y in keys_and_values_for_zero if y>B]

    # 根据差值进行排序
    differences.sort()
    closest_keys = [x[1] for x in differences[:top_k]]
    return closest_keys

def get_cindex_from_dick_old_statue(original,a):
    # key:[statue,pred_risk,time]
    original_statue = []

    true_time=[]
    true_statue =[]
    pred_risk = []
    for value in a.values():
        if len(value)>=3:
            true_statue.append(value[0])
            pred_risk.append(value[1])
            true_time.append(value[2])

    for value1 in original.values():
        if len(value1)>=3:
            original_statue.append(value1[0])
    
    result_cindex = harrell_c(true_time, pred_risk, true_statue)

    return result_cindex



def get_cindex_from_dick_new_statue(original,a):
    # key:[statue,pred_risk,time]
    original_statue = []

    true_time=[]
    true_statue =[]
    pred_risk = []
    for value in a.values():
        if len(value)>=3:
            true_statue.append(value[0])
            pred_risk.append(value[1])
            true_time.append(value[2])

    for value1 in original.values():
        if len(value1)>=3:
            original_statue.append(value1[0])

    result_cindex = harrell_c(true_time, pred_risk, true_statue)

    return result_cindex

def get_num_of_non_censored(dict):
    total = 0
    for value in dict.values():
        total+=value[0]
    
    return total



def get_new_statue_and_time2(original,a,contributed,top_k):
    # key:[statue,pred_risk,time]

    # old_num = get_num_of_non_censored(original)
    # new_num = get_num_of_non_censored(a)

    # print('yuanshi',old_num)
    # print('xinde',new_num)

    cc = original
    for (key, value),(key2,value2) in zip(original.items(),a.items()):
        
        Referenced_time = []
        if value[0]==0:
          
            closed_keys = find_closest_keys(a,value2[1],value[2],top_k)
            for i in closed_keys:
                if a[i][2]>a[key][2]:
                    Referenced_time.append(a[i][2])

            if len(Referenced_time)==0:
                continue
            else:
            #print('Referenced_time', Referenced_time)

                old_cindex = get_cindex_from_dick_old_statue(original,a)
                

                orignal_a_two = a[key2][2]
                a[key2][2] = sum(Referenced_time)/len(Referenced_time)
                a[key2][0] = 1
                new_cindex = get_cindex_from_dick_new_statue(original,a)
                
                
                if new_cindex-contributed>old_cindex:
                    continue
                else:
                    a[key2][2]=orignal_a_two
                    a[key2][0] = 0
    return cc, a

    
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass    
    
def get_edge_index_full(leng):
    start = []
    end = []
    for i in range(leng):
        for j in range(leng):
            if i!=j:
                start.append(i)
                end.append(j)
    return torch.tensor([start,end],dtype=torch.long).to(device)    
    
def adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    


if __name__ == "__main__":
    
    
        patients = joblib.load('/home/yuehailin/HGCN-YHL/data/UCEC/ucec_patients.pkl')
        sur_and_time = joblib.load('/home/yuehailin/HGCN-YHL/data/UCEC/ucec_patients.pkl')
        all_data=joblib.load('/home/yuehailin/HGCN-YHL/data/UCEC/ucec_data.pkl')             
        seed_fit_split = joblib.load('/home/yuehailin/HGCN-YHL/data/UCEC/ucec_split.pkl')
        patient_sur_type, patient_and_time, kf_label = get_patients_information(patients,sur_and_time)
        print(patient_sur_type)

