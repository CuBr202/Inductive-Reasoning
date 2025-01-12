from lf_loader import read_txt
from gpt_cli import post_process, str2list
from tabulate import tabulate
import json


human_correctness_data = [35.496*100/46, 47.298*100/77, 21.630*100/52, 11.607*100/56, 0.771*100/19]
def get_correctness_info(log):
    '''
        获取基本的准确率数据
    '''
    dic = log[1]
    length = len(log)-2
    
    total_correct = 0
    for i,key in enumerate(dic.keys()):
        right = dic[key][0]
        total = dic[key][1]
        perc = 0
        total_correct += right
        if not total == 0:
            perc = right*100/total
        print(' {key:<10}  machine:{right:>2}/{total},{perc:.2f}%      human:{human_right:.2f}%'
              .format(key=key, right=right, perc=perc, human_right=human_correctness_data[i], total=total))
    print(' total correctness: {tc}/{len}, {por:.2f}%\n'.format(tc=total_correct, len=length, por=total_correct/length*100))    
        
def get_correctness_info_lists(log_list: list):
    '''
        获取基本的准确率数据,并求平均
    '''
    right = [0,0,0,0,0]
    total = [0,0,0,0,0]
    perc = [0,0,0,0,0]
    for log in log_list:
        dic = log[1]
        length = len(log)-2
        ll = len(log_list)
        
        total_correct = 0
        for i,key in enumerate(dic.keys()):
            
            right[i] += dic[key][0]/ll
            total[i] += dic[key][1]/ll
            
            total_correct += right[i]
            if not total == 0:
                perc[i] += (right[i]*100/total[i])/3

    for i,key in enumerate(log_list[0][1].keys()):            
        print('    {key:<10}  machine:{right:.2f}/{total:.0f},{perc:.2f}%      human:{human_right:.2f}%'
                .format(key=key, right=right[i], perc=perc[i], human_right=human_correctness_data[i], total=total[i]))
    print('    total correctness: {tc:.2f}/{len}, {por:.2f}%'.format(tc=total_correct, len=length, por=total_correct/length*100))  

def get_wrong(log):
    '''
        计算有多少数组因为顺序不对，重复多，不重复多，少了而被判错
    '''
    cnt_wrong_order = 0
    cnt_more_rep = 0
    cnt_more_nonrep = 0
    cnt_less = 0
    for i,data in enumerate(log):
        if i<2:
            continue
        filt_answer = post_process(data['output'])
        filt_answer_list = str2list(filt_answer['output'])
        right_answer_list = str2list(data['right_answer'])

        #选择长度数据都相同，但是数据顺序不同的答案。
        if(set(filt_answer_list) == set(right_answer_list)
           and len(filt_answer_list) == len(right_answer_list) 
           and not filt_answer_list == right_answer_list):
            cnt_wrong_order += 1

        #选择相对于原list多了一些的数据
        if(set(right_answer_list).issubset(set(filt_answer_list))
           and not len(filt_answer_list) == len(right_answer_list)):
            cnt_more_nonrep += 1

        if(set(filt_answer_list) == set(right_answer_list)
           and not len(filt_answer_list) == len(right_answer_list)):
            cnt_more_rep += 1

        #选择更短的数据
        if(set(filt_answer_list).issubset(set(right_answer_list))
           and not len(filt_answer_list) == len(right_answer_list)):
            cnt_less += 1
    print(' Times of wrong order: ',cnt_wrong_order)
    print(' Times of more but not repetitive: ',cnt_more_nonrep)
    print(' Times of more and repetitive: ',cnt_more_rep)
    print(' Times of less: ',cnt_less)
    print(' Total: ', cnt_less+cnt_more_nonrep+cnt_more_rep+cnt_wrong_order)



"""

log_path = "./logs/log_raw3.json"
with open(log_path,"r") as f:
    print("\n--Report of {lp}--".format(lp=log_path))
    log = json.load(f)
    print(" Basic Correctness Data")
    get_correctness_info(log)
    print(" Special Wrong Cases")
    get_wrong(log)
"""


# part of cross info generation
name = "pc"
log_list = ["./logs/log_{}.json".format(name),"./logs/log_{}2.json".format(name),"./logs/log_{}3.json".format(name)]
cross_log = "./logs/inf123_cross.json"
log1 = json.load(open(log_list[0]))
log2 = json.load(open(log_list[1]))
log3 = json.load(open(log_list[2]))
cross_info = json.load(open(cross_log))

# 1. 123cross
"""
def get_answer(log,i):
    answer = log[i]["model_answer"]
    filt_answer_list = str2list(answer)
    return filt_answer_list

wrong_dict = {"cross_mistake_number":[0,0,0,0],"0":[], "1":[], "2":[], "3":[],"w3":[]}
w = wrong_dict["cross_mistake_number"]
for i in range(2,len(log1)-1):
    idx = int(log1[i]['correctness']==False) + int(log2[i]['correctness']==False) + int(log3[i]['correctness']==False)
    w[idx] += 1
    wrong_dict[str(idx)].append(log1[i]['id'])

    if idx==3 and get_answer(log1,i)==get_answer(log2,i) and get_answer(log1,i)==get_answer(log3,i):
        wrong_dict["w3"].append(log1[i]['id'])
#print(wrong_dict)


cross_info.setdefault(name, wrong_dict)
with open(cross_log,'w') as f:
    f.write(json.dumps(cross_info))

"""

# 2.types
"""
type_log = "./logs/types.json"
type_info = json.load(open(type_log))

type_cluster = []
log_names = [log1, log2, log3]
for log in log_names:
    tdict={"num":[0,0,0,0],"0":[],"1":[],"2":[],"3":[]}
    for i in range(2,len(log)-1):
        correctness = log[i]['correctness']

        mid_answer = post_process(log[i]["inputs_and_outputs"][2]["content"])["output"]
        right_answer = log[i]['right_answer']
        corr = (str2list(mid_answer) == str2list(right_answer))

        current_type = int(corr)*2+int(correctness)
        tdict["num"][current_type] += 1
        tdict[str(current_type)].append(log[i]['id'])
    type_cluster.append(tdict)
    

type_info.setdefault(name, type_cluster)
with open(type_log,'w') as f:
    f.write(json.dumps(type_info))
"""

# 3.corr
log_names = [log1, log2, log3]
print(name)
print('{')
get_correctness_info_lists(log_names)
print('}')



#part of processing cross info
cross_log = "./logs/inf123_cross.json"
cross_info = json.load(open(cross_log))

# 1. w3_iou
"""
def get_iou(list1,list2):
    i = float(len(list(set(list1) & set(list2))))
    u = float(len(list(set(list1) | set(list2))))
    return i/u

def print_table(obj):
    table=[]
    keys = cross_info.keys()    
    table.append([obj])
    table[0].extend(keys)
    for key1 in keys:
        line=[key1]
        for key2 in keys:
            line.append(get_iou(cross_info[key1][obj],cross_info[key2][obj]))
        table.append(line)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


obj_list=["0","1","2","3","r3"]
for obj in obj_list:
    print_table(obj)
    print()
    
"""