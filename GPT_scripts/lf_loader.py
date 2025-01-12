import copy


from torch.utils.data import Dataset, DataLoader

'''
./datasets/listfunction.txt
'''



def read_txt(path = './datasets_reasoning/listfunction.txt'):
    data_list = []
    data_dict = {"human_performance":float,"desc_length":int,"id":str,
                "desc":str,"lambda":str,
                "input":[],"output":[]}
    with open(path, mode="r", encoding="utf-8") as r:
        for idx, line in enumerate(r):
            if idx%7 == 0:
                spline = line.split(' ')
                data_dict["human_performance"] = float(spline[0])
                data_dict["desc_length"] = int(spline[1])
                data_dict["id"] = spline[2]

                del spline[0:3]
                data_dict["desc"] = ' '.join(spline).strip('\n')

            elif idx%7 == 1:
                data_dict["lambda"] = line.strip('\n')

            else:
                spline = line.strip('\n').split(' → ')
                data_dict["input"].append(spline[0])
                data_dict["output"].append(spline[1])
                if idx%7 == 6:
                    data_dict_dc = copy.deepcopy(data_dict)
                    data_list.append(data_dict_dc)
                    data_dict["input"] = []
                    data_dict["output"] = []

    return data_list

