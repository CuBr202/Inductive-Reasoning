from gpt_batch_inference import(
    SYSTEM_PROMPT, USER_PROMPT, 
    USER_PROMPT_2, USER_PROMPT_DR, USER_PROMPT_IOE,
    USER_PROMPT_3, USER_PROMPT_ANSWER, USER_PROMPT_SELFCHECK,
    OPENAI_API_KEY, 
    parse_args
    )
from api import chat_api,chat_api_manual, fees
import os
import sys
import re
import json
import random

from lf_loader import read_txt

api_key=OPENAI_API_KEY
if api_key is None:
    api_key = os.getenv('OPENAI_API_KEY')

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
lf = read_txt('./datasets_reasoning/listfunction.txt')
args = parse_args()

def post_process(response: str, ioe: bool=False):
    '''
    Filter reasoning and output in the raw original response.
    '''
    result = {
        'original': response,
        'reasoning':None,
        'output': None
        }
    
    pattern1 = re.compile(r"\[\[Output List\]\](.*?)\[\[END\]\]", re.DOTALL)
    matches = pattern1.findall(response)
    if len(matches) > 0:
        result['output'] = matches[0].strip()
    
    pattern2 = re.compile(r"\[\[Reasoning\]\](.*?)\[\[END\]\]", re.DOTALL)
    matches = pattern2.findall(response)
    if len(matches) > 0:
        result['reasoning'] = matches[0].strip()

    if ioe:
        pass
    return result

def str2list(s: str):
    try:
        lst = re.sub(r'[^0-9,]', '', s)
        lst = lst.split(',')
        if len(lst) == 0 or lst[0] == '':
            return []
        lst = [int(i) for i in lst]
    except:
        print("Small bugs.")
        lst=[]
    return lst

def comp_answer(conf, right_ans, ans1=None, ans2=None, ans3=None):
    if ans1 == ans2:
        if ans1 == right_ans:
            return 1    # right & confident 
        else:
            return 2    # wrong & confident
    else:
        if ans3 == right_ans:
            return 3    # right & unconfident
        else:
            return 4    # wrong & unconfident


score_dict = {"trivial":[0,0], "easy":[0,0],"medium":[0,0],"hard":[0,0],"very_hard":[0,0]}
border = [1, 0.7, 0.5, 0.3, 0.1, 0]
def calc_score(right: bool, human_perf: float):
    '''
    打分函数
    '''
    list_keys = list(score_dict.keys())
    for i in range(5):
        if human_perf<border[i] and human_perf>=border[i+1]:
            score_dict[list_keys[i]][1] += 1
            if right == True:
                score_dict[list_keys[i]][0] += 1
            return
    print("Out of bound!")



log_path = args.log_dir
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
else:
    log = [{"total_cost":0}, score_dict]

correct = 0
cost = 0
total_score = 0

def prompting(i: int, data_point):
    global correct, cost
    input_cases = data_point['input'][0:4]
    output_cases = data_point['output'][0:4]
    test_case = data_point['input'][4]
    right_answer = data_point['output'][4]
    question_id = data_point['id']
    answer, info=chat_api(
            system_content=SYSTEM_PROMPT,
            user_content=USER_PROMPT.format(input_list=input_cases, output_list=output_cases, user_input=test_case),
            api_key=api_key
        )
    

    filt_answer = post_process(answer)
    filt_answer['output'] = str2list(filt_answer['output'])
    right_answer_list = str2list(right_answer)
    print("The right answer is: ", right_answer_list)
    print("Your answer is:",filt_answer['output'])
    
    correctness = False
    if filt_answer['output'] == right_answer_list:
        correct += 1
        print("Correct!")
        correctness = True
    calc_score(correctness, data_point["human_performance"])
    log[1] = score_dict
    print("Accuracy: ", correct/(i+1-args.start))

    message = { "input": SYSTEM_PROMPT+USER_PROMPT.format(input_list=input_cases, output_list=output_cases, user_input=test_case),
                "output": answer,
                "input_tokens": info["input_tokens"],
                "output_tokens": info["output_tokens"],
                "right_answer": right_answer,
                "human_inference": data_point["desc"],
                "human_performance": data_point["human_performance"],
                "model": info['model'],
                "cost": fees['input'][info['model']]*info["input_tokens"]+fees['output'][info['model']]*info["output_tokens"],
                "correctness": correctness,
                "id": question_id
               }
    cost += message['cost']
    log[0]['total_cost']+=message['cost']
    log.append(message)
    if args.log == True:
        with open(log_path,'w') as f:
            f.write(json.dumps(log, indent=4))


def prompting_ioe(i: int, data_point):
    global correct, cost
    #Step 1: 暴露前4个例子
    input_cases = data_point['input'][0:4]
    output_cases = data_point['output'][0:4]
    test_case = data_point['input'][4]
    right_answer = data_point['output'][4]
    question_id = data_point['id']

    money = 0
    info = []
    prompts = [
        {
            'role': 'system', 
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': USER_PROMPT_2.format(input_list=input_cases, output_list=output_cases, user_input=test_case),
        },
    ]
    answer1, info1=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
    prompts.append({"role": "assistant", "content": answer1})
    info.append(info1)
    money += fees['input'][info1['model']]*info1["input_tokens"]+fees['output'][info1['model']]*info1["output_tokens"]
    filt_answer1 = post_process(answer1)
    filt_answer1_list = str2list(filt_answer1['output'])
    print("First answer: ",filt_answer1_list)

    prompts.append({"role": "user", "content": USER_PROMPT_IOE})
    answer2, info2=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
    prompts.append({"role": "assistant", "content": answer2})
    info.append(info2)
    money += fees['input'][info2['model']]*info2["input_tokens"]+fees['output'][info2['model']]*info2["output_tokens"]
    filt_answer2 = post_process(answer2, ioe=True)
    filt_answer2_list = str2list(filt_answer2['output'])
    conf = (filt_answer1_list == filt_answer2_list)
    print("Second answer: ",filt_answer2_list)
    print("Conf: ",conf)


    if conf:
        final_answer_list = filt_answer1_list
        info3 = {}
    elif conf == 'Unconfident' or filt_answer1_list != filt_answer2_list:
        prompts.append({"role": "user", "content": USER_PROMPT_DR})
        answer3, info3=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
        prompts.append({"role": "assistant", "content": answer3})
        info.append(info3)
        money += fees['input'][info3['model']]*info3["input_tokens"]+fees['output'][info3['model']]*info3["output_tokens"]
        filt_answer3 = post_process(answer3)
        final_answer_list = str2list(filt_answer3['output'])
    else:
        print("The model is not following instructions: ",conf)



    right_answer_list = str2list(right_answer)
    ptype = comp_answer(conf, right_answer_list, filt_answer1_list, filt_answer2_list, final_answer_list)
    print("Your answer is: ", final_answer_list)
    print("The right answer is: ", right_answer_list)

    
    correctness = False
    if final_answer_list == right_answer_list:
        correct += 1
        print("Correct!")
        correctness = True
    calc_score(correctness, data_point["human_performance"])
    log[1] = score_dict
    print("Accuracy: ", correct/(i+1-args.start))

    message = { "inputs_and_outputs": prompts,
                "tokens": info,
                "type": ptype,
                "model_answer": str(final_answer_list),
                "right_answer": right_answer,
                "human_inference": data_point["desc"],
                "human_performance": data_point["human_performance"],
                "model": info1['model'],
                "cost": money,
                "correctness": correctness,
                "id": question_id
               }
    cost += message['cost']
    log[0]['total_cost']+=message['cost']
    log.append(message)
    if args.log == True:
        with open(log_path,'w') as f:
            f.write(json.dumps(log, indent=4))


def prompting_sc(i: int, data_point):
    global correct, cost
    #Step 1: 暴露前3个例子，第四个例子先只给问题让他自行推断
    input_cases = data_point['input'][0:3]
    output_cases = data_point['output'][0:3]
    self_test_case = data_point['input'][3]
    self_right_answer = data_point['output'][3]

    input_cases_full = data_point['input'][0:4]
    output_cases_full = data_point['output'][0:4]
    test_case = data_point['input'][4]
    right_answer = data_point['output'][4]
    question_id = data_point['id']

    money = 0
    info = []
    prompts = [
        {
            'role': 'system', 
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': USER_PROMPT_2.format(input_list=input_cases, output_list=output_cases, user_input=self_test_case),
        },
    ]
    answer1, info1=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
    #prompts.append({"role": "assistant", "content": answer1})
    info.append(info1)
    money += fees['input'][info1['model']]*info1["input_tokens"]+fees['output'][info1['model']]*info1["output_tokens"]
    filt_answer1 = post_process(answer1)
    self_test_list = str2list(filt_answer1['output'])
    self_answer_list = str2list(self_right_answer)
    self_reasoning = filt_answer1['reasoning']
    print("First try: ",self_test_list)
    print("The truth: ",self_answer_list)

    # Step 2: SelfCheck, 判断第四个例子推断是否正确
    corr = (self_test_list == self_answer_list)
    reasoning = ''
    if corr:
        print('First correct!')
        reasoning = self_reasoning
    else:
        prompts.append({"role": "user", "content": USER_PROMPT_SELFCHECK
                        .format(ans=self_test_list, corr=self_answer_list)})
        answer2, info2=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
        info.append(info2)
        money += fees['input'][info2['model']]*info2["input_tokens"]+fees['output'][info2['model']]*info2["output_tokens"]
        filt_answer2 = post_process(answer2)
        reasoning = filt_answer2['reasoning']

    #Step 3:给出结果
    last_prompts = [
        {
            'role': 'system', 
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': USER_PROMPT_3.format(input_list=input_cases_full, output_list=output_cases_full),
        },
        {
            'role':'assistant',
            'content': reasoning,
        },
        {
            'role': 'user',
            'content': USER_PROMPT_ANSWER.format(input_list=test_case),           
        }
    ]
    answer3, info3=chat_api_manual(
            msg = last_prompts,
            api_key=api_key
        )
    last_prompts.append({'role':'assistant','content':answer3})
    info.append(info3)
    money += fees['input'][info3['model']]*info3["input_tokens"]+fees['output'][info3['model']]*info3["output_tokens"]
    filt_answer3 = post_process(answer3)
    final_answer_list = str2list(filt_answer3['output'])



    right_answer_list = str2list(right_answer)
    print("Your answer is: ", final_answer_list)
    print("The right answer is: ", right_answer_list)

    
    correctness = False
    if final_answer_list == right_answer_list:
        correct += 1
        print("Correct!")
        correctness = True
    calc_score(correctness, data_point["human_performance"])
    ptype = int(corr)*2 + int(correctness)
    log[1] = score_dict
    print("Accuracy: ", correct/(i+1-args.start))

    message = { "inputs_and_outputs": prompts,
                "last_try": last_prompts,
                "tokens": info,
                "type": ptype,
                "model_answer": str(final_answer_list),
                "right_answer": right_answer,
                "human_inference": data_point["desc"],
                "human_performance": data_point["human_performance"],
                "model": info1['model'],
                "cost": money,
                "correctness": correctness,
                "id": question_id
               }
    cost += message['cost']
    log[0]['total_cost']+=message['cost']
    log.append(message)
    if args.log == True:
        with open(log_path,'w') as f:
            f.write(json.dumps(log, indent=4))


def prompting_pc(i: int, data_point):
    global correct, cost
    #Step 1: 暴露所有例子，先给个答案
    input_cases = data_point['input'][0:4]
    output_cases = data_point['output'][0:4]
    test_case = data_point['input'][4]
    right_answer = data_point['output'][4]
    question_id = data_point['id']

    money = 0
    info = []
    prompts = [
        {
            'role': 'system', 
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': USER_PROMPT_2.format(input_list=input_cases, output_list=output_cases, user_input=test_case),
        },
    ]
    answer1, info1=chat_api_manual(
            msg = prompts,
            api_key=api_key
        )
    #prompts.append({"role": "assistant", "content": answer1})
    info.append(info1)
    money += fees['input'][info1['model']]*info1["input_tokens"]+fees['output'][info1['model']]*info1["output_tokens"]
    filt_answer1 = post_process(answer1)
    self_test_list = str2list(filt_answer1['output'])
    self_reasoning = filt_answer1['reasoning']
    print("First try: ",self_test_list)

    # Step 2: 用0，1，2，4推3的结果
    self_input_cases = data_point['input'][0:3].append(data_point['input'][4])
    self_output_cases = data_point['output'][0:3].append(self_test_list)
    self_test_case = data_point['input'][3]
    self_right_answer = data_point['output'][3]

    last_prompts = [
        {
            'role': 'system', 
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': USER_PROMPT_3.format(input_list=self_input_cases, output_list=self_output_cases),
        },
        {
            'role':'assistant',
            'content': self_reasoning,
        },
        {
            'role': 'user',
            'content': USER_PROMPT_ANSWER.format(input_list=self_test_case),           
        }
    ]
    answer2, info2=chat_api_manual(
            msg = last_prompts,
            api_key=api_key
        )
    last_prompts.append({'role':'assistant','content':answer2})
    info.append(info2)
    money += fees['input'][info2['model']]*info2["input_tokens"]+fees['output'][info2['model']]*info2["output_tokens"]
    filt_answer2 = post_process(answer2)
    answer2_list = str2list(filt_answer2['output'])

    # Step 3: SelfCheck, 判断第四个例子推断是否正确
    corr = (str2list(self_right_answer) == answer2_list)
    reasoning = ''
    if corr:
        print('Guessed correct!')
        reasoning = self_reasoning
    else:
        last_prompts.append({"role": "user", "content": USER_PROMPT_SELFCHECK
                        .format(ans=answer2_list, corr=self_right_answer)})
        answer3, info3=chat_api_manual(
            msg = last_prompts,
            api_key=api_key
        )
        info.append(info3)
        money += fees['input'][info3['model']]*info3["input_tokens"]+fees['output'][info3['model']]*info3["output_tokens"]
        filt_answer3 = post_process(answer3)
        reasoning = filt_answer3['reasoning']

    # Step 4: 得出结果
    if corr:
        final_prompts = []
        final_answer_list = self_test_list
    else:
        final_prompts = [
            {
                'role': 'system', 
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': USER_PROMPT_3.format(input_list=input_cases, output_list=output_cases),
            },
            {
                'role':'assistant',
                'content': reasoning,
            },
            {
                "role": "user", 
                "content": USER_PROMPT_ANSWER.format(input_list=test_case)
            }
        ]
        answer4, info4=chat_api_manual(
            msg = final_prompts,
            api_key=api_key
        )
        final_prompts.append({'role':'assistant','content':answer4})
        info.append(info4)
        money += fees['input'][info4['model']]*info4["input_tokens"]+fees['output'][info4['model']]*info4["output_tokens"]
        filt_answer4 = post_process(answer4)
        final_answer_list = str2list(filt_answer4['output'])




    right_answer_list = str2list(right_answer)
    print("Your answer is: ", final_answer_list)
    print("The right answer is: ", right_answer_list)

    
    correctness = False
    if final_answer_list == right_answer_list:
        correct += 1
        print("Correct!")
        correctness = True
    calc_score(correctness, data_point["human_performance"])
    ptype = int(corr)*2 + int(correctness)
    log[1] = score_dict
    print("Accuracy: ", correct/(i+1-args.start))

    message = { "inputs_and_outputs": prompts,
                "last_try": last_prompts,
                "final_try": final_prompts,
                "tokens": info,
                "type": ptype,
                "model_answer": str(final_answer_list),
                "right_answer": right_answer,
                "human_inference": data_point["desc"],
                "human_performance": data_point["human_performance"],
                "model": info1['model'],
                "cost": money,
                "correctness": correctness,
                "id": question_id
               }
    cost += message['cost']
    log[0]['total_cost']+=message['cost']
    log.append(message)
    if args.log == True:
        with open(log_path,'w') as f:
            f.write(json.dumps(log, indent=4))

def main():
    prompt_dict={
        "raw":prompting,
        "ioe":prompting_ioe,
        "sc":prompting_sc,
        "pc":prompting_pc,
    }
    prompt_type = prompt_dict[args.type]
    with open("./logs/raw123_cross.json", 'r') as f:
        cross_log = json.load(f)
    lst3 = cross_log[1]["3"]

    if args.random == True:
        random.shuffle(lf)

    for i, data_point in enumerate(lf):
        if i<args.start:
            continue
        #if data_point['id'] not in lst3:
        #    continue
        prompt_type(i, data_point)

        if args.batch == 1:
            print("Do you want to continue? (y/n)")
            if input() == 'n':
                break
        else:
            print()
            if args.batch == i+1:
                break



    print(f"The total cost is: {cost} USD")
    print(f"The overall budget used is: {log[0]['total_cost']} USD")

if __name__=='__main__':
    main()

'''
C:/Users/28013/anaconda3/envs/w3/python.exe GPT_scripts/gpt_cli.py --type pc --batch 250 --start 242


把conf去掉，改为通过答案直接判断


'''