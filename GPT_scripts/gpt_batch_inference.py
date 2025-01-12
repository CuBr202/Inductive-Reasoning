import json
import os
import re
from pathlib import Path
import argparse
import hashlib

from api import openai_api, fees

# TODO: Need to change the template for reasoning task
SYSTEM_PROMPT = "You are an agent helping solve inductive reasoning tasks."

USER_PROMPT = """
Here is the reasoning task you need to solve:
We have a black box that takes a list of integers as input and outputs a new list of integers. The input list is as follows:
[[Input Lists]] {input_list}

And the output list is as follows:
[[Output Lists]] {output_list}

Please find out the rule that the black box follows and predict the output list for the following input list:
[[Input List]] {user_input}

And please format your answer as following:
[[Output List]]<Your answer>[[END]]
[[Reasoning]]<Your reasoning>[[END]]
"""

USER_PROMPT_2 = """
Here is the reasoning task you need to solve:
We have a black box that takes a list of integers as input and outputs a new list of integers. The input list is as follows:
[[Input Lists]] {input_list}

And the output list is as follows:
[[Output Lists]] {output_list}

Please think step by step, firstly find out the rule that the black box follows, and finally predict the output list for the following input list:
[[Input List]] {user_input}

And please format your answer as following:
[[Reasoning]]<Your reasoning>[[END]]
[[Output List]]<Your answer>[[END]]
"""

USER_PROMPT_IOE = """
Please review your previous answer. 
If you're very confident about your answer, please maintain your answer. Otherwise, please update your answer.

And please format your answer as following:
[[Reasoning]]<Your reasoning>[[END]]
[[Output List]]<Your answer>[[END]]
"""

USER_PROMPT_DR = """
You give two different answers in previous responses. 
Please carefully check the problem and your two answers again, and give the best answer.

And please format your answer as following:
[[Reasoning]]<Your reasoning>[[END]]
[[Output List]]<Your answer>[[END]]
"""

USER_PROMPT_3 = """
Here is the reasoning task you need to solve:
We have a black box that takes a list of integers as input and outputs a new list of integers. The input list is as follows:
[[Input Lists]] {input_list}

And the output list is as follows:
[[Output Lists]] {output_list}

Please think step by step, and find out the rule that the black box follows.
"""

USER_PROMPT_SELFCHECK = """
Your answer {ans} is NOT correct. The correct answer should be {corr}.
Please rethink your reasoning and give a different one that's corresponding to the previous 4 input-output pairs.
You should think step by step.
And please format your reasoning as following:
[[Reasoning]]<Your reasoning>[[END]]
"""

USER_PROMPT_ANSWER = """
Given an input list as follows: 
[[Input list]]{input_list}

Based on your reasoning, what's your answer?
Please format your answer as following:
[[Output List]]<Your answer>[[END]]
"""




OPENAI_API_KEY='<key>'
log_path = './logs/log.json'
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
else:
    log = [{"total_cost":0}]

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration of test types.')
    parser.add_argument(
        '--log',
        type=bool,
        default=True,
        help='Whether to log or not',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/log.json',
        help='Directory where the log will be saved.',
    )
    parser.add_argument(
        '--random',
        type=bool,
        default=False,
        help='Whether to randomize the input or not.',
    )
    parser.add_argument(
        '--type',
        type=str,
        default="raw",
        help='Type of prompting.',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Test a batch of examples.',
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='The start point, including the input number',
    )
    parser.add_argument(
        '--end',
        type=int,
        default=249,
        help='The end point, including the input number',
    )
    
    return parser.parse_args()

# TODO: Need to change the post_process function for reasoning task
def post_process(response: str):
    result = {'output': response}

    '''
    pattern = re.compile(r"\[\[Refined Prompt\]\](.*?)\[\[END\]\]", re.DOTALL)
    matches = pattern.findall(response)
    if len(matches) > 0:
        result['output'] = matches[0].strip()'''
    return result


def cache_checker(result: dict) -> bool:
    return result['output'] is not None

def main():
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"
    api_key = OPENAI_API_KEY
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    else:
        os.environ['OPENAI_API_KEY'] = api_key
    args = parse_args()
    filename = os.path.basename(args.filepath)

    with open(args.filepath, 'r', encoding='utf-8') as f:
        origin = json.load(f)

    data = [
        {
            'uuid': hashlib.md5(line['input'].encode()).hexdigest(),
            'input': line['input'],
            'output': None,
        }
        for line in origin
    ][:10]
    empty_index = list(range(len(data)))

    max_repeat = 5
    while len(empty_index) > 0 and max_repeat > 0:
        print(f'Remaining repeat times: {max_repeat}')
        contents = [
            (
                SYSTEM_PROMPT,
                USER_PROMPT.format(user_input=line['input']),
            )
            for line in [data[i] for i in empty_index]
        ]

        results, info = openai_api(
            contents,
            action='chat',
            post_process=post_process,
            cache_checker=cache_checker,
            num_workers=10,
            cache_dir=args.cache_dir,
            print_interval=10,
        )

        for index, result in zip(empty_index, results):
            data[index]['output'] = result['output']
        
        for i in range(len(info)):
            message = {
                'input': contents[i][0] + contents[i][1],
                'output': results[i]['output'],
                'input_tokens': info[i]['input_tokens'],
                'output_tokens': info[i]['output_tokens'],
                'model': info[i]['model'],
                'cost': fees['input'][info['model']]*info["input_tokens"]+fees['output'][info['model']]*info["output_tokens"]
            }
            log[0]['total_cost'] += message['cost']
            log.append(message)
            with open('log.json', 'w') as f:
                f.write(json.dumps(log, indent=4))

        empty_index = [i for i in empty_index if data[i]['output'] is None]
        print(f'Number of inference failed: {len(empty_index)}')
        max_repeat -= 1

    save_path = os.path.join(args.save_dir, filename)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Save to {save_path}')
    print(f"The overall budget used is: {log[0]['total_cost']} USD")
    


if __name__ == '__main__':
    main()
