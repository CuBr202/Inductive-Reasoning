# LLM-reasoning

This repository is created for handin project.
The previous work is all done locally.

## dateset_reasoning
This the List Function dataset by Rule et al.(2020)


## GPT-scripts
Here is the code of this project.  
api.py:  Handle OpenAI API.  
gpt_batch_inference.py:  Handle prompts and OpenAI API key.  
gpt_cli.py: Core of the experiments. Prompt to the LLM and handle the LLM's output.
lf_loader.py: Read and load the dataset.  
log_transformer.py and read_log.py: Used for reading the logs. Donot run this file.

If you want to rerun the experiments, please fill your api-key in gpt_batch_inference.py and execute gpt_cli.py. 


## logs
Record logs of multiple rounds of experiments.  
The logs are firstly printed into log.json, and we duplicate it and give it another name.  
Run clear_log.py to clear all the contents in log.json.
