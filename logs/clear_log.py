import json


log_path = './logs/log.json'
with open(log_path,'r') as f:
    log=json.load(f)

log=log[0:2]


with open(log_path,'w') as f:
        f.write(json.dumps(log, indent=4))