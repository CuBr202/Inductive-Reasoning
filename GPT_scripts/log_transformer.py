import json
import re
from lf_loader import read_txt

lst = read_txt()

with open('./logs/log_raw1.json', 'r') as f:
    log = json.load(f)
    for i, datas in enumerate(log):
        if i<2:
            continue
        datas.setdefault("id",lst[i-2]["id"])
        

with open('./logs/log_raw.json','w') as f:
    f.write(json.dumps(log, indent=4))