import numpy as np
from tqdm import tqdm # timing bar for nice looks
from pathlib import Path
import params


import sys
import time



chunk_list = []


from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")




# directory_path = Path('./Training_Data/raw') 
# files_list = [p for p in directory_path.iterdir() if p.is_file()]

with tqdm(total=len(ds['train'])) as pbar:
    for file in ds['train']:
        pbar.update(1)
        chunk_list += ["<STARTTEXT>"] + file['text'].split(" ")+ ["<ENDTEXT>"]

        print(f"File has size {sys.getsizeof(chunk_list)} bytes")
        # if(sys.getsizeof(chunk_list)>15018942232):
        if(sys.getsizeof(chunk_list)>8942232):
            break








def embed(rule_list, case):
    i = 0
    with tqdm(total=len(rule_list)) as pbar:
        while(i < len(rule_list)):
            pbar.update(1)
            j=0
            max_occ = rule_list[i]
            while(j < len(case)-1):
                if(max_occ == (case[j], case[j+1])):
                    case[j]+=case[j+1]
                    case.pop(j+1)
                    j-=1
                j+=1
            i+=1
    return case

num_times = params.num_times

with open('bpe_rules.txt', 'r') as f:
    rule_list = []
    i = 0
    while(i < num_times):
        try:
            rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
        except:
            print(i)
            break
        i+=1
#     for i in range(len(train)):
#         train[i] = embed(rule_list, train[i])
        # print(len(train[i])) #max length is aroudn 200, howvers around 40-70 usually

svocabDict = {}
vocab_list = []
svocabDict[" "] = 3
svocabDict["END_TOKEN"] = 1
svocabDict["START_TOKEN"] = 0
svocabDict["PAD_TOKEN"] = 2
i = 4



with open('bpe_vocablist.txt', 'r', encoding="utf-8") as f:
    for line in f:
        vocab_list.append(line[:-1].replace('\\n', '\n'))
        svocabDict[vocab_list[-1]] = i
        i+=1



