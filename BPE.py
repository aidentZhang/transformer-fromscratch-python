import numpy as np
from tqdm import tqdm # timing bar for nice looks
from pathlib import Path
import params



# import time

# start = time.time()
# print("hello")
# end = time.time()
# print(end - start)




tok_list = []
vocab_list = []
tok_set = set()
directory_path = Path('./Training_Data/raw') 
files_list = [p for p in directory_path.iterdir() if p.is_file()]
for file in files_list:
    try:
        if(str(file)[-6:-4]!='_f'):
            print(file)
            with open(file, 'r') as f:
                tok_list_temp = f.read()
                tok_list_temp = list(tok_list_temp)
                tok_list+=tok_list_temp
    except:
        print("error opening a file")

for character in tok_list:
    if character not in tok_set:
        tok_set.add(character)
        vocab_list.append(character)


num_times = params.num_times
i = 0
with tqdm(total=num_times) as pbar:
    with open('bpe_rules.txt', 'w') as f:
        while i < num_times:
            pbar.update(1)
            freq_dict = {}
            max_occ = (tok_list[0], tok_list[1])
            freq_dict[max_occ] = 1
            j = 1
            while j < len(tok_list)-1:
                if (tok_list[j], tok_list[j+1]) in freq_dict:

                    freq_dict[(tok_list[j], tok_list[j+1])]+=1
                    if(freq_dict[(tok_list[j], tok_list[j+1])]>freq_dict.get(max_occ)):
                        max_occ = (tok_list[j], tok_list[j+1])
                else:
                    freq_dict[(tok_list[j], tok_list[j+1])] = 1
                j+=1
            f.write(f"{max_occ[0].replace('\n', '\\n')}\n{max_occ[1].replace('\n', '\\n')}\n")
            j = 0
            while(j < len(tok_list)-1):
                if(max_occ == (tok_list[j], tok_list[j+1])):
                    tok_list[j]+=tok_list[j+1]
                    tok_list.pop(j+1)
                    j-=1
                j+=1
            i+=1

tok_set = set()
vocab_list = []
with open('Training_Data/tokenized/train.txt', 'w') as t:
    with open('bpe_vocablist.txt', 'w') as f:
        for word in tok_list:
            t.write(f"{word.replace('\n', '\\n')}\n")
            if word not in tok_set:
                tok_set.add(word)
                vocab_list.append(word)
                f.write(f"{word.replace('\n', '\\n')}\n")

# print(tok_list)
# print(freq_dict[max_occ])
# print(vocab_list)
