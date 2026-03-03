import numpy as np
from tqdm import tqdm # timing bar for nice looks
from pathlib import Path
import params


import sys
import time






chunk_list = []


from datasets import load_dataset

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)




# directory_path = Path('./Training_Data/raw') 
# files_list = [p for p in directory_path.iterdir() if p.is_file()]
i = 0
for file in ds:
    if i > 100000: break
    chunk_list += ["<STARTTEXT>"] + file['text'].split(" ")+ ["<ENDTEXT>"]

    print(f"File has size {sys.getsizeof(chunk_list)} bytes")
    # if(sys.getsizeof(chunk_list)>15018942232):
    # if(sys.getsizeof(chunk_list)>8942232):
    #     break
    i+=1


chunk_dict = {}

for chunk in chunk_list:
    if chunk not in chunk_dict:
        chunk_dict[chunk] = 1
    else:
        chunk_dict[chunk]+=1 

num_times = params.num_times
i = 0



global_freq = {}
local_freq = []
locs = {}
k=0
while k < len(chunk_list):
    chunk = chunk_list[k]
    local_freq.append([])
    if(len(chunk) != 1):
        j=0
        while j < len(chunk)-1:
            local_freq[-1].append((chunk[j], chunk[j+1]))

            if (chunk[j], chunk[j+1]) in global_freq:
                locs[(chunk[j], chunk[j+1])].add(k)

                global_freq[(chunk[j], chunk[j+1])]+=chunk_dict[chunk_list[k]]

            else:
                locs[(chunk[j], chunk[j+1])] = {k}
                global_freq[(chunk[j], chunk[j+1])] = chunk_dict[chunk_list[k]]
            j+=1
    k+=1


with tqdm(total=num_times) as pbar:
    with open('bpe_rules.txt', 'w', encoding="utf-8") as f:
        while i < num_times:
            pbar.update(1)



            k = 0
            search_st = time.perf_counter()
            try:
                max_occ = max(global_freq, key=global_freq.get)
            except:
                print(str(i-1)+ " is num iterations, terminated due to no more merges being possible")
                break

            # print(global_freq)
            # print(local_freq)
            # print(locs)
            # print("\n")
            # print(max_occ)
            search_et = time.perf_counter()
            # print(f"initializing took {search_et-search_st:.4f} seconds.")


            f.write(f"{max_occ[0].replace('\n', '\\n')}\n{max_occ[1].replace('\n', '\\n')}\n")
            # print(max_occ)
            # print(global_freq[max_occ])
            global_freq.pop(max_occ)
            update_st = time.perf_counter()
            for index in locs[max_occ]:
                j = 0
                if(len(local_freq[index])==1):
                    local_freq[index].pop(0)
                while j < len(local_freq[index]):
                    if(local_freq[index][j] == max_occ):
                        if(j!=0):
                            if local_freq[index][j-1] != max_occ:
                                global_freq[local_freq[index][j-1]] -= chunk_dict[chunk_list[index]]
                                if(global_freq[local_freq[index][j-1]]) == 0:
                                    global_freq.pop(local_freq[index][j-1])

                            target = (local_freq[index][j-1][0], max_occ[0]+max_occ[1])
                            if target not in locs:
                                locs[target] = set()
                            
                            if target not in global_freq:
                                global_freq[target] = 0

                            locs[target].add(index)
                            local_freq[index][j-1] = target
                            global_freq[target] += chunk_dict[chunk_list[index]]


                        if(j!=len(local_freq[index])-1):
                            if local_freq[index][j+1] != max_occ:
                                global_freq[local_freq[index][j+1]] -= chunk_dict[chunk_list[index]]

                                if(global_freq[local_freq[index][j+1]]) == 0:
                                    global_freq.pop(local_freq[index][j+1])
                            
                            target = (max_occ[0]+max_occ[1], local_freq[index][j+1][1])
                            if target not in locs:
                                locs[target] = set()
                            if target not in global_freq:
                                global_freq[target] = 0

                            locs[target].add(index)
                            local_freq[index][j+1] = target
                            global_freq[target] += chunk_dict[chunk_list[index]]
                        local_freq[index].pop(j)
                        j-=1
                    j+=1
            
            locs.pop(max_occ)

            update_et = time.perf_counter()
            # print(f"Updating took {update_et-update_st:.4f} seconds.")
            i+=1


# print(chunk_list_w)
tok_set = set()
vocab_list = []
with open('Training_Data/tokenized/train.txt', 'w', encoding="utf-8") as t:
    with open('bpe_vocablist.txt', 'w', encoding="utf-8") as f:
        i = 0
        while i < len(local_freq):
            if(len(local_freq[i])==0):
                t.write(f"{chunk_list[i].replace('\n', '\\n')}\n \n")
                if chunk_list[i] not in tok_set:
                    f.write(f"{chunk_list[i].replace('\n', '\\n')}\n")
                tok_set.add(chunk_list[i])
            else:
                tempset = set()
                j = 0
                while(j < len(local_freq[i])):
                    if local_freq[i][j][0] not in tok_set:
                        tok_set.add(local_freq[i][j][0])
                        f.write(f"{local_freq[i][j][0].replace('\n', '\\n')}\n")
                    t.write(f"{local_freq[i][j][0].replace('\n', '\\n')}\n")
                    j+=1
                if local_freq[i][-1][-1] not in tok_set:
                    tok_set.add(local_freq[i][-1][-1])
                    f.write(f"{local_freq[i][-1][-1].replace('\n', '\\n')}\n")
                t.write(f"{local_freq[i][-1][-1].replace('\n', '\\n')}\n \n")

            i+=1
        

# print(chunk_list[:10])
# print(local_freq[:10])
# print(freq_dict[max_occ])
# print(vocab_list)
