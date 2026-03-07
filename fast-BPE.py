import numpy as np
from tqdm import tqdm # timing bar for nice looks
from pathlib import Path
import params
from indexed_priority_queue import IndexedPriorityQueue
import sys
import time
import re





chunk_list = []


from datasets import load_dataset

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)


BYTE_LOOKUP = [f"{i:02x}" for i in range(256)]

# directory_path = Path('./Training_Data/raw') 
# files_list = [p for p in directory_path.iterdir() if p.is_file()]
i = 0
print("loading files ...")
chunk_dict = {}
with tqdm(total=1000000) as pbar:
    for file in ds:
        pbar.update(1)
        if i > 1000000: break

        raw_chunks = re.findall(r"\w+|[^\w\s]|\n", file['text'])
        print(raw_chunks)
        for chunk in raw_chunks:
            byte_data = chunk.encode("utf-8")
            byte_tokens = [BYTE_LOOKUP[b] for b in byte_data]
            byte_tokens.append("</w>")
            chunk = tuple(byte_tokens)


            # print(chunk)


            if chunk not in chunk_dict:
                chunk_dict[chunk] = 1
                chunk_list.append(chunk)
            else:
                chunk_dict[chunk]+=1 
        i+=1


    # chunk_list += ["<STARTTEXT>"] + file['text'].split(" ")+ ["<ENDTEXT>"]

    # print(f"File has size {sys.getsizeof(chunk_list)} bytes")
    # if(sys.getsizeof(chunk_list)>15018942232):
    # if(sys.getsizeof(chunk_list)>8942232):
    #     break
    i+=1

print(f"File has size {sys.getsizeof(chunk_list)} bytes")

       
num_times = params.num_times


global_freq =IndexedPriorityQueue()
local_freq = []
locs = {}
k=0
with tqdm(total=len(chunk_list)) as pbar:
    for k in range(len(chunk_list)):
        pbar.update(1)
        chunk = chunk_list[k]
        local_freq.append([])
        if(len(chunk) != 1):
           for j in range(len(chunk)-1):
                local_freq[-1].append((chunk[j], chunk[j+1]))

                if (chunk[j], chunk[j+1]) in global_freq:
                    locs[(chunk[j], chunk[j+1])].add(k)

                    global_freq.update((chunk[j], chunk[j+1]), global_freq.priority((chunk[j], chunk[j+1]))-chunk_dict[chunk_list[k]])
                else:
                    locs[(chunk[j], chunk[j+1])] = {k}
                    global_freq.push((chunk[j], chunk[j+1]),  -chunk_dict[chunk_list[k]])

# print(chunk_list[:10])
# print(local_freq)

with tqdm(total=num_times) as pbar:
    with open('bpe_rules.txt', 'w', encoding="utf-8") as f:
        with open('bpe_vocablist.txt', 'w', encoding="utf-8") as v:
            for i in range(256):
                v.write(f"{hex(i)[2:]}\n")
            for i in range(num_times):
                pbar.update(1)


                search_st = time.perf_counter()

                try:
                    max_occ, freq = global_freq.pop()
                    # if(i%100==0):
                    #     print(max_occ)
                    #     print(type(max_occ[0]), " ", type(max_occ[1]))
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

                
                f.write("{}\n{}\n".format(
                    max_occ[0].replace('\n', '\\n'),
                    max_occ[1].replace('\n', '\\n')
                ))
                vocabword = max_occ[0].replace('\n', '\\n')+max_occ[1].replace('\n', '\\n')
                v.write(f"{vocabword}\n")
                # print(max_occ)
                # print(global_freq[max_occ])

                update_st = time.perf_counter()
                for index in locs[max_occ]:
    
                    j = 0
                    if(len(local_freq[index])==1):
                        local_freq[index].pop(0)
                    while j < len(local_freq[index]):
                        if(local_freq[index][j] == max_occ):
                            if(j!=0):
                                if local_freq[index][j-1] != max_occ:
                                    global_freq.update(local_freq[index][j-1], global_freq.priority(local_freq[index][j-1]) + chunk_dict[chunk_list[index]])
                                    if global_freq.priority(local_freq[index][j-1]) == 0:
                                        global_freq.delete(local_freq[index][j-1])

                                target = (local_freq[index][j-1][0], max_occ[0]+max_occ[1])
                                if target not in locs:
                                    locs[target] = set()
                                
                                if target not in global_freq:
                                    global_freq.push(target, 0)

                                locs[target].add(index)
                                local_freq[index][j-1] = target

                                global_freq.update(target, global_freq.priority(target) - chunk_dict[chunk_list[index]])


                            if(j!=len(local_freq[index])-1):
                                if local_freq[index][j+1] != max_occ:
                                    global_freq.update(local_freq[index][j+1], global_freq.priority(local_freq[index][j+1]) + chunk_dict[chunk_list[index]])

                                    if global_freq.priority(local_freq[index][j+1]) == 0:
                                        global_freq.delete(local_freq[index][j+1])
                                
                                target = (max_occ[0]+max_occ[1], local_freq[index][j+1][1])
                                if target not in locs:
                                    locs[target] = set()
                                if target not in global_freq:
                                    global_freq.push(target, 0)

                                locs[target].add(index)
                                local_freq[index][j+1] = target
                                global_freq.update(target, global_freq.priority(target) - chunk_dict[chunk_list[index]])
                            local_freq[index].pop(j)
                            j-=1
                        j+=1
                
                locs.pop(max_occ)

                update_et = time.perf_counter()
                # print(f"Updating took {update_et-update_st:.4f} seconds.")


# print(chunk_list_w)
# tok_set = set()
# print(local_freq[1])


# with open('bpe_vocablist.txt', 'w', encoding="utf-8") as f:
#     i = 0
#     while i < len(local_freq):
#         if(chunk_dict[chunk_list[i]]>5):
#             if(len(local_freq[i])==0):
#                 if chunk_list[i] not in tok_set:
#                     escaped = ''.join(chunk_list[i]).replace('\n', '\\n')
#                     f.write(f"{escaped}\n")
#                 tok_set.add(chunk_list[i])
#             else:
#                 tempset = set()
#                 j = 0
#                 while(j < len(local_freq[i])):
#                     if local_freq[i][j][0] not in tok_set:
#                         tok_set.add(local_freq[i][j][0])
#                         escaped = (local_freq[i][j][0]).replace('\n', '\\n')
#                         f.write(f"{escaped}\n")
#                     j+=1
#                 if local_freq[i][-1][-1] not in tok_set:
#                     tok_set.add(local_freq[i][-1][-1])
#                     escaped = local_freq[i][-1][-1].replace('\n', '\\n')
#                     f.write(f"{escaped}\n")

#         i+=1
        

# print(chunk_list[:10])
# print(local_freq[:10])
# print(freq_dict[max_occ])
# print(vocab_list)
