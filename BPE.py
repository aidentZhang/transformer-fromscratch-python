import numpy as np
from tqdm import tqdm # timing bar for nice looks

tok_list = []
vocab_list = []
tok_set = set()
with open('romeo_juliet_proj_gutenburg.txt', 'r') as f:
    tok_list = f.read()
tok_list = list(tok_list)

for character in tok_list:
    if character not in tok_set:
        tok_set.add(character)
        vocab_list.append(character)


num_times = 500
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
            f.write(f"{max_occ[0]}\n{max_occ[1]}\n")

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
with open('bpe_vocablist.txt', 'w') as f:
    for word in tok_list:
        if word not in tok_set:
            tok_set.add(word)
            vocab_list.append(word)
            f.write(f"{word}\n")

# print(tok_list)
# print(freq_dict[max_occ])
# print(vocab_list)
