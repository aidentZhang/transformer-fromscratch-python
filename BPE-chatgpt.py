import numpy as np
from tqdm import tqdm # timing bar for nice looks
from pathlib import Path
import params


import sys
import time







from collections import Counter
from datasets import load_dataset

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)
word_freq = Counter()
i=0
for example in ds:
    if i > 10: 
        break
    # print(example)
    words = example["text"].split()
    word_freq.update(words)
    i+=1

print("JERE")
vocab = {
    list(word) + ("</w>",): freq
    for word, freq in word_freq.items()
}
print("HI")

from collections import defaultdict

def get_pair_freq(vocab):
    pair_freq = defaultdict(int)

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += freq

    return pair_freq




num_times = params.num_times





with tqdm(total=num_times) as pbar:
    with open('bpe_rules.txt', 'w') as f:
        for i in range(num_times):
            pbar.update(1)
            pair_freq = get_pair_freq(vocab)
            if not pair_freq:
                break


            search_st = time.perf_counter()
            best_pair = max(pair_freq, key=pair_freq.get)

            f.write(f"{best_pair[0].replace('\n', '\\n')}\n{best_pair[1].replace('\n', '\\n')}\n")
            search_et = time.perf_counter()

            print(f"find max took {search_et-search_st:.4f} seconds.")

            new_vocab = {}

            search_st = time.perf_counter()

            for word, freq in vocab.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word)-1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(word[j] + word[j+1])
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_vocab[list(new_word)] = freq

            vocab = new_vocab
            search_et = time.perf_counter()

            print(f"updating took {search_et-search_st:.4f} seconds.")
                

with open('bpe_vocablist.txt', 'w') as f:
    for word, freq in vocab.items():
        f.write(f"{word.replace('\n', '\\n')}\n")


# print(freq_dict[max_occ])
# print(vocab_list)
