from transformer_class import transformer
import cupy as cp

weights = cp.load("./Weights/weights.npz")

from params import train_params
num_times=train_params["num_times"]

with open('bpe_rules.txt', 'r', encoding="utf-8") as f:
    rule_list = []
    i = 0
    while(i < num_times):
        try:
            rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
        except:
            print(i)
            break
        i+=1

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
loss = 0

model = transformer(weights, rule_list, vocab_list, svocabDict, train_params)
model.train(60000, 120000, 30000)
model.run_model()
