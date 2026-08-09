import numpy as np
from transformer_large_kv import transformer_inf_large



train_params = {
    "k_DModel": 384,
    "k_ContextLength": 256,
    "k_VocabSize": 25258 + 5,  # plus 5 for special tokens
    "k_Attheads": 4,
    "k_AttBlocks": 4,
    "k_DQuery": 96,
    "num_times": 25000,
    "k_ShiftFactor": 2,
    "k_BatchSize": 1, #changed a bit
    "k_Alpha": 0.0003,
    "k_Beta1": 0.9,
    "k_Beta2": 0.999,
    "k_Epsilon": 0.0002,
    "k_Lambda": 0.01,
    "k_Temp": 1
}


#Load Wikipedia model
svocabDict = {}
vocab_list = []
svocabDict[" "] = 3
svocabDict["<END>"] = 1
svocabDict["<STA>"] = 0
svocabDict["<PAD>"] = 2
WEIGHTS_W = np.load("/Users/aidenzhang/Documents/machine_learning/transformer-fromscratch-python/transformer/app/Weights/weights_w.npz")


i = 4

with open('/Users/aidenzhang/Documents/machine_learning/transformer-fromscratch-python/transformer/app/wiki_BPE/bpe_vocablist.txt', 'r', encoding="utf-8") as f:
    for line in f:
        vocab_list.append(line[:-1].replace('\\n', '\n'))
        svocabDict[vocab_list[-1]] = i
        i+=1
loss = 0


# Initialize the model once
MODEL_W = transformer_inf_large(WEIGHTS_W, vocab_list, svocabDict, train_params)

for word in MODEL_W.run_model("World war 2"):
    print(word, flush = True, end = "")

# @app.get("/run_inference_s/{seed}")
# async def run_inference(seed: str):
#     return StreamingResponse(generate_output_shakespeare(seed), media_type="text/plain")


# @app.get("/run_inference_w/{seed}")
# async def run_inference(seed: str):
#     return StreamingResponse(MODEL_W.run_model(seed), media_type="text/plain")