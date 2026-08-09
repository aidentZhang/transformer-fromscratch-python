import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.transformer import transformer # Ensure this is your optimized version
from app.transformer_large import transformer_inf_large
from app.transformer_large_kv import transformer_inf_large as transformer_inf_large_kv
app = FastAPI()

# --- Load Everything Globally for Warm Starts ---
# This happens when the container starts up
WEIGHTS_S = np.load("app/Weights/weights_s.npz")
NUM_TIMES_S = 5000

with open('app/shakespeare_BPE/bpe_rules.txt', 'r') as f:
    RULE_LIST_S = [(next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')) for _ in range(NUM_TIMES_S)]

SVOCAB_DICT_S = {" ": 3, "END_TOKEN": 1, "START_TOKEN": 0, "PAD_TOKEN": 2}
VOCAB_LIST_S = []
with open('app/shakespeare_BPE/bpe_vocablist.txt', 'r') as f:
    for i, line in enumerate(f, 4):
        v = line[:-1].replace('\\n', '\n')
        VOCAB_LIST_S.append(v)
        SVOCAB_DICT_S[v] = i

# Initialize the model once
MODEL_S = transformer(WEIGHTS_S, RULE_LIST_S, VOCAB_LIST_S, SVOCAB_DICT_S, NUM_TIMES_S)



train_params = {
    "k_DModel": 384,
    "k_ContextLength": 256,
    "k_VocabSize": 25258 + 5,  # plus 5 for special tokens
    "k_Attheads": 4,
    "k_AttBlocks": 4,
    "k_DQuery": 96,
    "num_times": 25000,
    "k_ShiftFactor": 2,
    "k_BatchSize": 32,
    "k_Alpha": 0.0003,
    "k_Beta1": 0.9,
    "k_Beta2": 0.999,
    "k_Epsilon": 0.0002,
    "k_Lambda": 0.01,
    "k_Temp": 0.9
}


#Load Wikipedia model
svocabDict = {}
vocab_list = []
svocabDict[" "] = 3
svocabDict["END_TOKEN"] = 1
svocabDict["START_TOKEN"] = 0
svocabDict["PAD_TOKEN"] = 2
WEIGHTS_W = np.load("app/Weights/weights_w.npz")


i = 4

with open('app/wiki_BPE/bpe_vocablist.txt', 'r', encoding="utf-8") as f:
    for line in f:
        vocab_list.append(line[:-1].replace('\\n', '\n'))
        svocabDict[vocab_list[-1]] = i
        i+=1
loss = 0


# Initialize the model once
MODEL_S = transformer(WEIGHTS_S, RULE_LIST_S, VOCAB_LIST_S, SVOCAB_DICT_S, NUM_TIMES_S)
MODEL_W = transformer_inf_large(WEIGHTS_W, vocab_list, svocabDict, train_params)
MODEL_2_5 = transformer_inf_large_kv(WEIGHTS_W, vocab_list, svocabDict, train_params)

# import time
# import asyncio
async def generate_output_shakespeare(seed: str):
    q = list(seed)
    q = MODEL_S.embed(RULE_LIST_S, q)
    k = len(q)
    while k < MODEL_S.k_ContextLength:
        # Use your forwardprop
        # Note: If you haven't vectorized forwardprop yet, it will be slow!
        E, *_ = MODEL_S.fowardprop(q, SVOCAB_DICT_S)
        prediction = MODEL_S.decode(E, VOCAB_LIST_S)
        
        token = prediction[k]
        yield token
        # await asyncio.sleep(0.1)        
        q.append(token)
        k += 1
        if token == "END": break


    


@app.get("/run_inference_s/{seed}")
async def run_inference(seed: str):
    return StreamingResponse(generate_output_shakespeare(seed), media_type="text/plain")


@app.get("/run_inference_w/{seed}")
async def run_inference(seed: str):
    return StreamingResponse(MODEL_W.run_model(seed), media_type="text/plain")

@app.get("/run_inference_2_5/{seed}")
async def run_inference_2_5(seed: str):
    return StreamingResponse(MODEL_2_5.run_model(seed), media_type="text/plain")