import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.transformer import transformer # Ensure this is your optimized version

app = FastAPI()

# --- Load Everything Globally for Warm Starts ---
# This happens when the container starts up
WEIGHTS = np.load("app/Weights/weights.npz")
NUM_TIMES = 5000

with open('app/bpe_rules.txt', 'r') as f:
    RULE_LIST = [(next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')) for _ in range(NUM_TIMES)]

SVOCAB_DICT = {" ": 3, "END_TOKEN": 1, "START_TOKEN": 0, "PAD_TOKEN": 2}
VOCAB_LIST = []
with open('app/bpe_vocablist.txt', 'r') as f:
    for i, line in enumerate(f, 4):
        v = line[:-1].replace('\\n', '\n')
        VOCAB_LIST.append(v)
        SVOCAB_DICT[v] = i

# Initialize the model once
MODEL = transformer(WEIGHTS, RULE_LIST, VOCAB_LIST, SVOCAB_DICT, NUM_TIMES)
# import time
# import asyncio
async def generate_output(seed: str):
    q = list(seed)
    q = MODEL.embed(RULE_LIST, q)
    k = len(q)
    while k < MODEL.k_ContextLength:
        # Use your forwardprop
        # Note: If you haven't vectorized forwardprop yet, it will be slow!
        E, *_ = MODEL.fowardprop(q, SVOCAB_DICT)
        prediction = MODEL.decode(E, VOCAB_LIST)
        
        token = prediction[k]
        yield token
        # await asyncio.sleep(0.1)        
        q.append(token)
        k += 1
        if token == "END": break

@app.get("/run_inference/{seed}")
async def run_inference(seed: str):
    return StreamingResponse(generate_output(seed), media_type="text/plain")