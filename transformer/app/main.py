import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.transformer import transformer
from fastapi.responses import StreamingResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    
    weights = np.load("app/Weights/weights.npz")

    num_times = 5000


    with open('app/bpe_rules.txt', 'r') as f:
        rule_list = []
        i = 0
        while(i < num_times):
            rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
            i+=1


    svocabDict = {}
    vocab_list = []
    svocabDict[" "] = 3
    svocabDict["END_TOKEN"] = 1
    svocabDict["START_TOKEN"] = 0
    svocabDict["PAD_TOKEN"] = 2
    i = 4



    with open('app/bpe_vocablist.txt', 'r') as f:
        for line in f:
            vocab_list.append(line[:-1].replace('\\n', '\n'))
            svocabDict[vocab_list[-1]] = i
            i+=1


    shakespeare_t = transformer(weights, rule_list, vocab_list, svocabDict, num_times)
    app.state.rule_list=rule_list
    app.state.vocab_list=vocab_list
    app.state.shakespeare_t=shakespeare_t
    
    yield   # <-- app runs here


    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"transformer"}

def generate_output(trans, seed):
    q = list(seed)
    q = trans.embed(app.state.rule_list, q)
    k = len(q)
    print(q)
    while k < trans.k_ContextLength:
        E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = trans.fowardprop(q, trans.svocabDict)
        prediction = trans.decode(E, app.state.vocab_list)
        # loss, onehot_cache = findLoss(E, q, svocabDict)
        # print(loss)
        # print(prediction)
        q.append(prediction[k])
        yield(prediction[k])
        k+=1


@app.get("/run_inference/{seed}")
def run_inference(seed:str):
    return StreamingResponse(generate_output(app.state.shakespeare_t, seed), media_type="text/plain")
