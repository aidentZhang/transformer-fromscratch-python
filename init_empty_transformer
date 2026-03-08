from transformer_class import transformer
import cupy as cp
from params import train_params
# weights = cp.load("./Weights/weights.npz")




k_DModel = train_params["k_DModel"]
k_ContextLength = train_params["k_ContextLength"]
k_VocabSize = train_params["k_VocabSize"]  # plus 5 for special tokens
k_Attheads = train_params["k_Attheads"]
k_AttBlocks = train_params["k_AttBlocks"]
k_DQuery = train_params["k_DQuery"]
num_times = train_params["num_times"]
k_DKey = k_DQuery
k_ShiftFactor = train_params["k_ShiftFactor"]
k_BatchSize = train_params["k_BatchSize"]
k_Alpha = train_params["k_Alpha"]
k_Beta1 = train_params["k_Beta1"]
k_Beta2 = train_params["k_Beta2"]
k_Epsilon = train_params["k_Epsilon"]
k_Lambda = train_params["k_Lambda"]
k_Temp = train_params["k_Temp"]


sWe = cp.random.normal(loc=0, scale=0.02, size=(k_VocabSize, k_DModel), dtype=cp.float32).astype(cp.float32)
sWe[2] = cp.zeros(k_DModel, dtype=cp.float32)
sWpos = cp.random.normal(loc=0, scale=0.02, size=(k_ContextLength, k_DModel), dtype=cp.float32).astype(cp.float32)

# Scaling operations can happen before or after the cast; leaving them after is fine.
sWq = cp.random.normal(loc=0, scale=cp.sqrt(2/(k_DModel+k_DKey)), size=(k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32).astype(cp.float32) 
sWk = cp.random.normal(loc=0, scale=cp.sqrt(2/(k_DModel+k_DKey)), size=(k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32).astype(cp.float32) 
sWv = cp.random.normal(loc=0, scale=cp.sqrt(1/k_DModel), size=(k_AttBlocks, k_Attheads, k_DModel, k_DModel//k_Attheads), dtype=cp.float32).astype(cp.float32) 
sWo = cp.random.normal(loc=0, scale=cp.sqrt(1/(k_DModel)), size=(k_AttBlocks, k_DModel, k_DModel), dtype=cp.float32) / cp.sqrt(k_Attheads)

sMLPW1 = cp.random.normal(loc=0, scale=cp.sqrt(2/(k_DModel+4*k_DModel)), size=(k_AttBlocks, k_DModel, k_DModel*4), dtype=cp.float32).astype(cp.float32)
sMLPW2 = cp.random.normal(loc=0, scale=cp.sqrt(2/(k_DModel+4*k_DModel)), size=(k_AttBlocks, 4*k_DModel, k_DModel), dtype=cp.float32).astype(cp.float32)

# cp.zeros and cp.ones support float32 natively, so these don't need casting
sMLPb1 = cp.zeros((k_AttBlocks, 1, k_DModel*4), dtype=cp.float32)
sMLPb2 = cp.zeros((k_AttBlocks, 1, k_DModel), dtype=cp.float32)
sLNGain = cp.ones((k_AttBlocks, 2, k_DModel), dtype=cp.float32) 
sLNBias = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)

sLW = cp.random.normal(loc=0, scale=cp.sqrt(2/(k_DModel+k_VocabSize)), size=(k_DModel, k_VocabSize), dtype=cp.float32).astype(cp.float32)
sLB = cp.zeros((1, k_VocabSize), dtype=cp.float32) 


cp.savez("./Weights/weights.npz", sWe=sWe, sWpos=sWpos, sWq=sWq, sWk=sWk, sWv=sWv, sMLPW1=sMLPW1, sMLPW2=sMLPW2, sMLPb1=sMLPb1, sMLPb2=sMLPb2, sLNGain=sLNGain, sLNBias=sLNBias, sLW=sLW, sLB=sLB, sWo = sWo)