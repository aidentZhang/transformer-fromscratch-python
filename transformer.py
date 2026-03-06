import numpy as np
from tqdm import tqdm # timing bar for nice looks
import params
import cupy as cp
from pathlib import Path
import time
import itertools

#------------------
#PARAM DECLARATIONS
k_DModel = params.k_DModel #32
k_ContextLength = params.k_ContextLength#8
k_VocabSize = params.k_VocabSize #plus four for start, end, pad, and space      last one is no idea
k_Attheads = params.k_Attheads
k_AttBlocks = params.k_AttBlocks
#these should all be the same
k_DQuery = params.k_DQuery
k_DKey = k_DQuery
#------------------
#SETUP DATA STRUCTS

# Generate as float32, then cast to float32
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

sSoftmaxMask = cp.nan_to_num(-cp.inf * cp.triu(cp.ones((k_ContextLength, k_ContextLength), dtype=cp.float32), k=1), nan=0.0)
#E has dimension k_ContextLength x k_DModel
#testing
# sWq[0][0] = [[1,0],[0,1]]
# sWk[0][0] = [[1,0],[0,1]]
# sWv[0][0] = [[1,0],[0,1]]
# sMLPW1 = [[[1,0],[0,1]]]
# sMLPW2 = [[[1, 0], [0,1]]]
# sMLPb1 = cp.zeros((k_AttBlocks, 1, 2))
# sMLPb2 = cp.zeros((k_AttBlocks, 1, 2))
# sLNGain = cp.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
# sLNBias = cp.zeros((k_AttBlocks, 2, k_DModel))
# sLW = cp.random.normal(0, cp.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, 1))
# sLB = cp.ones((k_VocabSize)) #ADDED TO ALL TOKENS
# E = [[1, 2]]   
#------------------
#TRANSFORMER FUNCTIONS

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)


def layerNorm(E, attLayer, prePostMLP):
    temp = (E-cp.mean(E, axis = -1, keepdims = True))/cp.sqrt((cp.nan_to_num(cp.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001))
    return sLNBias[attLayer, prePostMLP] + temp * (sLNGain[attLayer, prePostMLP]), temp


def softmax(E):
    exp_ = cp.exp(E-cp.max(E, axis=-1, keepdims=True))
    return cp.nan_to_num(exp_/(cp.sum(exp_, axis=-1, keepdims=True)))


def relu(E):
    return cp.maximum(0, E)

def relu_deriv(E):
    return (E > 0).astype(cp.float32)

def decode(E, svocabList):
    temp = cp.argmax(E, axis = -1)
    answer = []
    for i in temp:
        i = int(i)
        if(i > 3):
            if i-4 >= len(svocabList):
                answer.append("NA")
            else:
                answer.append(svocabList[i-4])

        elif (i==3):
            answer.append(" ")
        elif (i==1):
            answer.append("END")
        elif (i==2):
            answer.append("PAD")
        else:
            answer.append("STA")
    return answer
#------------------------------------------------------------------------------------------------------------------------------FINISH
def findLoss(E, input_llm, svocabDict):
    loss = []
    onehot_cache = cp.zeros((k_BatchSize, k_ContextLength, k_VocabSize))
    for j in range(len(input_llm)):
        temp_loss = 0
        i=0
        while i < len(input_llm[j]):
            if input_llm[j][i] in svocabDict:
                onehot_cache[j, i, svocabDict[input_llm[j][i]]] = 1
                temp_loss += -cp.log(E[j][i][svocabDict[input_llm[j][i]]]+0.00001)
            else:
                onehot_cache[j, i, k_VocabSize-1] = 1
                temp_loss += -cp.log(E[j][i][k_VocabSize-1]+0.00001)
            i+=1
        temp_loss = (temp_loss-cp.log(E[j][len(input_llm[j])][1]+0.001))/len(input_llm[j])
        onehot_cache[j, len(input_llm[j]), 1] = 1
        loss.append(temp_loss)
    return loss, onehot_cache
#------------------
#FORWARD PROPAGATE
def fowardprop(input_llm, svocabDict):
    #------------------
    #CACHEING
    E_preln_cache = cp.zeros((k_AttBlocks, 2, k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)
    E_midln_cache = cp.zeros((k_AttBlocks, 2, k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)
    E_postln_cache = cp.zeros((k_AttBlocks, 2, k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)
    E_soft_cache = cp.zeros((k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_ContextLength), dtype=cp.float32)
    E_lin_cache = cp.zeros((k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)
    E_relu_cache = cp.zeros((k_AttBlocks, k_BatchSize, k_ContextLength, k_DModel*4), dtype=cp.float32)
    E_conc_cache= cp.zeros((k_AttBlocks, k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)
    Q_cache = cp.zeros((k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_DKey))
    K_cache = cp.zeros((k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_DKey))
    V_cache = cp.zeros((k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_DModel//k_Attheads))
    We_to_E_cache = cp.zeros((k_BatchSize, k_ContextLength, k_VocabSize), dtype=cp.float32)    
    
    E = cp.zeros((k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)

    padMask = cp.zeros((k_BatchSize,  k_ContextLength, k_ContextLength))
    for i in range(k_BatchSize):
        padMask[i, :, len(input_llm[i])+1:k_ContextLength] = -cp.inf

        temp = cp.zeros(k_VocabSize)
        temp[2] = 1
        We_to_E = cp.zeros((k_ContextLength, k_VocabSize))
        We_to_E[0, 0] = 1
        o = 1
        for j in input_llm[i]:
            if j in svocabDict:
                We_to_E[o, [svocabDict[j]]] = 1
            else:
                We_to_E[o, k_VocabSize-1] = 1
            o+=1

        while(o<k_ContextLength):
            We_to_E[o, 2] = 1
            o+=1

        E[i] = We_to_E@sWe
        We_to_E_cache[i]=We_to_E
        for j in range(len(input_llm[i])+1):
            E[i][j]+=sWpos[j]


    currAttBlock = 0
    while(currAttBlock < k_AttBlocks):
        E_preln_cache[currAttBlock, 0] = cp.array(E)
        E_ln, E_midln_cache[currAttBlock, 0] = layerNorm(E, currAttBlock, 0)
        E_postln_cache[currAttBlock, 0] = cp.array(E_ln)

        Q=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWq[currAttBlock], [1, 0, 2]), [k_DModel, k_DKey*k_Attheads]), [k_BatchSize, k_ContextLength, k_Attheads, k_DKey]), [0, 2, 1, 3])
        K=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWk[currAttBlock], [1, 0, 2]), [k_DModel, k_DKey*k_Attheads]), [k_BatchSize, k_ContextLength, k_Attheads, k_DKey]), [0, 2, 1, 3])
        V=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWv[currAttBlock], [1, 0, 2]), [k_DModel, k_DModel]), [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3])
        Q_cache[currAttBlock]=Q
        K_cache[currAttBlock]=K
        V_cache[currAttBlock]=V
        E_soft_cache[currAttBlock] = softmax(1/cp.sqrt(k_DKey) * Q@cp.transpose(K, [0, 1, 3, 2])+sSoftmaxMask+cp.expand_dims(padMask, axis=1))
        # print(cp.shape(cp.reshape(cp.transpose(E_soft_cache[currAttBlock]@(V), [0, 2, 1, 3]), [k_BatchSize, k_ContextLength, k_DModel])))
        E_conc_cache[currAttBlock] = cp.reshape(cp.transpose(E_soft_cache[currAttBlock]@(V), [0, 2, 1, 3]), [k_BatchSize, k_ContextLength, k_DModel])
        E+= E_conc_cache[currAttBlock]@sWo[currAttBlock]

        E_preln_cache[currAttBlock, 1] = cp.array(E)
        E_ln, E_midln_cache[currAttBlock, 1] = layerNorm(E, currAttBlock, 1)
        E_postln_cache[currAttBlock, 1] = cp.array(E_ln)
        E_relu_cache[currAttBlock] = relu(E_ln@sMLPW1[currAttBlock]+sMLPb1[currAttBlock])
        E += E_relu_cache[currAttBlock]@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
        currAttBlock+=1
    
    E_lin_cache = cp.array(E)
    E=E@sLW+sLB
    E=softmax(E)

    return E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E_cache, E_conc_cache, Q_cache, K_cache, V_cache
#------------------
#BACKPROP
def backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache):
    global g_We 
    global g_Wpos 
    global g_Wq 
    global g_Wk
    global g_Wv
    global g_MLPW1
    global g_MLPW2
    global g_MLPb1
    global g_MLPb2
    global g_LNGain
    global g_LNBias
    global g_LW

    global g_LB


    g_LW+=cp.sum(cp.transpose(E_lin_cache, [0, 2, 1])@(E-onehot_cache), axis=0)/ k_ContextLength
    g_LB+=cp.sum(cp.sum((E-onehot_cache), axis=1), axis=0)/ k_ContextLength
    
    G = (E-onehot_cache)@sLW.T/k_ContextLength
    currAttBlock = k_AttBlocks-1
    currAttBlock = k_AttBlocks-1

    while(currAttBlock>=0):
        G_preln = cp.array(G)
        g_MLPb2[currAttBlock]+=cp.sum(cp.sum(G, axis=1), axis=0)
        g_MLPW2[currAttBlock]+=cp.sum(cp.transpose(E_relu_cache[currAttBlock], [0, 2, 1])@G, axis=0)


        g_MLPb1[currAttBlock]+=cp.sum(((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])), axis=[0, 1])
        g_MLPW1[currAttBlock]+=cp.sum((cp.transpose(E_postln_cache[currAttBlock, 1], [0, 2, 1])) @ ((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])), axis=0)

        G = (G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])@sMLPW1[currAttBlock].T

        g_LNBias[currAttBlock, 1] += cp.sum(cp.sum(G, axis = 1), axis=0)
        g_LNGain[currAttBlock, 1] += cp.sum(cp.sum((E_midln_cache[currAttBlock, 1])*G, axis = 1), axis=0)
        
        
        Xhat_mean = E_midln_cache[currAttBlock, 1]*(cp.mean((G*sLNGain[currAttBlock, 1])*E_midln_cache[currAttBlock, 1], axis = 2, keepdims=True))
        G= G_preln+(1/cp.sqrt(cp.var(E_preln_cache[currAttBlock, 1], axis = 2, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 1]-cp.mean(G*sLNGain[currAttBlock, 1], axis = 2, keepdims=True)-Xhat_mean)


        G_preln = cp.array(G)


        

        g_Wo[currAttBlock]+=cp.sum(cp.transpose(E_conc_cache[currAttBlock], [0, 2, 1])@G, axis=0)
        # print(cp.shape(G@sWo[currAttBlock].T))
        # print(cp.shape(cp.transpose(E_postln_cache[currAttBlock, 0], [0, 2, 1])[:, None, :, :]@cp.transpose(E_soft_cache[currAttBlock], [0, 1, 3, 2])@cp.transpose(cp.reshape(G@sWo[currAttBlock].T, [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3])))

        g_Wv[currAttBlock]+=cp.sum(cp.transpose(E_postln_cache[currAttBlock, 0], [0, 2, 1])[:, None, :, :]@cp.transpose(E_soft_cache[currAttBlock], [0, 1, 3, 2])@cp.transpose(cp.reshape(G@sWo[currAttBlock].T, [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3]), axis=0)
        # print(cp.shape(E_postln_cache[currAttBlock, 0]))
        # print(cp.shape(sWv[currAttBlock]))
        dA = cp.transpose(cp.reshape(G@sWo[currAttBlock].T, [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3])@cp.transpose(V_cache[currAttBlock], [0, 1, 3, 2])


        d_softmax = (E_soft_cache[currAttBlock])*(dA-cp.sum(E_soft_cache[currAttBlock]*(dA), axis = 3, keepdims = True))


 
        g_Wq[currAttBlock] += cp.sum((1/cp.sqrt(k_DKey))*cp.transpose(E_postln_cache[currAttBlock, 0], [0, 2, 1])[:, None, :, :]@d_softmax@K_cache[currAttBlock], axis=0)
       
       

        g_Wk[currAttBlock] += cp.sum((1/cp.sqrt(k_DKey))*cp.transpose(E_postln_cache[currAttBlock, 0], [0, 2, 1])[:, None, :, :]@cp.transpose(d_softmax, [0, 1, 3, 2])@Q_cache[currAttBlock], axis=0)

        # print(cp.shape(cp.transpose(E_soft_cache[currAttBlock], [0, 1, 3, 2])))
        # print(cp.shape(G))
        # print(cp.shape(cp.transpose(sWv[currAttBlock], [0, 2, 1])))
        G1 = cp.transpose(E_soft_cache[currAttBlock], [0, 1, 3, 2])@cp.transpose(cp.reshape(G@sWo[currAttBlock].T, [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3])@cp.transpose(sWv[currAttBlock], [0, 2, 1])
        # G1 =  cp.transpose(E_soft_cache[currAttBlock], [0, 1, 3, 2])@G@cp.transpose(sWv[currAttBlock], [0, 2, 1])
        # print(cp.shape(G1))
        # print(cp.shape(d_softmax))
        # print(cp.shape(E_postln_cache[currAttBlock, 0][:, None, :, :]))
        # print(cp.shape(sWk[currAttBlock]))
        # print(cp.shape(cp.transpose(sWq[currAttBlock], [0, 2, 1])))

        G2 = (1/cp.sqrt(k_DKey))*d_softmax@(K_cache[currAttBlock])@cp.transpose(sWq[currAttBlock], [0, 2, 1])


        G3 = (1/cp.sqrt(k_DKey))*cp.transpose(d_softmax, [0, 1, 3, 2])@Q_cache[currAttBlock]@cp.transpose(sWk[currAttBlock], [0, 2, 1])
        # print(cp.shape(G1), " ", cp.shape(G2), " ", cp.shape(G3), " ")

        G_preatt = cp.sum(G1+G2+G3, axis=1)
        # print(cp.shape(G_preatt))
        G=cp.array(G_preatt)

        g_LNBias[currAttBlock, 0] += cp.sum(cp.sum(G, axis = 1), axis=0)
        g_LNGain[currAttBlock, 0] += cp.sum(cp.sum((E_midln_cache[currAttBlock, 0])*G, axis = 1), axis=0)

        Xhat_mean = E_midln_cache[currAttBlock, 0]*(cp.mean((G*sLNGain[currAttBlock, 0])*E_midln_cache[currAttBlock, 0], axis = 2, keepdims=True))
        G= G_preln+(1/cp.sqrt(cp.var(E_preln_cache[currAttBlock, 0], axis = 2, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 0]-cp.mean(G*sLNGain[currAttBlock, 0], axis = 2, keepdims=True)-Xhat_mean)
        currAttBlock-=1

    g_Wpos+=cp.sum(G, axis=0)
    # print(cp.shape(We_to_E))
    g_We += cp.sum(cp.transpose(We_to_E, [0, 2, 1])@G, axis=0)
    g_We[2] = cp.zeros(k_DModel)


k_BatchSize = params.k_BatchSize
k_Alpha = params.k_Alpha
k_Beta1 = params.k_Beta1
k_Beta2 = params.k_Beta2
k_Epsilon = params.k_Epsilon
k_Lambda = params.k_Lambda

g_We = cp.zeros((k_VocabSize, k_DModel), dtype=cp.float32)
g_Wpos = cp.zeros((k_ContextLength, k_DModel), dtype=cp.float32)
g_Wq = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32)
g_Wk = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32)
g_Wv = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel//k_Attheads), dtype=cp.float32)
g_Wo = cp.zeros((k_AttBlocks, k_DModel, k_DModel), dtype=cp.float32)
g_MLPW1 = cp.zeros((k_AttBlocks, k_DModel, k_DModel*4), dtype=cp.float32)
g_MLPW2 = cp.zeros((k_AttBlocks, k_DModel*4, k_DModel), dtype=cp.float32)
g_MLPb1 = cp.zeros((k_AttBlocks, 1, k_DModel*4), dtype=cp.float32)
g_MLPb2 = cp.zeros((k_AttBlocks, 1, k_DModel), dtype=cp.float32)
g_LNGain = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
g_LNBias = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
g_LW = cp.zeros((k_DModel, k_VocabSize), dtype=cp.float32)
g_LB = cp.zeros((k_VocabSize), dtype=cp.float32)

admt_We = cp.zeros((k_VocabSize, k_DModel), dtype=cp.float32)
admt_Wpos = cp.zeros((k_ContextLength, k_DModel), dtype=cp.float32)
admt_Wq = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32)
admt_Wk = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype = cp.float32)
admt_Wv = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel//k_Attheads), dtype=cp.float32)
admt_Wo = cp.zeros((k_AttBlocks, k_DModel, k_DModel), dtype=cp.float32)
admt_MLPW1 = cp.zeros((k_AttBlocks, k_DModel, k_DModel*4), dtype=cp.float32)
admt_MLPW2 = cp.zeros((k_AttBlocks, k_DModel*4, k_DModel), dtype=cp.float32)
admt_MLPb1 = cp.zeros((k_AttBlocks, 1, k_DModel*4), dtype=cp.float32)
admt_MLPb2 = cp.zeros((k_AttBlocks, 1, k_DModel), dtype=cp.float32)
admt_LNGain = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
admt_LNBias = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
admt_LW = cp.zeros((k_DModel, k_VocabSize), dtype=cp.float32)
admt_LB = cp.zeros((k_VocabSize), dtype=cp.float32)


advt_We = cp.zeros((k_VocabSize, k_DModel), dtype=cp.float32)
advt_Wpos = cp.zeros((k_ContextLength, k_DModel), dtype=cp.float32)
advt_Wq = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32)
advt_Wk = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey), dtype=cp.float32)
advt_Wv = cp.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel//k_Attheads), dtype=cp.float32)
advt_Wo = cp.zeros((k_AttBlocks, k_DModel, k_DModel), dtype=cp.float32)
advt_MLPW1 = cp.zeros((k_AttBlocks, k_DModel, k_DModel*4), dtype=cp.float32)
advt_MLPW2 = cp.zeros((k_AttBlocks, k_DModel*4, k_DModel), dtype=cp.float32)
advt_MLPb1 = cp.zeros((k_AttBlocks, 1, k_DModel*4), dtype=cp.float32)
advt_MLPb2 = cp.zeros((k_AttBlocks, 1, k_DModel), dtype=cp.float32)
advt_LNGain = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
advt_LNBias = cp.zeros((k_AttBlocks, 2, k_DModel), dtype=cp.float32)
advt_LW = cp.zeros((k_DModel, k_VocabSize), dtype=cp.float32)
advt_LB = cp.zeros((k_VocabSize), dtype=cp.float32)

#---------------------------------
#Data processing functions
def embed(svocabDict, case):
    embeded=[]

    words = case.split()

    processed_text = [
        [hex(ord(char))[2:] for char in word] + ["</w>"] 
        for word in words
    ]
    case=processed_text
    # with tqdm(total=len(case)) as pbar:
    for word in case:
        # pbar.update(1)
        i=0
        last = len(word)
        while(i!=len(word)):
            # print(word, " ", i,  " ", last, " ", ''.join(word[i:last]))
            if(''.join(word[i:last]) in svocabDict):
                embeded.append(''.join(word[i:last]))
                i=last
                last=len(word)
            else:
                last-=1
                if last <= i:
                    embeded.append("</UNKOWN>")
                    break
    return embeded
#---------------------------------
#Data processing

# train = [[]]



# directory_path = Path('./Training_Data/tokenized') 
# files_list = [p for p in directory_path.iterdir() if p.is_file()]
# for file in files_list:
#     with open(file, 'r', encoding='utf-8') as f:
#         for line in f:
#             train[0].append(line[:-1].replace('\w', '\w'))
num_times = params.num_times

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
#     for i in range(len(train)):
#         train[i] = embed(rule_list, train[i])
        # print(len(train[i])) #max length is aroudn 200, howvers around 40-70 usually

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

shift_factor = params.k_ShiftFactor
t = 0
avgloss = 0



from datasets import load_dataset

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)


i = 0
with open('results.txt', 'w', encoding="utf-8") as f:
    with tqdm(total=6000) as pbar:
        for text in ds['text']:
            amnt = 0
            
            if i == 6000:
                break
            i+=1

            if(i%1500==0):
                cp.savez(f"./Weights/weights{i}.npz", sWe=sWe, sWpos=sWpos, sWq=sWq, sWk=sWk, sWv=sWv, sMLPW1=sMLPW1, sMLPW2=sMLPW2, sMLPb1=sMLPb1, sMLPb2=sMLPb2, sLNGain=sLNGain, sLNBias=sLNBias, sLW=sLW, sLB=sLB, sWo = sWo)

            # print("tokenizing!")
            search_st = time.perf_counter()

            text=embed(svocabDict, text)

            search_et = time.perf_counter()
            # print(f"embedding took {search_et-search_st:.4f} seconds.")

            # print(text)
            pbar.update(1)

            curr_start =   k_ContextLength-1
            input_batch = []


            while(curr_start<len(text)-k_ContextLength):
                search_st = time.perf_counter()

                word = text[curr_start:(curr_start+k_ContextLength-1)]
                curr_start+=int(k_ContextLength/shift_factor)
                if(len(word)<k_ContextLength):
                    amnt+=1
                    input_batch.append(word)


                if(amnt%k_BatchSize == 0):   

                    E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache = fowardprop(input_batch, svocabDict)
                    # prediction = decode(E, vocab_list)
                    # print(prediction)
                    loss, onehot_cache = findLoss(E, input_batch, svocabDict)
                    backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache)
                    input_batch = []
                    loss = cp.array(loss)
                    t+=1
                    f.write(f"{cp.mean(loss)}\n")
                    avgloss=0
                    g_LB/=k_BatchSize
                    g_LW/=k_BatchSize
                    g_MLPb2/=k_BatchSize
                    g_MLPW2/=k_BatchSize
                    g_MLPb1/=k_BatchSize
                    g_MLPW1/=k_BatchSize
                    g_LNBias/=k_BatchSize
                    g_LNGain/=k_BatchSize
                    g_Wv/=k_BatchSize
                    g_Wq/=k_BatchSize
                    g_Wk/=k_BatchSize
                    g_Wo/=k_BatchSize
                    g_Wpos/=k_BatchSize
                    g_We/=k_BatchSize

                    admt_We = k_Beta1*admt_We + (1-k_Beta1)*g_We
                    admt_Wpos = k_Beta1*admt_Wpos + (1-k_Beta1)*g_Wpos
                    admt_Wq = k_Beta1*admt_Wq + (1-k_Beta1)*g_Wq
                    admt_Wk = k_Beta1*admt_Wk + (1-k_Beta1)*g_Wk
                    admt_Wv = k_Beta1*admt_Wv + (1-k_Beta1)*g_Wv
                    admt_Wo = k_Beta1*admt_Wo + (1-k_Beta1)*g_Wo
                    admt_MLPW1 = k_Beta1*admt_MLPW1 + (1-k_Beta1)*g_MLPW1
                    admt_MLPW2 = k_Beta1*admt_MLPW2 + (1-k_Beta1)*g_MLPW2
                    admt_MLPb1 = k_Beta1*admt_MLPb1 + (1-k_Beta1)*g_MLPb1
                    admt_MLPb2 = k_Beta1*admt_MLPb2 + (1-k_Beta1)*g_MLPb2
                    admt_LNGain = k_Beta1*admt_LNGain + (1-k_Beta1)*g_LNGain
                    admt_LNBias = k_Beta1*admt_LNBias + (1-k_Beta1)*g_LNBias
                    admt_LW = k_Beta1*admt_LW + (1-k_Beta1)*g_LW
                    admt_LB = k_Beta1*admt_LB + (1-k_Beta1)*g_LB


                    advt_We = k_Beta2*advt_We + (1-k_Beta2)*cp.square(g_We)
                    advt_Wpos = k_Beta2*advt_Wpos + (1-k_Beta2)*cp.square(g_Wpos)
                    advt_Wq = k_Beta2*advt_Wq + (1-k_Beta2)*cp.square(g_Wq)
                    advt_Wk = k_Beta2*advt_Wk + (1-k_Beta2)*cp.square(g_Wk)
                    advt_Wv = k_Beta2*advt_Wv + (1-k_Beta2)*cp.square(g_Wv)
                    advt_Wo = k_Beta2*advt_Wo + (1-k_Beta2)*cp.square(g_Wo)
                    advt_MLPW1 = k_Beta2*advt_MLPW1 + (1-k_Beta2)*cp.square(g_MLPW1)
                    advt_MLPW2 = k_Beta2*advt_MLPW2 + (1-k_Beta2)*cp.square(g_MLPW2)
                    advt_MLPb1 = k_Beta2*advt_MLPb1 + (1-k_Beta2)*cp.square(g_MLPb1)
                    advt_MLPb2 = k_Beta2*advt_MLPb2 + (1-k_Beta2)*cp.square(g_MLPb2)
                    advt_LNGain = k_Beta2*advt_LNGain + (1-k_Beta2)*cp.square(g_LNGain)
                    advt_LNBias = k_Beta2*advt_LNBias + (1-k_Beta2)*cp.square(g_LNBias)
                    advt_LW = k_Beta2*advt_LW + (1-k_Beta2)*cp.square(g_LW)
                    advt_LB = k_Beta2*advt_LB + (1-k_Beta2)*cp.square(g_LB)

                    sWe -= k_Alpha*(((admt_We/(1-k_Beta1**t))/(cp.sqrt(advt_We/(1-k_Beta2**t))+k_Epsilon))+sWe*k_Lambda)
                    sWpos -= k_Alpha*(((admt_Wpos/(1-k_Beta1**t))/(cp.sqrt(advt_Wpos/(1-k_Beta2**t))+k_Epsilon))+sWpos*k_Lambda)
                    sWq -= k_Alpha*(((admt_Wq/(1-k_Beta1**t))/(cp.sqrt(advt_Wq/(1-k_Beta2**t))+k_Epsilon))+sWq*k_Lambda)
                    sWk -= k_Alpha*(((admt_Wk/(1-k_Beta1**t))/(cp.sqrt(advt_Wk/(1-k_Beta2**t))+k_Epsilon))+sWk*k_Lambda)
                    sWv -= k_Alpha*(((admt_Wv/(1-k_Beta1**t))/(cp.sqrt(advt_Wv/(1-k_Beta2**t))+k_Epsilon))+sWv*k_Lambda)
                    sWo -= k_Alpha*(((admt_Wo/(1-k_Beta1**t))/(cp.sqrt(advt_Wo/(1-k_Beta2**t))+k_Epsilon))+sWo*k_Lambda)
                    sMLPW1 -= k_Alpha*(((admt_MLPW1/(1-k_Beta1**t))/(cp.sqrt(advt_MLPW1/(1-k_Beta2**t))+k_Epsilon))+sMLPW1*k_Lambda)
                    sMLPW2-= k_Alpha*(((admt_MLPW2/(1-k_Beta1**t))/(cp.sqrt(advt_MLPW2/(1-k_Beta2**t))+k_Epsilon))+sMLPW2*k_Lambda)
                    sMLPb1 -= k_Alpha*(((admt_MLPb1/(1-k_Beta1**t))/(cp.sqrt(advt_MLPb1/(1-k_Beta2**t))+k_Epsilon))+sMLPb1*k_Lambda)
                    sMLPb2 -= k_Alpha*(((admt_MLPb2/(1-k_Beta1**t))/(cp.sqrt(advt_MLPb2/(1-k_Beta2**t))+k_Epsilon))+sMLPb2*k_Lambda)
                    sLNGain -= k_Alpha*(((admt_LNGain/(1-k_Beta1**t))/(cp.sqrt(advt_LNGain/(1-k_Beta2**t))+k_Epsilon))+sLNGain*k_Lambda)
                    sLNBias-= k_Alpha*(((admt_LNBias/(1-k_Beta1**t))/(cp.sqrt(advt_LNBias/(1-k_Beta2**t))+k_Epsilon))+sLNBias*k_Lambda)
                    sLW -= k_Alpha*(((admt_LW/(1-k_Beta1**t))/(cp.sqrt(advt_LW/(1-k_Beta2**t))+k_Epsilon))+sLW*k_Lambda)
                    sLB -= k_Alpha*(((admt_LB/(1-k_Beta1**t))/(cp.sqrt(advt_LB/(1-k_Beta2**t))+k_Epsilon))+sLB*k_Lambda)
                    g_We.fill(0)
                    g_Wpos.fill(0)
                    g_Wq.fill(0)
                    g_Wk.fill(0)
                    g_Wv.fill(0)
                    g_Wo.fill(0)
                    g_MLPW1.fill(0)
                    g_MLPW2.fill(0)
                    g_MLPb1.fill(0)
                    g_MLPb2.fill(0)
                    g_LNGain.fill(0)
                    g_LNBias.fill(0)
                    g_LW.fill(0)
                    g_LB.fill(0)
                search_et = time.perf_counter()
                # print(f"loop took {search_et-search_st:.4f} seconds.")




cp.savez("./Weights/weights.npz", sWe=sWe, sWpos=sWpos, sWq=sWq, sWk=sWk, sWv=sWv, sMLPW1=sMLPW1, sMLPW2=sMLPW2, sMLPb1=sMLPb1, sMLPb2=sMLPb2, sLNGain=sLNGain, sLNBias=sLNBias, sLW=sLW, sLB=sLB, sWo = sWo)

import string
def is_hex(s):
    return all(c in string.hexdigits for c in s)


k_BatchSize=1
while(True):
    q = input("input_llm part of a word, a char, or something: ")
    q = embed(svocabDict, q)
    k = len(q)
    print(q) 
    while k < k_ContextLength:
        E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, temp1, temp2, temp3= fowardprop(q, svocabDict)
        prediction = decode(E[0], vocab_list)
        # loss, onehot_cache = findLoss(E, q, svocabDict)
        # print(loss)
        # print(prediction)
        text = prediction[k].split("</w>")
        q.append(prediction[k])
        post_processed = [
            bytes.fromhex(word).decode("utf-8") if is_hex(word) else word
            for word in text
        ]

        final_text = " ".join(post_processed)

        print(final_text, end='')
        k+=1
    print("")
    # print(q)