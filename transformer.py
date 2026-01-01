import numpy as np
from tqdm import tqdm # timing bar for nice looks
import params
import mlx.core as mx
from pathlib import Path
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
sWe = mx.random.normal((k_VocabSize, k_DModel), loc = 0, scale = 0.02)
sWe[2] = mx.zeros(k_DModel)
sWpos = mx.random.normal((k_ContextLength, k_DModel), loc = 0, scale = 0.02)
sWq = mx.random.normal((k_AttBlocks, k_Attheads, k_DModel, k_DKey), loc = 0, scale = mx.sqrt(2/(k_DModel+k_DKey)))/mx.sqrt(k_Attheads)
sWk = mx.random.normal((k_AttBlocks, k_Attheads, k_DModel, k_DKey), loc = 0, scale = mx.sqrt(2/(k_DModel+k_DKey)))/mx.sqrt(k_Attheads)
sWv = mx.random.normal((k_AttBlocks, k_Attheads, k_DModel, k_DModel), loc = 0, scale = mx.sqrt(1/(k_DModel)))/mx.sqrt(k_Attheads)
sMLPW1 = mx.random.normal((k_AttBlocks, k_DModel, k_DModel*4), loc = 0, scale = mx.sqrt(2/(k_DModel+4*k_DModel)))
sMLPW2 = mx.random.normal((k_AttBlocks, 4*k_DModel, k_DModel), loc = 0, scale = mx.sqrt(2/(k_DModel+4*k_DModel)))
sMLPb1 = mx.zeros((k_AttBlocks, 1, k_DModel*4))
sMLPb2 = mx.zeros((k_AttBlocks, 1, k_DModel))
sLNGain = mx.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
sLNBias = mx.zeros((k_AttBlocks, 2, k_DModel))
sLW = mx.random.normal((k_DModel, k_VocabSize), loc = 0, scale = mx.sqrt(2/(k_DModel+k_VocabSize)))
sLB = mx.zeros((1,k_VocabSize)) #ADDED TO ALL TOKENS
sSoftmaxMask =  mx.nan_to_num(-mx.inf*mx.triu(mx.ones((k_ContextLength, k_ContextLength)), k=1), nan = 0)
#E has dimension k_ContextLength x k_DModel
#testing
# sWq[0][0] = [[1,0],[0,1]]
# sWk[0][0] = [[1,0],[0,1]]
# sWv[0][0] = [[1,0],[0,1]]
# sMLPW1 = [[[1,0],[0,1]]]
# sMLPW2 = [[[1, 0], [0,1]]]
# sMLPb1 = mx.zeros((k_AttBlocks, 1, 2))
# sMLPb2 = mx.zeros((k_AttBlocks, 1, 2))
# sLNGain = mx.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
# sLNBias = mx.zeros((k_AttBlocks, 2, k_DModel))
# sLW = mx.random.normal(0, mx.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, 1))
# sLB = mx.ones((k_VocabSize)) #ADDED TO ALL TOKENS
# E = [[1, 2]]
#------------------
#TRANSFORMER FUNCTIONS
def layerNorm(E, attLayer, prePostMLP):
    temp = mx.nan_to_num((E-mx.mean(E, axis = -1, keepdims = True))/mx.sqrt((mx.nan_to_num(mx.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001)))
    return sLNBias[attLayer, prePostMLP] + temp * (sLNGain[attLayer, prePostMLP]), temp

def softmax(E):
    return mx.nan_to_num(mx.exp(E)/(mx.exp(E)@mx.ones((E.shape[1],1))), nan = 0)

def relu(E):
    return mx.maximum(0, E)

def relu_deriv(E):
    return mx.minimum(1, E)

def decode(E, svocabList):
    temp = mx.argmax(E, axis = -1)
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
    loss = 0
    i = 0
    onehot_cache = mx.zeros((k_ContextLength, k_VocabSize))

    while i < len(input_llm):
        if input_llm[i] in svocabDict:
            onehot_cache[i, svocabDict[input_llm[i]]] = 1
            loss += -mx.log(E[i][svocabDict[input_llm[i]]]+0.00001)
        else:
            onehot_cache[i, k_VocabSize-1] = 1
            loss += -mx.log(E[i][k_VocabSize-1]+0.00001)

        i+=1
    loss = (loss-mx.log(E[len(input_llm)][1]+0.001))/len(input_llm)
    onehot_cache[len(input_llm), 1] = 1
    return loss, onehot_cache
#------------------
#FORWARD PROPAGATE
def fowardprop(input_llm, svocabDict):
    #------------------
    #CACHEING
    E_preln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_midln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_postln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_soft_cache = mx.zeros((k_AttBlocks, k_Attheads, k_ContextLength, k_ContextLength))
    E_lin_cache = mx.zeros((k_ContextLength, k_DModel))
    E_relu_cache = mx.zeros((k_AttBlocks, k_ContextLength, k_DModel*4))
    

    padMask = mx.zeros((k_ContextLength, k_ContextLength))
    padMask[:, len(input_llm)+1:k_ContextLength] = -mx.inf

    temp = mx.zeros(k_VocabSize)
    temp[2] = 1
    We_to_E = mx.zeros((k_ContextLength, k_VocabSize))
    We_to_E[0, 0] = 1
    i = 1
    for j in input_llm:
        if j in svocabDict:
            We_to_E[i, [svocabDict[j]]] = 1
        else:
            We_to_E[i, k_VocabSize-1] = 1
        i+=1
    while(i<k_ContextLength):
        We_to_E[i, 2] = 1
        i+=1
    E = We_to_E@sWe

    i = 0
    while(i<len(input_llm)+1):
        E[i]+=sWpos[i]
        i+=1

    currAttBlock = 0
    while(currAttBlock < k_AttBlocks):
        E_preln_cache[currAttBlock, 0] = mx.array(E)
        E_ln, E_midln_cache[currAttBlock, 0] = layerNorm(E, currAttBlock, 0)
        E_postln_cache[currAttBlock, 0] = mx.array(E_ln)
        currAttHead = 0
        while(currAttHead<k_Attheads):
            E_soft_cache[currAttBlock, currAttHead] = softmax(1/mx.sqrt(k_DKey) * E_ln@sWq[currAttBlock, currAttHead]@(E_ln@sWk[currAttBlock, currAttHead]).T+sSoftmaxMask+padMask)
            E+= E_soft_cache[currAttBlock, currAttHead]@(E_ln@sWv[currAttBlock, currAttHead])
            currAttHead+=1
        E_preln_cache[currAttBlock, 1] = mx.array(E)
        E_ln, E_midln_cache[currAttBlock, 1] = layerNorm(E, currAttBlock, 1)
        E_postln_cache[currAttBlock, 1] = mx.array(E_ln)
        E_relu_cache[currAttBlock] = relu(E_ln@sMLPW1[currAttBlock]+sMLPb1[currAttBlock])
        E += E_relu_cache[currAttBlock]@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
        currAttBlock+=1
    E_lin_cache = mx.array(E)
    E=E@sLW+sLB
    E=softmax(E)

    return E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E
#------------------
#BACKPROP
def backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E):
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
    g_LW+=E_lin_cache.T@(E-onehot_cache)
    g_LB+=mx.sum((E-onehot_cache), axis=0)
    
    G = (E-onehot_cache)@sLW.T
    currAttBlock = k_AttBlocks-1
    while(currAttBlock>=0):
        G_preln = mx.array(G)
        g_MLPb2[currAttBlock]+=mx.sum(G, axis=0)
        g_MLPW2[currAttBlock]+=E_relu_cache[currAttBlock].T@G
        g_MLPb1[currAttBlock]+=mx.sum(((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))[0], axis=0)
        g_MLPW1[currAttBlock]+=(E_postln_cache[currAttBlock, 1].T) @ ((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))
        G = (G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])@sMLPW1[currAttBlock].T
        g_LNBias[currAttBlock, 1] += mx.sum(G, axis = 0)
        g_LNGain[currAttBlock, 1] += mx.sum((E_midln_cache[currAttBlock, 1])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 1]*(mx.mean((G*sLNGain[currAttBlock, 1])*E_midln_cache[currAttBlock, 1], axis = 1, keepdims=True))
        G= G_preln+(1/mx.sqrt(mx.var(E_preln_cache[currAttBlock, 1], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 1]-mx.mean(G*sLNGain[currAttBlock, 1], axis = 1, keepdims=True)-Xhat_mean)
        currAttHead = 0
        G_preatt = mx.zeros((k_ContextLength, k_DModel))
        G_preln = mx.array(G)
        while(currAttHead<k_Attheads):
            g_Wv[currAttBlock, currAttHead]+=E_postln_cache[currAttBlock, 0].T@(E_soft_cache[currAttBlock, currAttHead].T@G)
            V = E_postln_cache[currAttBlock, 0]@sWv[currAttBlock, currAttHead]
            d_softmax = (E_soft_cache[currAttBlock, currAttHead])*(G@V.T-mx.sum(E_soft_cache[currAttBlock, currAttHead]*(G@V.T), axis = 1, keepdims = True))
            g_Wq += (1/mx.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax@E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead]
            g_Wk += (1/mx.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]
            G1 =  E_soft_cache[currAttBlock, currAttHead].T@G@sWv[currAttBlock, currAttHead].T
            G2 = (1/mx.sqrt(k_DKey))*d_softmax@(E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead])@sWq[currAttBlock, currAttHead].T
            G3 = (1/mx.sqrt(k_DKey))*d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]@sWk[currAttBlock, currAttHead].T
            G_preatt += G1+G2+G3
            currAttHead+=1
        G=mx.array(G_preatt)

        g_LNBias[currAttBlock, 0] += mx.sum(G, axis = 0)
        g_LNGain[currAttBlock, 0] += mx.sum((E_midln_cache[currAttBlock, 0])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 0]*(mx.mean((G*sLNGain[currAttBlock, 0])*E_midln_cache[currAttBlock, 0], axis = 1, keepdims=True))
        G= G_preln+(1/mx.sqrt(mx.var(E_preln_cache[currAttBlock, 0], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 0]-mx.mean(G*sLNGain[currAttBlock, 0], axis = 1, keepdims=True)-Xhat_mean)
        currAttBlock-=1
    g_Wpos+=G
    g_We += We_to_E.T@G
    g_We[2] = mx.zeros(k_DModel)


k_BatchSize = params.k_BatchSize
k_Alpha = params.k_Alpha
k_Beta1 = params.k_Beta1
k_Beta2 = params.k_Beta2
k_Epsilon = params.k_Epsilon

g_We = mx.zeros((k_VocabSize, k_DModel))
g_Wpos = mx.zeros((k_ContextLength, k_DModel))
g_Wq = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
g_Wk = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
g_Wv = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
g_MLPW1 = mx.zeros((k_AttBlocks, k_DModel, k_DModel*4))
g_MLPW2 = mx.zeros((k_AttBlocks, k_DModel*4, k_DModel))
g_MLPb1 = mx.zeros((k_AttBlocks, 1, k_DModel*4))
g_MLPb2 = mx.zeros((k_AttBlocks, 1, k_DModel))
g_LNGain = mx.zeros((k_AttBlocks, 2, k_DModel))
g_LNBias = mx.zeros((k_AttBlocks, 2, k_DModel))
g_LW = mx.zeros((k_DModel, k_VocabSize))
g_LB = mx.zeros((k_VocabSize))

admt_We = mx.zeros((k_VocabSize, k_DModel))
admt_Wpos = mx.zeros((k_ContextLength, k_DModel))
admt_Wq = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
admt_Wk = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
admt_Wv = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
admt_MLPW1 = mx.zeros((k_AttBlocks, k_DModel, k_DModel*4))
admt_MLPW2 = mx.zeros((k_AttBlocks, k_DModel*4, k_DModel))
admt_MLPb1 = mx.zeros((k_AttBlocks, 1, k_DModel*4))
admt_MLPb2 = mx.zeros((k_AttBlocks, 1, k_DModel))
admt_LNGain = mx.zeros((k_AttBlocks, 2, k_DModel))
admt_LNBias = mx.zeros((k_AttBlocks, 2, k_DModel))
admt_LW = mx.zeros((k_DModel, k_VocabSize))
admt_LB = mx.zeros((k_VocabSize))


advt_We = mx.zeros((k_VocabSize, k_DModel))
advt_Wpos = mx.zeros((k_ContextLength, k_DModel))
advt_Wq = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
advt_Wk = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
advt_Wv = mx.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
advt_MLPW1 = mx.zeros((k_AttBlocks, k_DModel, k_DModel*4))
advt_MLPW2 = mx.zeros((k_AttBlocks, k_DModel*4, k_DModel))
advt_MLPb1 = mx.zeros((k_AttBlocks, 1, k_DModel*4))
advt_MLPb2 = mx.zeros((k_AttBlocks, 1, k_DModel))
advt_LNGain = mx.zeros((k_AttBlocks, 2, k_DModel))
advt_LNBias = mx.zeros((k_AttBlocks, 2, k_DModel))
advt_LW = mx.zeros((k_DModel, k_VocabSize))
advt_LB = mx.zeros((k_VocabSize))

#---------------------------------
#Data processing functions
def embed(rule_list, case):
    i = 0
    with tqdm(total=len(rule_list)) as pbar:
        while(i < len(rule_list)):
            pbar.update(1)
            j=0
            max_occ = rule_list[i]
            while(j < len(case)-1):
                if(max_occ == (case[j], case[j+1])):
                    case[j]+=case[j+1]
                    case.pop(j+1)
                    j-=1
                j+=1
            i+=1
    return case
#---------------------------------
#Data processing

train = []



directory_path = Path('./Training_Data') 
files_list = [p for p in directory_path.iterdir() if p.is_file()]
for file in files_list:
    with open(file, 'r') as f:
        train.append(list(f.read()))

num_times = params.num_times

with open('bpe_rules.txt', 'r') as f:
    rule_list = []
    i = 0
    while(i < num_times):
        rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
        i+=1
    for i in range(len(train)):
        train[i] = embed(rule_list, train[i])
        # print(len(train[i])) #max length is aroudn 200, howvers around 40-70 usually

svocabDict = {}
vocab_list = []
svocabDict[" "] = 3
svocabDict["END_TOKEN"] = 1
svocabDict["START_TOKEN"] = 0
svocabDict["PAD_TOKEN"] = 2
i = 4



with open('bpe_vocablist.txt', 'r') as f:
    for line in f:
        vocab_list.append(line[:-1].replace('\\n', '\n'))
        svocabDict[vocab_list[-1]] = i
        i+=1
amnt = 0
a = 0
loss = 0


# E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(train[0], svocabDict)
# prediction = decode(E, vocab_list)
# print(prediction)
# loss, onehot_cache = findLoss(E, train[0], svocabDict)
# print(loss)
# backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E)

#----------SINGLE EXAMPLE TRAINING
# word = train[0]
# with tqdm(total=1000) as pbar:
#     with open('results.txt', 'w') as f:
#         while(a<1000):
#                 pbar.update(1)
#                 if(len(word)<k_ContextLength):
#                     word = list(word)
#                     E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(word, svocabDict)
#                     prediction = decode(E, vocab_list)
#                     # print(prediction)
#                     loss, onehot_cache = findLoss(E, word, svocabDict)
#                     backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E)
#                     f.write(f"{loss}\n")
#                 if(a%k_BatchSize == 0):
#                     g_LB/=k_BatchSize
#                     g_LW/=k_BatchSize
#                     g_MLPb2/=k_BatchSize
#                     g_MLPW2/=k_BatchSize
#                     g_MLPb1/=k_BatchSize
#                     g_MLPW1/=k_BatchSize
#                     g_LNBias/=k_BatchSize
#                     g_LNGain/=k_BatchSize
#                     g_Wv/=k_BatchSize
#                     g_Wq/=k_BatchSize
#                     g_Wk/=k_BatchSize
#                     g_Wpos/=k_BatchSize
#                     g_We/=k_BatchSize

#                     admt_We = k_Beta1*admt_We + (1-k_Beta1)*g_We
#                     admt_Wpos = k_Beta1*admt_Wpos + (1-k_Beta1)*g_Wpos
#                     admt_Wq = k_Beta1*admt_Wq + (1-k_Beta1)*g_Wq
#                     admt_Wk = k_Beta1*admt_Wk + (1-k_Beta1)*g_Wk
#                     admt_Wv = k_Beta1*admt_Wv + (1-k_Beta1)*g_Wv
#                     admt_MLPW1 = k_Beta1*admt_MLPW1 + (1-k_Beta1)*g_MLPW1
#                     admt_MLPW2 = k_Beta1*admt_MLPW2 + (1-k_Beta1)*g_MLPW2
#                     admt_MLPb1 = k_Beta1*admt_MLPb1 + (1-k_Beta1)*g_MLPb1
#                     admt_MLPb2 = k_Beta1*admt_MLPb2 + (1-k_Beta1)*g_MLPb2
#                     admt_LNGain = k_Beta1*admt_LNGain + (1-k_Beta1)*g_LNGain
#                     admt_LNBias = k_Beta1*admt_LNBias + (1-k_Beta1)*g_LNBias
#                     admt_LW = k_Beta1*admt_LW + (1-k_Beta1)*g_LW
#                     admt_LB = k_Beta1*admt_LB + (1-k_Beta1)*g_LB


#                     advt_We = k_Beta2*advt_We + (1-k_Beta2)*mx.square(g_We)
#                     advt_Wpos = k_Beta2*advt_Wpos + (1-k_Beta2)*mx.square(g_Wpos)
#                     advt_Wq = k_Beta2*advt_Wq + (1-k_Beta2)*mx.square(g_Wq)
#                     advt_Wk = k_Beta2*advt_Wk + (1-k_Beta2)*mx.square(g_Wk)
#                     advt_Wv = k_Beta2*advt_Wv + (1-k_Beta2)*mx.square(g_Wv)
#                     advt_MLPW1 = k_Beta2*advt_MLPW1 + (1-k_Beta2)*mx.square(g_MLPW1)
#                     advt_MLPW2 = k_Beta2*advt_MLPW2 + (1-k_Beta2)*mx.square(g_MLPW2)
#                     advt_MLPb1 = k_Beta2*advt_MLPb1 + (1-k_Beta2)*mx.square(g_MLPb1)
#                     advt_MLPb2 = k_Beta2*advt_MLPb2 + (1-k_Beta2)*mx.square(g_MLPb2)
#                     advt_LNGain = k_Beta2*advt_LNGain + (1-k_Beta2)*mx.square(g_LNGain)
#                     advt_LNBias = k_Beta2*advt_LNBias + (1-k_Beta2)*mx.square(g_LNBias)
#                     advt_LW = k_Beta2*advt_LW + (1-k_Beta2)*mx.square(g_LW)
#                     advt_LB = k_Beta2*advt_LB + (1-k_Beta2)*mx.square(g_LB)

#                     sWe -= k_Alpha*(admt_We/(1-k_Beta1))/(mx.sqrt(advt_We/(1-k_Beta2))+k_Epsilon)
#                     sWpos -= k_Alpha*(admt_Wpos/(1-k_Beta1))/(np.sqrt(advt_Wpos/(1-k_Beta2))+k_Epsilon)
#                     sWq -= k_Alpha*(admt_Wq/(1-k_Beta1))/(np.sqrt(advt_Wq/(1-k_Beta2))+k_Epsilon)
#                     sWk -= k_Alpha*(admt_Wk/(1-k_Beta1))/(np.sqrt(advt_Wk/(1-k_Beta2))+k_Epsilon)
#                     sWv -= k_Alpha*(admt_Wv/(1-k_Beta1))/(np.sqrt(advt_Wv/(1-k_Beta2))+k_Epsilon)
#                     sMLPW1 -= k_Alpha*(admt_MLPW1/(1-k_Beta1))/(np.sqrt(advt_MLPW1/(1-k_Beta2))+k_Epsilon)
#                     sMLPW2-= k_Alpha*(admt_MLPW2/(1-k_Beta1))/(np.sqrt(advt_MLPW2/(1-k_Beta2))+k_Epsilon)
#                     sMLPb1 -= k_Alpha*(admt_MLPb1/(1-k_Beta1))/(np.sqrt(advt_MLPb1/(1-k_Beta2))+k_Epsilon)
#                     sMLPb2 -= k_Alpha*(admt_MLPb2/(1-k_Beta1))/(np.sqrt(advt_MLPb2/(1-k_Beta2))+k_Epsilon)
#                     sLNGain -= k_Alpha*(admt_LNGain/(1-k_Beta1))/(np.sqrt(advt_LNGain/(1-k_Beta2))+k_Epsilon)
#                     sLNBias-= k_Alpha*(admt_LNBias/(1-k_Beta1))/(np.sqrt(advt_LNBias/(1-k_Beta2))+k_Epsilon)
#                     sLW -= k_Alpha*(admt_LW/(1-k_Beta1))/(np.sqrt(advt_LW/(1-k_Beta2))+k_Epsilon)
#                     sLB -= k_Alpha*(admt_LB/(1-k_Beta1))/(np.sqrt(advt_LB/(1-k_Beta2))+k_Epsilon)
                    
#                     g_We = np.zeros((k_VocabSize, k_DModel))
#                     g_Wpos = np.zeros((k_ContextLength, k_DModel))
#                     g_Wq = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
#                     g_Wk = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
#                     g_Wv = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
#                     g_MLPW1 = np.zeros((k_AttBlocks, k_DModel, k_DModel*4))
#                     g_MLPW2 = np.zeros((k_AttBlocks, k_DModel*4, k_DModel))
#                     g_MLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
#                     g_MLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
#                     g_LNGain = np.zeros((k_AttBlocks, 2, k_DModel))
#                     g_LNBias = np.zeros((k_AttBlocks, 2, k_DModel))
#                     g_LW = np.zeros((k_DModel, k_VocabSize))
#                     g_LB = np.zeros((k_VocabSize))
#                 a+=1
# E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(word, svocabDict)
# prediction = decode(E, vocab_list)
# print(prediction)
# loss, onehot_cache = findLoss(E, word, svocabDict)
# print(loss)


if(True):
    with open('results.txt', 'w') as f:
        for text in train:
            with tqdm(total=int(2*(len(text)/k_ContextLength))) as pbar:
                curr_start = 0


                while(curr_start<len(text)-k_ContextLength):
                    word = text[curr_start:(curr_start+k_ContextLength-1)]
                    curr_start+=int(k_ContextLength/2)
                    pbar.update(1)

                    if(len(word)<k_ContextLength):
                        amnt+=1
                        word = list(word)
                        E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(word, svocabDict)
                        prediction = decode(E, vocab_list)
                        # print(prediction)
                        loss, onehot_cache = findLoss(E, word, svocabDict)
                        backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E)
                        f.write(f"{loss}\n")

                    if(amnt%k_BatchSize == 0):
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
                        g_Wpos/=k_BatchSize
                        g_We/=k_BatchSize

                        admt_We = k_Beta1*admt_We + (1-k_Beta1)*g_We
                        admt_Wpos = k_Beta1*admt_Wpos + (1-k_Beta1)*g_Wpos
                        admt_Wq = k_Beta1*admt_Wq + (1-k_Beta1)*g_Wq
                        admt_Wk = k_Beta1*admt_Wk + (1-k_Beta1)*g_Wk
                        admt_Wv = k_Beta1*admt_Wv + (1-k_Beta1)*g_Wv
                        admt_MLPW1 = k_Beta1*admt_MLPW1 + (1-k_Beta1)*g_MLPW1
                        admt_MLPW2 = k_Beta1*admt_MLPW2 + (1-k_Beta1)*g_MLPW2
                        admt_MLPb1 = k_Beta1*admt_MLPb1 + (1-k_Beta1)*g_MLPb1
                        admt_MLPb2 = k_Beta1*admt_MLPb2 + (1-k_Beta1)*g_MLPb2
                        admt_LNGain = k_Beta1*admt_LNGain + (1-k_Beta1)*g_LNGain
                        admt_LNBias = k_Beta1*admt_LNBias + (1-k_Beta1)*g_LNBias
                        admt_LW = k_Beta1*admt_LW + (1-k_Beta1)*g_LW
                        admt_LB = k_Beta1*admt_LB + (1-k_Beta1)*g_LB


                        advt_We = k_Beta2*advt_We + (1-k_Beta2)*mx.square(g_We)
                        advt_Wpos = k_Beta2*advt_Wpos + (1-k_Beta2)*mx.square(g_Wpos)
                        advt_Wq = k_Beta2*advt_Wq + (1-k_Beta2)*mx.square(g_Wq)
                        advt_Wk = k_Beta2*advt_Wk + (1-k_Beta2)*mx.square(g_Wk)
                        advt_Wv = k_Beta2*advt_Wv + (1-k_Beta2)*mx.square(g_Wv)
                        advt_MLPW1 = k_Beta2*advt_MLPW1 + (1-k_Beta2)*mx.square(g_MLPW1)
                        advt_MLPW2 = k_Beta2*advt_MLPW2 + (1-k_Beta2)*mx.square(g_MLPW2)
                        advt_MLPb1 = k_Beta2*advt_MLPb1 + (1-k_Beta2)*mx.square(g_MLPb1)
                        advt_MLPb2 = k_Beta2*advt_MLPb2 + (1-k_Beta2)*mx.square(g_MLPb2)
                        advt_LNGain = k_Beta2*advt_LNGain + (1-k_Beta2)*mx.square(g_LNGain)
                        advt_LNBias = k_Beta2*advt_LNBias + (1-k_Beta2)*mx.square(g_LNBias)
                        advt_LW = k_Beta2*advt_LW + (1-k_Beta2)*mx.square(g_LW)
                        advt_LB = k_Beta2*advt_LB + (1-k_Beta2)*mx.square(g_LB)

                
                        sWe -= k_Alpha*(admt_We/(1-k_Beta1))/(mx.sqrt(advt_We/(1-k_Beta2))+k_Epsilon)
                        sWpos -= k_Alpha*(admt_Wpos/(1-k_Beta1))/(mx.sqrt(advt_Wpos/(1-k_Beta2))+k_Epsilon)
                        sWq -= k_Alpha*(admt_Wq/(1-k_Beta1))/(mx.sqrt(advt_Wq/(1-k_Beta2))+k_Epsilon)
                        sWk -= k_Alpha*(admt_Wk/(1-k_Beta1))/(mx.sqrt(advt_Wk/(1-k_Beta2))+k_Epsilon)
                        sWv -= k_Alpha*(admt_Wv/(1-k_Beta1))/(mx.sqrt(advt_Wv/(1-k_Beta2))+k_Epsilon)
                        sMLPW1 -= k_Alpha*(admt_MLPW1/(1-k_Beta1))/(mx.sqrt(advt_MLPW1/(1-k_Beta2))+k_Epsilon)
                        sMLPW2-= k_Alpha*(admt_MLPW2/(1-k_Beta1))/(mx.sqrt(advt_MLPW2/(1-k_Beta2))+k_Epsilon)
                        sMLPb1 -= k_Alpha*(admt_MLPb1/(1-k_Beta1))/(mx.sqrt(advt_MLPb1/(1-k_Beta2))+k_Epsilon)
                        sMLPb2 -= k_Alpha*(admt_MLPb2/(1-k_Beta1))/(mx.sqrt(advt_MLPb2/(1-k_Beta2))+k_Epsilon)
                        sLNGain -= k_Alpha*(admt_LNGain/(1-k_Beta1))/(mx.sqrt(advt_LNGain/(1-k_Beta2))+k_Epsilon)
                        sLNBias-= k_Alpha*(admt_LNBias/(1-k_Beta1))/(mx.sqrt(advt_LNBias/(1-k_Beta2))+k_Epsilon)
                        sLW -= k_Alpha*(admt_LW/(1-k_Beta1))/(mx.sqrt(advt_LW/(1-k_Beta2))+k_Epsilon)
                        sLB -= k_Alpha*(admt_LB/(1-k_Beta1))/(mx.sqrt(advt_LB/(1-k_Beta2))+k_Epsilon)
                        
                        g_We*=0
                        g_Wpos*=0
                        g_Wq*=0
                        g_Wk*=0
                        g_Wv*=0
                        g_MLPW1*=0
                        g_MLPW2*=0
                        g_MLPb1*=0
                        g_MLPb2*=0
                        g_LNGain*=0
                        g_LNBias*=0
                        g_LW*=0
                        g_LB*=0


while(True):
    q = input("input_llm part of a word, a char, or something: ")
    q = list(q)
    q = embed(rule_list, q)
    k = len(q)
    print(q)
    while k < k_ContextLength:
        E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(q, svocabDict)
        prediction = decode(E, vocab_list)
        loss, onehot_cache = findLoss(E, q, svocabDict)
        # print(loss)
        # print(prediction)
        q.append(prediction[k])
        print(prediction[k], end='')
        k+=1
    print("")
    # print(q)