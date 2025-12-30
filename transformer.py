import numpy as np
from tqdm import tqdm # timing bar for nice looks
print("YAYAA")
#------------------
#PARAM DECLARATIONS
k_DModel = 256 #32
k_ContextLength = 8#8
k_VocabSize = 26+5 #plus four for start, end, pad, and space      last one is no idea
k_Attheads = 4
k_AttBlocks = 2
#these should all be the same
k_DQuery = 64
k_DKey = k_DQuery
#------------------
#SETUP DATA STRUCTS
svocabList = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
svocabDict = {}
sWe = np.random.normal(0, 0.02, size = (k_VocabSize, k_DModel))
sWe[2] = np.zeros(k_DModel)
sWpos = np.random.normal(0, 0.02, size = (k_ContextLength, k_DModel))
sWq = np.random.normal(0, np.sqrt(2/(k_DModel+k_DKey)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DKey))/np.sqrt(k_Attheads)
sWk = np.random.normal(0, np.sqrt(2/(k_DModel+k_DKey)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DKey))/np.sqrt(k_Attheads)
sWv = np.random.normal(0, np.sqrt(1/(k_DModel)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DModel))/np.sqrt(k_Attheads)
sMLPW1 = np.random.normal(0, np.sqrt(2/(k_DModel+4*k_DModel)), size = (k_AttBlocks, k_DModel, k_DModel*4))
sMLPW2 = np.random.normal(0, np.sqrt(2/(k_DModel+4*k_DModel)), size = (k_AttBlocks, k_DModel*4, k_DModel))
sMLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
sMLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
sLNGain = np.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
sLNBias = np.zeros((k_AttBlocks, 2, k_DModel))
sLW = np.random.normal(0, np.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, k_VocabSize))
sLB = np.zeros((1,k_VocabSize)) #ADDED TO ALL TOKENS
sSoftmaxMask = np.nan_to_num(-np.inf*np.triu(np.ones((k_ContextLength, k_ContextLength)), k=1), nan = 0)
#E has dimension k_ContextLength x k_DModel
#testing
# sWq[0][0] = [[1,0],[0,1]]
# sWk[0][0] = [[1,0],[0,1]]
# sWv[0][0] = [[1,0],[0,1]]
# sMLPW1 = [[[1,0],[0,1]]]
# sMLPW2 = [[[1, 0], [0,1]]]
# sMLPb1 = np.zeros((k_AttBlocks, 1, 2))
# sMLPb2 = np.zeros((k_AttBlocks, 1, 2))
# sLNGain = np.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
# sLNBias = np.zeros((k_AttBlocks, 2, k_DModel))
# sLW = np.random.normal(0, np.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, 1))
# sLB = np.ones((k_VocabSize)) #ADDED TO ALL TOKENS
# E = [[1, 2]]
#------------------
#PRE PROCESS
i = 4
for j in svocabList:
    svocabDict[j] = i
    i+=1
svocabDict[" "] = 3
#------------------
#TRANSFORMER FUNCTIONS
def layerNorm(E, attLayer, prePostMLP):
    temp = np.nan_to_num((E-np.mean(E, axis = -1, keepdims = True))/np.sqrt((np.nan_to_num(np.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001)))
    return sLNBias[attLayer, prePostMLP] + temp * (sLNGain[attLayer, prePostMLP]), temp

def softmax(E):
    return np.nan_to_num(np.exp(E)/(np.exp(E)@np.ones((E.shape[1],1))), nan = 0)

def relu(E):
    return np.maximum(0, E)

def relu_deriv_elementwise(e):
    return 1 if e>0 else 0

def decode(E):
    temp = np.argmax(E, axis = -1)
    answer = []
    for i in temp:
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
def findLoss(E, input_llm):
    loss = 0
    i = 0
    onehot_cache = np.zeros((k_ContextLength, k_VocabSize))

    while i < len(input_llm):
        if input_llm[i] in svocabDict:
            onehot_cache[i, svocabDict[input_llm[i]]] = 1
            loss += -np.log(E[i][svocabDict[input_llm[i]]]+0.00001)
        else:
            onehot_cache[i, k_VocabSize-1] = 1
            loss += -np.log(E[i][k_VocabSize-1]+0.00001)

        i+=1
    loss = (loss-np.log(E[len(input_llm)][1]+0.001))/len(input_llm)
    onehot_cache[len(input_llm), 1] = 1
    return loss, onehot_cache
#------------------
#INPUT
# input_llm = ["c", "a", "t"]
#------------------
#FORWARD PROPAGATE
def fowardprop(input_llm):
    #------------------
    #CACHEING
    E_preln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_midln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_postln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_soft_cache = np.empty((k_AttBlocks, k_Attheads, k_ContextLength, k_ContextLength))
    E_lin_cache = np.empty((k_ContextLength, k_DModel))
    E_relu_cache = np.empty((k_AttBlocks, k_ContextLength, k_DModel*4))
    svocabDict[" "] = 3
    svocabDict["END_TOKEN"] = 1
    svocabDict["START_TOKEN"] = 0
    svocabDict["PAD_TOKEN"] = 2

    padMask = np.zeros((k_ContextLength, k_ContextLength))
    padMask[:, len(input_llm)+1:k_ContextLength] = -np.inf

    temp = np.zeros(k_VocabSize)
    temp[2] = 1
    We_to_E = np.zeros((k_ContextLength, k_VocabSize))
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
        E_preln_cache[currAttBlock, 0] = E.copy()
        E, E_midln_cache[currAttBlock, 0] = layerNorm(E, currAttBlock, 0)
        E_postln_cache[currAttBlock, 0] = E.copy()
        currAttHead = 0
        while(currAttHead<k_Attheads):
            E_soft_cache[currAttBlock, currAttHead] = softmax(1/np.sqrt(k_DKey) * E@sWq[currAttBlock, currAttHead]@(E@sWk[currAttBlock, currAttHead]).T+sSoftmaxMask+padMask)
            E+= E_soft_cache[currAttBlock, currAttHead]@(E@sWv[currAttBlock, currAttHead])
            currAttHead+=1
        E_preln_cache[currAttBlock, 1] = E.copy()
        E, E_midln_cache[currAttBlock, 1] = layerNorm(E, currAttBlock, 1)
        E_postln_cache[currAttBlock, 1] = E.copy()
        E_relu_cache[currAttBlock] = relu(E@sMLPW1[currAttBlock]+sMLPb1[currAttBlock])
        E += E_relu_cache[currAttBlock]@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
        currAttBlock+=1
    E_lin_cache = E.copy()
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
    
    relu_deriv = np.vectorize(relu_deriv_elementwise)

    g_LW+=E_lin_cache.T@(E-onehot_cache)
    g_LB+=np.sum((E-onehot_cache), axis=0)
    
    G = (E-onehot_cache)@sLW.T
    currAttBlock = k_AttBlocks-1
    while(currAttBlock>=0):
        g_MLPb2[currAttBlock]+=np.sum(G, axis=0)
        g_MLPW2[currAttBlock]+=E_relu_cache[currAttBlock].T@G
        g_MLPb1[currAttBlock]+=np.sum(((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))[0], axis=0)
        g_MLPW1[currAttBlock]+=(E_postln_cache[currAttBlock, 1].T) @ ((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))
        G = (G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])@sMLPW1[currAttBlock].T
        g_LNBias[currAttBlock, 1] += np.sum(G, axis = 0)
        g_LNGain[currAttBlock, 1] += np.sum((E_midln_cache[currAttBlock, 1])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 1]*(np.mean((G*sLNGain[currAttBlock, 1])*E_midln_cache[currAttBlock, 1], axis = 1, keepdims=True))
        G= (1/np.sqrt(np.var(E_preln_cache[currAttBlock, 1], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 1]-np.mean(G*sLNGain[currAttBlock, 1], axis = 1, keepdims=True)-Xhat_mean)
        currAttHead = 0
        G_preatt = np.zeros((k_ContextLength, k_DModel))
        while(currAttHead<k_Attheads):
            g_Wv[currAttBlock, currAttHead]+=E_postln_cache[currAttBlock, 0].T@(E_soft_cache[currAttBlock, currAttHead].T@G)
            V = E_postln_cache[currAttBlock, 0]@sWv[currAttBlock, currAttHead]
            d_softmax = (E_soft_cache[currAttBlock, currAttHead])*(G@V.T-np.sum(E_soft_cache[currAttBlock, currAttHead]*(G@V.T), axis = 1, keepdims = True))
            g_Wq += (1/np.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax@E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead]
            g_Wk += (1/np.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]
            G1 =  E_soft_cache[currAttBlock, currAttHead].T@G@sWv[currAttBlock, currAttHead].T
            G2 = (1/np.sqrt(k_DKey))*d_softmax@(E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead])@sWq[currAttBlock, currAttHead].T
            G3 = (1/np.sqrt(k_DKey))*d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]@sWk[currAttBlock, currAttHead].T
            G_preatt += G1+G2+G3
            currAttHead+=1
        G=G_preatt
        g_LNBias[currAttBlock, 0] += np.sum(G, axis = 0)
        g_LNGain[currAttBlock, 0] += np.sum((E_midln_cache[currAttBlock, 0])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 0]*(np.mean((G*sLNGain[currAttBlock, 0])*E_midln_cache[currAttBlock, 0], axis = 1, keepdims=True))
        G= (1/np.sqrt(np.var(E_preln_cache[currAttBlock, 0], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 0]-np.mean(G*sLNGain[currAttBlock, 0], axis = 1, keepdims=True)-Xhat_mean)
        currAttBlock-=1
    g_Wpos+=G
    g_We += We_to_E.T@G
    g_We[2] = np.zeros(k_DModel)







k_BatchSize = 50
k_Alpha = 0.00005
k_Beta1 = 0.9
k_Beta2 = 0.98
k_Epsilon = 0.00000001

g_We = np.zeros((k_VocabSize, k_DModel))
g_Wpos = np.zeros((k_ContextLength, k_DModel))
g_Wq = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
g_Wk = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
g_Wv = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
g_MLPW1 = np.zeros((k_AttBlocks, k_DModel, k_DModel*4))
g_MLPW2 = np.zeros((k_AttBlocks, k_DModel*4, k_DModel))
g_MLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
g_MLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
g_LNGain = np.zeros((k_AttBlocks, 2, k_DModel))
g_LNBias = np.zeros((k_AttBlocks, 2, k_DModel))
g_LW = np.zeros((k_DModel, k_VocabSize))
g_LB = np.zeros((k_VocabSize))

admt_We = np.zeros((k_VocabSize, k_DModel))
admt_Wpos = np.zeros((k_ContextLength, k_DModel))
admt_Wq = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
admt_Wk = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
admt_Wv = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
admt_MLPW1 = np.zeros((k_AttBlocks, k_DModel, k_DModel*4))
admt_MLPW2 = np.zeros((k_AttBlocks, k_DModel*4, k_DModel))
admt_MLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
admt_MLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
admt_LNGain = np.zeros((k_AttBlocks, 2, k_DModel))
admt_LNBias = np.zeros((k_AttBlocks, 2, k_DModel))
admt_LW = np.zeros((k_DModel, k_VocabSize))
admt_LB = np.zeros((k_VocabSize))


advt_We = np.zeros((k_VocabSize, k_DModel))
advt_Wpos = np.zeros((k_ContextLength, k_DModel))
advt_Wq = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
advt_Wk = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
advt_Wv = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
advt_MLPW1 = np.zeros((k_AttBlocks, k_DModel, k_DModel*4))
advt_MLPW2 = np.zeros((k_AttBlocks, k_DModel*4, k_DModel))
advt_MLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
advt_MLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
advt_LNGain = np.zeros((k_AttBlocks, 2, k_DModel))
advt_LNBias = np.zeros((k_AttBlocks, 2, k_DModel))
advt_LW = np.zeros((k_DModel, k_VocabSize))
advt_LB = np.zeros((k_VocabSize))


import string

translator = str.maketrans('', '', string.punctuation)

word_list = []
with open('romeo_juliet_proj_gutenburg.txt', 'r') as f:
    while(len(word_list) < 100):
        x = next(f).lower()
        x = x.translate(translator)
        x = x.split()
        word_list+=x



amnt = 0
a = 0
loss = 0
with open('results.txt', 'w') as f:
    while(a<1):
        with tqdm(total=len(word_list)) as pbar:
            for word in word_list:
                pbar.update(1)

                if(len(word)<k_ContextLength):
                    amnt+=1
                    word = list(word)
                    E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(word)
                    prediction = decode(E)
                    # print(prediction)
                    loss, onehot_cache = findLoss(E, word)
                    # print(loss)
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


                    advt_We = k_Beta2*advt_We + (1-k_Beta2)*np.square(g_We)
                    advt_Wpos = k_Beta2*advt_Wpos + (1-k_Beta2)*np.square(g_Wpos)
                    advt_Wq = k_Beta2*advt_Wq + (1-k_Beta2)*np.square(g_Wq)
                    advt_Wk = k_Beta2*advt_Wk + (1-k_Beta2)*np.square(g_Wk)
                    advt_Wv = k_Beta2*advt_Wv + (1-k_Beta2)*np.square(g_Wv)
                    advt_MLPW1 = k_Beta2*advt_MLPW1 + (1-k_Beta2)*np.square(g_MLPW1)
                    advt_MLPW2 = k_Beta2*advt_MLPW2 + (1-k_Beta2)*np.square(g_MLPW2)
                    advt_MLPb1 = k_Beta2*advt_MLPb1 + (1-k_Beta2)*np.square(g_MLPb1)
                    advt_MLPb2 = k_Beta2*advt_MLPb2 + (1-k_Beta2)*np.square(g_MLPb2)
                    advt_LNGain = k_Beta2*advt_LNGain + (1-k_Beta2)*np.square(g_LNGain)
                    advt_LNBias = k_Beta2*advt_LNBias + (1-k_Beta2)*np.square(g_LNBias)
                    advt_LW = k_Beta2*advt_LW + (1-k_Beta2)*np.square(g_LW)
                    advt_LB = k_Beta2*advt_LB + (1-k_Beta2)*np.square(g_LB)

                    sWe -= k_Alpha*(admt_We/(1-k_Beta1))/(np.sqrt(advt_We/(1-k_Beta2))+k_Epsilon)
                    sWpos -= k_Alpha*(admt_Wpos/(1-k_Beta1))/(np.sqrt(advt_Wpos/(1-k_Beta2))+k_Epsilon)
                    sWq -= k_Alpha*(admt_Wq/(1-k_Beta1))/(np.sqrt(advt_Wq/(1-k_Beta2))+k_Epsilon)
                    sWk -= k_Alpha*(admt_Wk/(1-k_Beta1))/(np.sqrt(advt_Wk/(1-k_Beta2))+k_Epsilon)
                    sWv -= k_Alpha*(admt_Wv/(1-k_Beta1))/(np.sqrt(advt_Wv/(1-k_Beta2))+k_Epsilon)
                    sMLPW1 -= k_Alpha*(admt_MLPW1/(1-k_Beta1))/(np.sqrt(advt_MLPW1/(1-k_Beta2))+k_Epsilon)
                    sMLPW2-= k_Alpha*(admt_MLPW2/(1-k_Beta1))/(np.sqrt(advt_MLPW2/(1-k_Beta2))+k_Epsilon)
                    sMLPb1 -= k_Alpha*(admt_MLPb1/(1-k_Beta1))/(np.sqrt(advt_MLPb1/(1-k_Beta2))+k_Epsilon)
                    sMLPb2 -= k_Alpha*(admt_MLPb2/(1-k_Beta1))/(np.sqrt(advt_MLPb2/(1-k_Beta2))+k_Epsilon)
                    sLNGain -= k_Alpha*(admt_LNGain/(1-k_Beta1))/(np.sqrt(advt_LNGain/(1-k_Beta2))+k_Epsilon)
                    sLNBias-= k_Alpha*(admt_LNBias/(1-k_Beta1))/(np.sqrt(advt_LNBias/(1-k_Beta2))+k_Epsilon)
                    sLW -= k_Alpha*(admt_LW/(1-k_Beta1))/(np.sqrt(advt_LW/(1-k_Beta2))+k_Epsilon)
                    sLB -= k_Alpha*(admt_LB/(1-k_Beta1))/(np.sqrt(advt_LB/(1-k_Beta2))+k_Epsilon)
                    
                    g_We = np.zeros((k_VocabSize, k_DModel))
                    g_Wpos = np.zeros((k_ContextLength, k_DModel))
                    g_Wq = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
                    g_Wk = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DKey))
                    g_Wv = np.zeros((k_AttBlocks, k_Attheads, k_DModel, k_DModel))
                    g_MLPW1 = np.zeros((k_AttBlocks, k_DModel, k_DModel*4))
                    g_MLPW2 = np.zeros((k_AttBlocks, k_DModel*4, k_DModel))
                    g_MLPb1 = np.zeros((k_AttBlocks, 1, k_DModel*4))
                    g_MLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
                    g_LNGain = np.zeros((k_AttBlocks, 2, k_DModel))
                    g_LNBias = np.zeros((k_AttBlocks, 2, k_DModel))
                    g_LW = np.zeros((k_DModel, k_VocabSize))
                    g_LB = np.zeros((k_VocabSize))
        a+=1



while(True):
    q = input("input_llm part of a word, a char, or something: ")
    q = list(q)
    k = len(q)
    while k < k_ContextLength:
        E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(q)
        prediction = decode(E)
        loss, onehot_cache = findLoss(E, q)
        # print(loss)
        # print(prediction)
        q.append(prediction[k])
        # print(prediction[k])
        k+=1
    print(q)