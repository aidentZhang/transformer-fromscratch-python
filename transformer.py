import numpy as np
#------------------
#PARAM DECLARATIONS
k_DModel = 8 #32
k_ContextLength = 4#8
k_VocabSize = 26+4 #plus four for start, end, pad, and space
k_Attheads = 2
k_AttBlocks = 1
#these should all be the same
k_DQuery = 4
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
sMLPb1 = np.ones((k_AttBlocks, 1, k_DModel*4))
sMLPb2 = np.zeros((k_AttBlocks, 1, k_DModel))
sLNGain = np.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
sLNBias = np.zeros((k_AttBlocks, 2, k_DModel))
sLW = np.random.normal(0, np.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, k_VocabSize))
sLB = np.zeros((1,k_VocabSize)) #ADDED TO ALL TOKENS
sSoftmaxMask = np.nan_to_num(-np.inf*np.tril(np.ones((k_ContextLength, k_ContextLength)), k=-1), nan = 0)
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
    return np.exp(E)/(np.exp(E)@np.ones((E.shape[1],1)))

def relu(E):
    return np.maximum(0, E)

def relu_deriv_elementwise(e):
    return 1 if e>0 else 0

def decode(E):
    temp = np.argmax(E, axis = -1)
    answer = []
    for i in temp:
        if(i > 3):
            answer.append(svocabList[i-4])
        elif (i==3):
            answer.append(" ")
        elif (i==1):
            answer.append("END_TOKEN")
        elif (i==2):
            answer.append("PAD_TOKEN")
        else:
            answer.append("START_TOKEN")
    return answer
#------------------------------------------------------------------------------------------------------------------------------FINISH
def findLoss(E, input):
    loss = 0
    i = 0
    onehot_cache = np.zeros((k_ContextLength, k_VocabSize))

    while i < len(input):
        onehot_cache[i, svocabDict[input[i]]] = 1
        loss += -np.log(E[i][svocabDict[input[i]]])
        i+=1
    loss = (loss-np.log(E[len(input)][1]))/len(input)
    onehot_cache[len(input), 1] = 1
    return loss, onehot_cache
#------------------
#INPUT
input = ["c", "a", "t"]
#------------------
#FORWARD PROPAGATE
def fowardprop(input):
    #------------------
    #CACHEING
    E_preln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_midln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_postln_cache = np.empty((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_soft_cache = np.empty((k_AttBlocks, k_Attheads, k_ContextLength, k_ContextLength))
    E_lin_cache = np.empty((k_ContextLength, k_DModel))
    E_relu_cache = np.empty((k_AttBlocks, k_ContextLength, k_DModel*4))
    svocabDict[" "] = 3

    temp = np.zeros(k_VocabSize)
    temp[2] = 1
    We_to_E = np.zeros((k_ContextLength, k_VocabSize))
    We_to_E[0, 0] = 1
    i = 1
    for j in input:
        We_to_E[i, [svocabDict[j]]] = 1
        i+=1
    while(i<k_ContextLength):
        We_to_E[i, 2] = 1
        i+=1
    E = We_to_E@sWe

    i = 0
    while(i<len(input)+1):
        E[i]+=sWpos[i]
        i+=1

    currAttBlock = 0
    while(currAttBlock < k_AttBlocks):
        E_preln_cache[currAttBlock, 0] = E.copy()
        E, E_midln_cache[currAttBlock, 0] = layerNorm(E, currAttBlock, 0)
        E_postln_cache[currAttBlock, 0] = E.copy()
        currAttHead = 0
        while(currAttHead<k_Attheads):
            E_soft_cache[currAttBlock, currAttHead] = softmax(1/np.sqrt(k_DKey) * E@sWq[currAttBlock, currAttHead]@(E@sWk[currAttBlock, currAttHead]).T+sSoftmaxMask)
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

    g_LW-=E_lin_cache.T@(E-onehot_cache)
    g_LB-=np.sum((E-onehot_cache), axis=0)
    
    G = (E-onehot_cache)@sLW.T
    currAttBlock = k_AttBlocks-1
    while(currAttBlock>=0):
        g_MLPb2[currAttBlock]-=np.sum(G, axis=0)
        g_MLPW2[currAttBlock]-=E_relu_cache[currAttBlock].T@G
        g_MLPb1[currAttBlock]-=np.sum(((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))[0], axis=0)
        g_MLPW1[currAttBlock]-=(E_postln_cache[currAttBlock, 1].T) @ ((G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock]))
        G = (G@(sMLPW2[currAttBlock].T))*relu_deriv(E_relu_cache[currAttBlock])@sMLPW1[currAttBlock].T
        g_LNBias[currAttBlock, 1] -= np.sum(G, axis = 0)
        g_LNGain[currAttBlock, 1] -= np.sum((E_midln_cache[currAttBlock, 1])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 1]*(np.mean((G*sLNGain[currAttBlock, 1])*E_midln_cache[currAttBlock, 1], axis = 1, keepdims=True))
        G= (1/np.sqrt(np.var(E_preln_cache[currAttBlock, 1], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 1]-np.mean(G*sLNGain[currAttBlock, 1], axis = 1, keepdims=True)-Xhat_mean)
        currAttHead = 0
        G_preatt = np.zeros((k_ContextLength, k_DModel))
        while(currAttHead<k_Attheads):
            g_Wv[currAttBlock, currAttHead]-=E_postln_cache[currAttBlock, 0].T@(E_soft_cache[currAttBlock, currAttHead].T@G)
            V = E_postln_cache[currAttBlock, 0]@sWv[currAttBlock, currAttHead]
            d_softmax = (E_soft_cache[currAttBlock, currAttHead])*(G@V.T-np.sum(E_soft_cache[currAttBlock, currAttHead]*(G@V.T), axis = 1, keepdims = True))
            g_Wq -= (1/np.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax@E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead]
            g_Wk -= (1/np.sqrt(k_DKey))*E_postln_cache[currAttBlock, 0].T@d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]
            G1 =  E_soft_cache[currAttBlock, currAttHead].T@G@sWv[currAttBlock, currAttHead].T
            G2 = (1/np.sqrt(k_DKey))*d_softmax@(E_postln_cache[currAttBlock, 0]@sWk[currAttBlock, currAttHead])@sWq[currAttBlock, currAttHead].T
            G3 = (1/np.sqrt(k_DKey))*d_softmax.T@E_postln_cache[currAttBlock, 0]@sWq[currAttBlock, currAttHead]@sWk[currAttBlock, currAttHead].T
            G_preatt += G1+G2+G3
            currAttHead+=1
        G=G_preatt
        g_LNBias[currAttBlock, 0] -= np.sum(G, axis = 0)
        g_LNGain[currAttBlock, 0] -= np.sum((E_midln_cache[currAttBlock, 0])*G, axis = 0)
        Xhat_mean = E_midln_cache[currAttBlock, 0]*(np.mean((G*sLNGain[currAttBlock, 0])*E_midln_cache[currAttBlock, 0], axis = 1, keepdims=True))
        G= (1/np.sqrt(np.var(E_preln_cache[currAttBlock, 0], axis = 1, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 0]-np.mean(G*sLNGain[currAttBlock, 0], axis = 1, keepdims=True)-Xhat_mean)
        currAttBlock-=1
    g_Wpos-=G
    g_We -= We_to_E.T@G



lr = 0.001

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
i = 0
while(i<100):
    E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(input)
    prediction = decode(E)
    print(prediction)
    loss, onehot_cache = findLoss(E, input)
    print(loss)
    backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E)
    sLB+=g_LB*lr
    sLW+=g_LW*lr
    sMLPb2+=g_MLPb2*lr
    sMLPW2+=g_MLPW2*lr
    sMLPb1+=g_MLPb1*lr
    sMLPW1+=g_MLPW1*lr
    sLNBias+=g_LNBias*lr
    sLNGain+=g_LNGain*lr
    sWv+=g_Wv*lr
    sWq+=g_Wq*lr
    sWk+=g_Wk*lr
    sWpos+=g_Wpos*lr
    sWe+=g_We*lr
    i+=1

print("------")
testing = ["c"]
E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(testing)
prediction = decode(E)
print(prediction)
testing.append(prediction[1])

E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E = fowardprop(testing)
prediction = decode(E)
print(prediction)