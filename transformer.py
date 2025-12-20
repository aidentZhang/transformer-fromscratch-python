import numpy as np
#HELLOOOOO ADD MASKING BEFORE SOFTMAX

#------------------
#PARAM DECLARATIONS
k_DModel = 2 #32
k_ContextLength = 3 #8
k_VocabSize = 26+4 #plus four for start, end, pad, and space
k_Attheads = 2
k_AttBlocks = 1
#these should all be the same
k_DQuery = 16
k_DKey = k_DQuery
#------------------
#SETUP DATA STRUCTS
svocabList = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
svocabDict = {}
sWe = np.random.normal(0, 0.02, size = (k_VocabSize, k_DModel))
sWe[2] = np.zeros(k_DModel)
sWpos = np.random.normal(0, 0.02, size = (k_VocabSize, k_DModel))
sWq = np.random.normal(0, np.sqrt(2/(k_DModel+k_DKey)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DKey))/np.sqrt(k_Attheads)
sWk = np.random.normal(0, np.sqrt(2/(k_DModel+k_DKey)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DKey))/np.sqrt(k_Attheads)
sWv = np.random.normal(0, np.sqrt(1/(k_DModel)), size = (k_AttBlocks, k_Attheads, k_DModel, k_DModel))/np.sqrt(k_Attheads)
sMLPW1 = np.random.normal(0, np.sqrt(2/(k_DModel+4*k_DModel)), size = (k_AttBlocks, k_DModel, k_DModel*4))
sMLPW2 = np.random.normal(0, np.sqrt(2/(k_DModel+4*k_DModel)), size = (k_AttBlocks, k_DModel*4, k_DModel))
sMLPb1 = np.zeros((k_AttBlocks, k_ContextLength, k_DModel*4))
sMLPb2 = np.zeros((k_AttBlocks, k_ContextLength, k_DModel))
sLNGain = np.ones((k_AttBlocks, 2, k_DModel)) #MULTIPLIED ELEMENT WISE
sLNBias = np.zeros((k_AttBlocks, 2, k_DModel))
sLW = np.random.normal(0, np.sqrt(2/(k_DModel+k_VocabSize)), size = (k_DModel, k_VocabSize))
sLB = np.zeros((k_VocabSize)) #ADDED TO ALL TOKENS
sSoftmaxMask = np.nan_to_num(-np.inf*np.tril(np.ones((k_ContextLength, k_ContextLength)), k=-1), nan = 0)
print(sSoftmaxMask)
#E has dimension k_ContextLength x k_DModel

#------------------
#PRE PROCESS
i = 4
for j in svocabList:
    svocabDict[j] = i
    i+=1

#------------------
#TRANSFORMER FUNCTIONS
def layerNorm(E, attLayer, prePostMLP):
    return sLNBias[attLayer, prePostMLP] + np.nan_to_num((E-np.mean(E, axis = -1, keepdims = True))/np.std(E, axis = -1, keepdims = True), nan = 0.)*(sLNGain[attLayer, prePostMLP])

def softmax(E):
    return np.exp(E)/(np.exp(E)@np.ones((E.shape[1],1)))

def relu(E):
    return np.maximum(0, E)


input = ["c"]

E = np.tile(sWe[2], (k_ContextLength, 1))
svocabDict[" "] = 3


E[0] = sWe[0]
i = 1
for j in input:
    E[i] = sWe[svocabDict[j]]
    i+=1

i = 0
while(i<len(input)+1):
    E[i]+=sWpos[i]
    i+=1

currAttBlock = 0
while(currAttBlock < k_AttBlocks):
    E = layerNorm(E, currAttBlock, 0)
    currAttHead = 0
    while(currAttHead<k_Attheads):
        E+=softmax(1/np.sqrt(k_DKey) * E@sWq[currAttBlock, currAttHead]@(E@sWk[currAttBlock, currAttHead]).T+sSoftmaxMask)@(E@sWv[currAttBlock, currAttHead])
        currAttHead+=1
    E = layerNorm(E, currAttBlock, 1)
    E += (relu(E@sMLPW1[currAttBlock]+sMLPb1[currAttBlock]))@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
    currAttBlock+=1

E=E@sLW+sLB
E=softmax(E)
print(E)