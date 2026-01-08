import mlx.core as mx
import params

weights = mx.load("./Weights/weights.npz")



sWe = weights["sWe"]
sWpos = weights["sWpos"]
sWq = weights["sWq"]
sWk = weights["sWk"]
sWv = weights["sWv"]
sMLPW1 = weights["sMLPW1"]
sMLPW2 = weights["sMLPW2"]
sMLPb1 = weights["sMLPb1"]
sMLPb2 = weights["sMLPb2"]
sLNGain = weights["sLNGain"]
sLNBias = weights["sLNBias"]
sLW = weights["sLW"]
sLB = weights["sLB"]

k_VocabSize=len(sWe)
k_DModel=len(sWe[0])
k_AttBlocks=len(sWk)
k_Attheads=len(sWk[0])
k_ContextLength=len(sWpos)
k_Dquery=len(sWk[0][0][0])
k_DKey = k_Dquery

print(k_VocabSize)
print(k_DModel)
print(k_ContextLength)
print(k_DKey)
print(k_AttBlocks)
print(k_Attheads)
print(k_Dquery)


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


def embed(rule_list, case):
    i = 0
    while(i < len(rule_list)):
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






def fowardprop(input_llm, svocabDict):
    #------------------
    #CACHEING
    E_preln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_midln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_postln_cache = mx.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
    E_soft_cache = mx.zeros((k_AttBlocks, k_Attheads, k_ContextLength, k_ContextLength))
    E_lin_cache = mx.zeros((k_ContextLength, k_DModel))
    E_relu_cache = mx.zeros((k_AttBlocks, k_ContextLength, k_DModel*4))
    
    sSoftmaxMask =  mx.nan_to_num(-mx.inf*mx.triu(mx.ones((k_ContextLength, k_ContextLength)), k=1), nan = 0)

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


num_times = params.num_times

with open('bpe_rules.txt', 'r') as f:
    rule_list = []
    i = 0
    while(i < num_times):
        rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
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



with open('bpe_vocablist.txt', 'r') as f:
    for line in f:
        vocab_list.append(line[:-1].replace('\\n', '\n'))
        svocabDict[vocab_list[-1]] = i
        i+=1
amnt = 0
loss = 0



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