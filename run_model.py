import cupy as cp

import params

weights = cp.load("./Weights/weights1500.npz")



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
sWo = weights["sWo"]
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
    temp = cp.nan_to_num((E-cp.mean(E, axis = -1, keepdims = True))/cp.sqrt((cp.nan_to_num(cp.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001)))
    return sLNBias[attLayer, prePostMLP] + temp * (sLNGain[attLayer, prePostMLP]), temp

def softmax(E):
    exp_ = cp.exp(E-cp.max(E, axis=-1, keepdims=True))
    return cp.nan_to_num(exp_/(cp.sum(exp_, axis=-1, keepdims=True)))

print(softmax(cp.array([[[1.3, 5.1, 2.2, 0.7, 1.1]]])))


def relu(E):
    return cp.maximum(0, E)

def relu_deriv(E):
    return cp.minimum(1, E)


#GEMINI WROTE THIS:
def decode(E, svocabList, top_k=5):
    # E is shape (k_ContextLength, k_VocabSize) and already contains probabilities
    temp = cp.zeros(E.shape[0], dtype=cp.int32)
    
    # We must sample row by row because cp.random.choice requires a 1D probability array
    for row_idx in range(E.shape[0]):
        row_probs = E[row_idx]
        
        # 1. Get the indices of the top K probabilities
        top_k_indices = cp.argsort(row_probs)[-top_k:]
        
        # 2. Create a blank mask of zeros
        masked_probs = cp.zeros_like(row_probs)
        
        # 3. Copy only the top K probabilities into the mask
        masked_probs[top_k_indices] = row_probs[top_k_indices]
        
        # 4. Re-normalize the probabilities so they sum to 1.0
        sum_probs = cp.sum(masked_probs)
        if sum_probs > 0:
            masked_probs = masked_probs / sum_probs
        else:
            # Fallback just in case of rounding errors
            masked_probs[top_k_indices] = 1.0 / top_k 
            
        # 5. Sample the token based on the filtered probabilities (added size=1)[0]
        temp[row_idx] = cp.random.choice(len(masked_probs), size=1, p=masked_probs)[0]

    # 6. Map the sampled IDs back to your vocabulary strings
    answer = []
    for i in temp:
        i = int(i)
        if (i > 3):
            if i - 4 >= len(svocabList):
                answer.append("<NA>")
            else:
                answer.append(svocabList[i-4])
        elif (i == 3):
            answer.append(" ")
        elif (i == 1):
            answer.append("<END>")
        elif (i == 2):
            answer.append("<PAD>")
        else:
            answer.append("<STA>")
            
    return answer


#------------------------------------------------------------------------------------------------------------------------------FINISH
def findLoss(E, input_llm, svocabDict):
    loss = 0
    i = 0
    onehot_cache = cp.zeros((k_ContextLength, k_VocabSize))

    while i < len(input_llm):
        if input_llm[i] in svocabDict:
            onehot_cache[i, svocabDict[input_llm[i]]] = 1
            loss += -cp.log(E[i][svocabDict[input_llm[i]]]+0.00001)
        else:
            onehot_cache[i, k_VocabSize-1] = 1
            loss += -cp.log(E[i][k_VocabSize-1]+0.00001)

        i+=1
    loss = (loss-cp.log(E[len(input_llm)][1]+0.001))/len(input_llm)
    onehot_cache[len(input_llm), 1] = 1
    return loss, onehot_cache







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
    sSoftmaxMask = cp.nan_to_num(-cp.inf * cp.triu(cp.ones((k_ContextLength, k_ContextLength), dtype=cp.float32), k=1), nan=0.0)

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

        E[i] = We_to_E@sWe*cp.sqrt(k_DModel)
        We_to_E_cache[i]=We_to_E
        E[i]+=sWpos


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
    E=softmax(E/0.8)

    return E

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


num_times = params.num_times


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
import re
import time
import string

def is_hex(s):
    return all(c in string.hexdigits for c in s)

import string
def is_hex(s):
    return all(c in string.hexdigits for c in s)


k_BatchSize=1
# while(True):
#     q = input("input_llm part of a word, a char, or something: ")
#     q = embed(svocabDict, q)
#     k = len(q)
#     print(q) 
#     while k < k_ContextLength:
#         E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, temp1, temp2, temp3= fowardprop(q, svocabDict)
#         prediction = decode(E[0], vocab_list)
#         # loss, onehot_cache = findLoss(E, q, svocabDict)
#         # print(loss)
#         # print(prediction)
#         q.append(prediction[k])
#         text = prediction[k].split("</w>")

#         post_processed = [
#             bytes.fromhex(word).decode("utf-8") if is_hex(word) else word
#             for word in text
#         ]

#         final_text = " ".join(post_processed)
        
#         print(final_text, end='')
#         k+=1
#     print("")
#     # print(q)
#     # print(q)


while(True):
    q = input("input_llm part of a word, a char, or something: ")
    q = embed(svocabDict, q)
    k = len(q)
    print(q)
    while k < k_ContextLength:
        print("", end='')
        E = fowardprop([q], svocabDict)
        prediction = decode(E[0], vocab_list)
        # loss, onehot_cache = findLoss(E, q, svocabDict)
        # print(loss)
        # print(prediction)
        q.append(prediction[k])

        text = prediction[k].split("</w>")

        post_processed = [
            bytes.fromhex(word).decode("utf-8", errors="replace") if is_hex(word) else word
            for word in text
        ]

        final_text = " ".join(post_processed)

        print(final_text, end='')
        k+=1

    print("")
    # print(q)