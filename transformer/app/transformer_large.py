import numpy as np
import numpy as cp
from pathlib import Path
import time
import itertools
import string
import re


class transformer_inf_large:
    def __init__(self, weights, vocab_list, svocabDict, train_params):
        #PARAM DECLARATIONS
        self.sWe = weights["sWe"]
        self.sWpos = weights["sWpos"]
        self.sWq = weights["sWq"]
        self.sWk = weights["sWk"]
        self.sWv = weights["sWv"]
        self.sMLPW1 = weights["sMLPW1"]
        self.sMLPW2 = weights["sMLPW2"]
        self.sMLPb1 = weights["sMLPb1"]
        self.sMLPb2 = weights["sMLPb2"]
        self.sLNGain = weights["sLNGain"]
        self.sLNBias = weights["sLNBias"]
        self.sLW = weights["sLW"]
        self.sLB = weights["sLB"]
        self.sWo = weights["sWo"]
        self.k_DModel = train_params["k_DModel"]
        self.k_ContextLength = train_params["k_ContextLength"]
        self.k_VocabSize = train_params["k_VocabSize"]  # plus 5 for special tokens
        self.k_Attheads = train_params["k_Attheads"]
        self.k_AttBlocks = train_params["k_AttBlocks"]
        self.k_DQuery = train_params["k_DQuery"]
        self.num_times = train_params["num_times"]
        self.k_DKey = self.k_DQuery
        self.k_ShiftFactor = train_params["k_ShiftFactor"]
        self.k_BatchSize = train_params["k_BatchSize"]
        self.k_Alpha = train_params["k_Alpha"]
        self.k_Beta1 = train_params["k_Beta1"]
        self.k_Beta2 = train_params["k_Beta2"]
        self.k_Epsilon = train_params["k_Epsilon"]
        self.k_Lambda = train_params["k_Lambda"]
        self.k_Temp = train_params["k_Temp"]

        self.sSoftmaxMask = cp.nan_to_num(-cp.inf * cp.triu(cp.ones((self.k_ContextLength, self.k_ContextLength), dtype=cp.float32), k=1), nan=0.0)

        self.svocabDict=svocabDict
        self.vocab_list=vocab_list
 


    def layerNorm(self, E, attLayer, prePostMLP):
        temp = (E-cp.mean(E, axis = -1, keepdims = True))/cp.sqrt((cp.nan_to_num(cp.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001))
        return self.sLNBias[attLayer, prePostMLP] + temp * (self.sLNGain[attLayer, prePostMLP]), temp


    def softmax(self, E):
        exp_ = cp.exp(E-cp.max(E, axis=-1, keepdims=True))
        return cp.nan_to_num(exp_/(cp.sum(exp_, axis=-1, keepdims=True)))


    def relu(self, E):
        return cp.maximum(0, E)

    def relu_deriv(self, E):
        return (E > 0).astype(cp.float32)

    #GEMINI WROTE THIS:
    def decode(self, E, svocabList, top_k=1):
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
    def findLoss(self, E, input_llm, svocabDict):
        k_BatchSize=len(input_llm)
        k_ContextLength= self.k_ContextLength
        k_VocabSize=self.k_VocabSize
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
    def fowardprop(self, input_llm, svocabDict, k_BatchSize, k_temp):
        sWe = self.sWe
        sWpos = self.sWpos
        sWq = self.sWq
        sWk = self.sWk
        sWv = self.sWv
        sWo = self.sWo
        k_DModel = self.k_DModel
        k_ContextLength = self.k_ContextLength
        k_VocabSize = self.k_VocabSize
        k_Attheads = self.k_Attheads
        k_AttBlocks = self.k_AttBlocks

        k_DKey = self.k_DKey

        sMLPW1 = self.sMLPW1
        sMLPW2 = self.sMLPW2
        sMLPb1 = self.sMLPb1
        sMLPb2 = self.sMLPb2
        sLW = self.sLW
        sLB = self.sLB
        sSoftmaxMask = self.sSoftmaxMask
        svocabDict = self.svocabDict

        relu=self.relu
        #------------------
        #CACHEING

        
        layerNorm=self.layerNorm
        softmax=self.softmax
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
                    We_to_E[o, svocabDict[j]] = 1
                    #used to have [] wrapped around svocabDict, changed because it was uncessary
                else:
                    We_to_E[o, k_VocabSize-1] = 1
                o+=1

            while(o<k_ContextLength):
                We_to_E[o, 2] = 1
                o+=1
            E[i] = We_to_E@sWe*cp.sqrt(k_DModel)
            E[i]+=sWpos


        currAttBlock = 0
        while(currAttBlock < k_AttBlocks):
            E_ln, *_ = layerNorm(E, currAttBlock, 0)

            start = time.perf_counter()
            Q=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWq[currAttBlock], [1, 0, 2]), [k_DModel, k_DKey*k_Attheads]), [k_BatchSize, k_ContextLength, k_Attheads, k_DKey]), [0, 2, 1, 3])
            K=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWk[currAttBlock], [1, 0, 2]), [k_DModel, k_DKey*k_Attheads]), [k_BatchSize, k_ContextLength, k_Attheads, k_DKey]), [0, 2, 1, 3])
            V=cp.transpose(cp.reshape(E_ln@cp.reshape(cp.transpose(sWv[currAttBlock], [1, 0, 2]), [k_DModel, k_DModel]), [k_BatchSize, k_ContextLength, k_Attheads, k_DModel//k_Attheads]), [0, 2, 1, 3])
            end = time.perf_counter()
            print(f"QKV calculation time for block {currAttBlock}: {end - start:.4f} seconds")

            # print(cp.shape(cp.reshape(cp.transpose(E_soft_cache[currAttBlock]@(V), [0, 2, 1, 3]), [k_BatchSize, k_ContextLength, k_DModel])))
            E+= cp.reshape(cp.transpose(softmax(1/cp.sqrt(k_DKey) * Q@cp.transpose(K, [0, 1, 3, 2])+sSoftmaxMask+cp.expand_dims(padMask, axis=1))@(V), [0, 2, 1, 3]), [k_BatchSize, k_ContextLength, k_DModel])@sWo[currAttBlock]

            E_ln, *_ = layerNorm(E, currAttBlock, 1)
            E += relu(E_ln@sMLPW1[currAttBlock]+sMLPb1[currAttBlock])@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
            currAttBlock+=1
        
        E=E@sLW+sLB
        E=softmax(E/k_temp)

        return E
    #------------------

    #---------------------------------
    #Data processing functions
    def embed(self, svocabDict, case, BYTE_LOOKUP):
        embeded=[]

        words = re.findall(r"\w+|[^\w\s]|\n", case)


        processed_text = [
            [BYTE_LOOKUP[b] for b in word.encode("utf-8")] + ["</w>"]
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
    

    def is_hex(self, s):
        return all(c in string.hexdigits for c in s)

    def run_model(self, q):
        embed=self.embed
        svocabDict=self.svocabDict
        k_ContextLength=self.k_ContextLength
        decode=self.decode
        vocab_list=self.vocab_list
        is_hex=self.is_hex
        fowardprop=self.fowardprop
        BYTE_LOOKUP = [f"{i:02x}" for i in range(256)]
        self.k_cache = np.zeros(())
        q = embed(svocabDict, q, BYTE_LOOKUP)
        k = len(q)
        # print(q) 
        while k < k_ContextLength:
            start = time.perf_counter()

            E = fowardprop([q], svocabDict, 1, 0.7)
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
            if(final_text=="<EOS>"):
                break
            yield(final_text)
            k+=1

            end = time.perf_counter()
            print(f"Execution time: {end - start:.4f} seconds")

