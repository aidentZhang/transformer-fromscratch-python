import numpy as np
from tqdm import tqdm # timing bar for nice looks
import params
import cupy as cp
from pathlib import Path
import time
import itertools
import string
from datasets import load_dataset
import re

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

class transformer:

    def __init__(self, weights, rule_list, vocab_list, svocabDict, train_params):
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

        self.rule_list=rule_list
        self.svocabDict=svocabDict
        self.vocab_list=vocab_list


        # Initialize gradients
        self.g_We = cp.zeros((self.k_VocabSize, self.k_DModel), dtype=cp.float32)
        self.g_Wpos = cp.zeros((self.k_ContextLength, self.k_DModel), dtype=cp.float32)
        self.g_Wq = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.g_Wk = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.g_Wv = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DModel//self.k_Attheads), dtype=cp.float32)
        self.g_Wo = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel), dtype=cp.float32)
        self.g_MLPW1 = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel*4), dtype=cp.float32)
        self.g_MLPW2 = cp.zeros((self.k_AttBlocks, self.k_DModel*4, self.k_DModel), dtype=cp.float32)
        self.g_MLPb1 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel*4), dtype=cp.float32)
        self.g_MLPb2 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel), dtype=cp.float32)
        self.g_LNGain = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.g_LNBias = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.g_LW = cp.zeros((self.k_DModel, self.k_VocabSize), dtype=cp.float32)
        self.g_LB = cp.zeros((self.k_VocabSize), dtype=cp.float32)

        self.admt_We = cp.zeros((self.k_VocabSize, self.k_DModel), dtype=cp.float32)
        self.admt_Wpos = cp.zeros((self.k_ContextLength, self.k_DModel), dtype=cp.float32)
        self.admt_Wq = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.admt_Wk = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.admt_Wv = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DModel//self.k_Attheads), dtype=cp.float32)
        self.admt_Wo = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel), dtype=cp.float32)
        self.admt_MLPW1 = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel*4), dtype=cp.float32)
        self.admt_MLPW2 = cp.zeros((self.k_AttBlocks, self.k_DModel*4, self.k_DModel), dtype=cp.float32)
        self.admt_MLPb1 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel*4), dtype=cp.float32)
        self.admt_MLPb2 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel), dtype=cp.float32)
        self.admt_LNGain = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.admt_LNBias = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.admt_LW = cp.zeros((self.k_DModel, self.k_VocabSize), dtype=cp.float32)
        self.admt_LB = cp.zeros((self.k_VocabSize), dtype=cp.float32)

        self.advt_We = cp.zeros((self.k_VocabSize, self.k_DModel), dtype=cp.float32)
        self.advt_Wpos = cp.zeros((self.k_ContextLength, self.k_DModel), dtype=cp.float32)
        self.advt_Wq = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.advt_Wk = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DKey), dtype=cp.float32)
        self.advt_Wv = cp.zeros((self.k_AttBlocks, self.k_Attheads, self.k_DModel, self.k_DModel//self.k_Attheads), dtype=cp.float32)
        self.advt_Wo = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel), dtype=cp.float32)
        self.advt_MLPW1 = cp.zeros((self.k_AttBlocks, self.k_DModel, self.k_DModel*4), dtype=cp.float32)
        self.advt_MLPW2 = cp.zeros((self.k_AttBlocks, self.k_DModel*4, self.k_DModel), dtype=cp.float32)
        self.advt_MLPb1 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel*4), dtype=cp.float32)
        self.advt_MLPb2 = cp.zeros((self.k_AttBlocks, 1, self.k_DModel), dtype=cp.float32)
        self.advt_LNGain = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.advt_LNBias = cp.zeros((self.k_AttBlocks, 2, self.k_DModel), dtype=cp.float32)
        self.advt_LW = cp.zeros((self.k_DModel, self.k_VocabSize), dtype=cp.float32)
        self.advt_LB = cp.zeros((self.k_VocabSize), dtype=cp.float32)
 


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

    def decode(self, E, svocabList):
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
    def fowardprop(self, input_llm, svocabDict, k_BatchSize):
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
        k_DQuery = self.k_DQuery
        num_times = self.num_times
        k_DKey = self.k_DKey
        k_ShiftFactor = self.k_ShiftFactor
        k_Alpha = self.k_Alpha
        k_Beta1 = self.k_Beta1
        k_Beta2 = self.k_Beta2
        k_Epsilon = self.k_Epsilon
        k_Lambda = self.k_Lambda
        k_Temp = self.k_Temp
        sMLPW1 = self.sMLPW1
        sMLPW2 = self.sMLPW2
        sMLPb1 = self.sMLPb1
        sMLPb2 = self.sMLPb2
        sLNGain = self.sLNGain
        sLNBias = self.sLNBias
        sLW = self.sLW
        sLB = self.sLB
        sSoftmaxMask = self.sSoftmaxMask
        rule_list = self.rule_list
        svocabDict = self.svocabDict

        relu=self.relu
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
        E=softmax(E)

        return E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E_cache, E_conc_cache, Q_cache, K_cache, V_cache
    #------------------
    #BACKPROP
    def backprop(self, E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache):
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
        k_DQuery = self.k_DQuery
        num_times = self.num_times
        k_DKey = self.k_DKey
        k_ShiftFactor = self.k_ShiftFactor
        k_BatchSize = len(E)
        k_Alpha = self.k_Alpha
        k_Beta1 = self.k_Beta1
        k_Beta2 = self.k_Beta2
        k_Epsilon = self.k_Epsilon
        k_Lambda = self.k_Lambda
        k_Temp = self.k_Temp
        sMLPW1 = self.sMLPW1
        sMLPW2 = self.sMLPW2
        sMLPb1 = self.sMLPb1
        sMLPb2 = self.sMLPb2
        sLNGain = self.sLNGain
        sLNBias = self.sLNBias
        sLW = self.sLW
        sLB = self.sLB
        sSoftmaxMask = self.sSoftmaxMask
        rule_list = self.rule_list
        svocabDict = self.svocabDict
        relu_deriv=self.relu_deriv
        g_We = self.g_We
        g_Wpos = self.g_Wpos
        g_Wq = self.g_Wq
        g_Wk = self.g_Wk
        g_Wv = self.g_Wv
        g_Wo = self.g_Wo
        g_MLPW1 = self.g_MLPW1
        g_MLPW2 = self.g_MLPW2
        g_MLPb1 = self.g_MLPb1
        g_MLPb2 = self.g_MLPb2
        g_LNGain = self.g_LNGain
        g_LNBias = self.g_LNBias
        g_LW = self.g_LW
        g_LB = self.g_LB



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

            G = cp.sum(G1+G2+G3, axis=1)
            # print(cp.shape(G_preatt))

            g_LNBias[currAttBlock, 0] += cp.sum(cp.sum(G, axis = 1), axis=0)
            g_LNGain[currAttBlock, 0] += cp.sum(cp.sum((E_midln_cache[currAttBlock, 0])*G, axis = 1), axis=0)

            Xhat_mean = E_midln_cache[currAttBlock, 0]*(cp.mean((G*sLNGain[currAttBlock, 0])*E_midln_cache[currAttBlock, 0], axis = 2, keepdims=True))
            G= G_preln+(1/cp.sqrt(cp.var(E_preln_cache[currAttBlock, 0], axis = 2, keepdims = True)+0.00001))*(G*sLNGain[currAttBlock, 0]-cp.mean(G*sLNGain[currAttBlock, 0], axis = 2, keepdims=True)-Xhat_mean)
            currAttBlock-=1

        g_Wpos+=cp.sum(G, axis=0)
        # print(cp.shape(We_to_E))
        g_We += cp.sum(cp.transpose(We_to_E, [0, 2, 1])@G, axis=0)*cp.sqrt(k_DModel)
        g_We[2] = cp.zeros(k_DModel)

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
    

    def train(self, startind, num_steps, save_duration):
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
        k_DQuery = self.k_DQuery
        num_times = self.num_times
        k_DKey = self.k_DKey
        shift_factor = self.k_ShiftFactor
        k_BatchSize = self.k_BatchSize
        k_Alpha = self.k_Alpha
        k_Beta1 = self.k_Beta1
        k_Beta2 = self.k_Beta2
        k_Epsilon = self.k_Epsilon
        k_Lambda = self.k_Lambda
        k_Temp = self.k_Temp
        sMLPW1 = self.sMLPW1
        sMLPW2 = self.sMLPW2
        sMLPb1 = self.sMLPb1
        sMLPb2 = self.sMLPb2
        sLNGain = self.sLNGain
        sLNBias = self.sLNBias
        sLW = self.sLW
        sLB = self.sLB
        sSoftmaxMask = self.sSoftmaxMask
        rule_list = self.rule_list
        svocabDict = self.svocabDict
        relu_deriv=self.relu_deriv
        g_We = self.g_We
        g_Wpos = self.g_Wpos
        g_Wq = self.g_Wq
        g_Wk = self.g_Wk
        g_Wv = self.g_Wv
        g_Wo = self.g_Wo
        g_MLPW1 = self.g_MLPW1
        g_MLPW2 = self.g_MLPW2
        g_MLPb1 = self.g_MLPb1
        g_MLPb2 = self.g_MLPb2
        g_LNGain = self.g_LNGain
        g_LNBias = self.g_LNBias
        g_LW = self.g_LW
        g_LB = self.g_LB
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
        )

        embed =self.embed
        findLoss = self.findLoss
        backprop = self.backprop
        fowardprop = self.fowardprop
        i = 0
        t=0
        token_stream=[]
        BYTE_LOOKUP = [f"{i:02x}" for i in range(256)]
        with open('results.txt', 'w', encoding="utf-8") as f:
            with tqdm(total=num_steps) as pbar:
                for text in ds['text']:
                    
                    amnt = 0
                    
                    if i == startind+num_steps:
                        break
                    i+=1

                    if(i < startind):
                        continue

                    if(i%save_duration==0):
                        cp.savez(f"./Weights/weights{i}.npz", sWe=sWe, sWpos=sWpos, sWq=sWq, sWk=sWk, sWv=sWv, sMLPW1=sMLPW1, sMLPW2=sMLPW2, sMLPb1=sMLPb1, sMLPb2=sMLPb2, sLNGain=sLNGain, sLNBias=sLNBias, sLW=sLW, sLB=sLB, sWo = sWo)

                    # print("tokenizing!")
                    search_st = time.perf_counter()

                    token_stream+= embed(svocabDict, text, BYTE_LOOKUP)+["<EOS>"]
                    search_et = time.perf_counter()
                    # print(f"embedding took {search_et-search_st:.4f} seconds.")

                    # print(text)
                    pbar.update(1)


                    if(i%10==0):
                        curr_start =   0
                        input_batch = []
                        text=token_stream

                        while(curr_start<=len(text)-k_ContextLength):
                            search_st = time.perf_counter()

                            word = text[curr_start:(curr_start+k_ContextLength-1)]
                            curr_start+=int(k_ContextLength/shift_factor)
                            if(len(word)<k_ContextLength):
                                amnt+=1
                                input_batch.append(word)


                            if(amnt%k_BatchSize == 0 or curr_start+2*int(k_ContextLength/shift_factor) > len(text)):   
                                E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache = fowardprop(input_batch, svocabDict, len(input_batch))
                                # prediction = decode(E, vocab_list)
                                # print(prediction)
                                loss, onehot_cache = findLoss(E, input_batch, svocabDict)
                                backprop(E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, onehot_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, Q_cache, K_cache, V_cache)
                                loss = cp.array(loss)
                                t+=1
                                f.write(f"{cp.mean(loss)}\n")
                                avgloss=0
                                g_LB/=len(input_batch)
                                g_LW/=len(input_batch)
                                g_MLPb2/=len(input_batch)
                                g_MLPW2/=len(input_batch)
                                g_MLPb1/=len(input_batch)
                                g_MLPW1/=len(input_batch)
                                g_LNBias/=len(input_batch)
                                g_LNGain/=len(input_batch)
                                g_Wv/=len(input_batch)
                                g_Wq/=len(input_batch)
                                g_Wk/=len(input_batch)
                                g_Wo/=len(input_batch)
                                g_Wpos/=len(input_batch)
                                g_We/=len(input_batch)
                                input_batch = []

                                self.admt_We = k_Beta1*self.admt_We + (1-k_Beta1)*g_We
                                self.admt_Wpos = k_Beta1*self.admt_Wpos + (1-k_Beta1)*g_Wpos
                                self.admt_Wq = k_Beta1*self.admt_Wq + (1-k_Beta1)*g_Wq
                                self.admt_Wk = k_Beta1*self.admt_Wk + (1-k_Beta1)*g_Wk
                                self.admt_Wv = k_Beta1*self.admt_Wv + (1-k_Beta1)*g_Wv
                                self.admt_Wo = k_Beta1*self.admt_Wo + (1-k_Beta1)*g_Wo
                                self.admt_MLPW1 = k_Beta1*self.admt_MLPW1 + (1-k_Beta1)*g_MLPW1
                                self.admt_MLPW2 = k_Beta1*self.admt_MLPW2 + (1-k_Beta1)*g_MLPW2
                                self.admt_MLPb1 = k_Beta1*self.admt_MLPb1 + (1-k_Beta1)*g_MLPb1
                                self.admt_MLPb2 = k_Beta1*self.admt_MLPb2 + (1-k_Beta1)*g_MLPb2
                                self.admt_LNGain = k_Beta1*self.admt_LNGain + (1-k_Beta1)*g_LNGain
                                self.admt_LNBias = k_Beta1*self.admt_LNBias + (1-k_Beta1)*g_LNBias
                                self.admt_LW = k_Beta1*self.admt_LW + (1-k_Beta1)*g_LW
                                self.admt_LB = k_Beta1*self.admt_LB + (1-k_Beta1)*g_LB


                                self.advt_We = k_Beta2*self.advt_We + (1-k_Beta2)*cp.square(g_We)
                                self.advt_Wpos = k_Beta2*self.advt_Wpos + (1-k_Beta2)*cp.square(g_Wpos)
                                self.advt_Wq = k_Beta2*self.advt_Wq + (1-k_Beta2)*cp.square(g_Wq)
                                self.advt_Wk = k_Beta2*self.advt_Wk + (1-k_Beta2)*cp.square(g_Wk)
                                self.advt_Wv = k_Beta2*self.advt_Wv + (1-k_Beta2)*cp.square(g_Wv)
                                self.advt_Wo = k_Beta2*self.advt_Wo + (1-k_Beta2)*cp.square(g_Wo)
                                self.advt_MLPW1 = k_Beta2*self.advt_MLPW1 + (1-k_Beta2)*cp.square(g_MLPW1)
                                self.advt_MLPW2 = k_Beta2*self.advt_MLPW2 + (1-k_Beta2)*cp.square(g_MLPW2)
                                self.advt_MLPb1 = k_Beta2*self.advt_MLPb1 + (1-k_Beta2)*cp.square(g_MLPb1)
                                self.advt_MLPb2 = k_Beta2*self.advt_MLPb2 + (1-k_Beta2)*cp.square(g_MLPb2)
                                self.advt_LNGain = k_Beta2*self.advt_LNGain + (1-k_Beta2)*cp.square(g_LNGain)
                                self.advt_LNBias = k_Beta2*self.advt_LNBias + (1-k_Beta2)*cp.square(g_LNBias)
                                self.advt_LW = k_Beta2*self.advt_LW + (1-k_Beta2)*cp.square(g_LW)
                                self.advt_LB = k_Beta2*self.advt_LB + (1-k_Beta2)*cp.square(g_LB)

                                sWe -= k_Alpha*(((self.admt_We/(1-k_Beta1**t))/(cp.sqrt(self.advt_We/(1-k_Beta2**t))+k_Epsilon))+sWe*k_Lambda)
                                sWpos -= k_Alpha*(((self.admt_Wpos/(1-k_Beta1**t))/(cp.sqrt(self.advt_Wpos/(1-k_Beta2**t))+k_Epsilon))+sWpos*k_Lambda)
                                sWq -= k_Alpha*(((self.admt_Wq/(1-k_Beta1**t))/(cp.sqrt(self.advt_Wq/(1-k_Beta2**t))+k_Epsilon))+sWq*k_Lambda)
                                sWk -= k_Alpha*(((self.admt_Wk/(1-k_Beta1**t))/(cp.sqrt(self.advt_Wk/(1-k_Beta2**t))+k_Epsilon))+sWk*k_Lambda)
                                sWv -= k_Alpha*(((self.admt_Wv/(1-k_Beta1**t))/(cp.sqrt(self.advt_Wv/(1-k_Beta2**t))+k_Epsilon))+sWv*k_Lambda)
                                sWo -= k_Alpha*(((self.admt_Wo/(1-k_Beta1**t))/(cp.sqrt(self.advt_Wo/(1-k_Beta2**t))+k_Epsilon))+sWo*k_Lambda)
                                sMLPW1 -= k_Alpha*(((self.admt_MLPW1/(1-k_Beta1**t))/(cp.sqrt(self.advt_MLPW1/(1-k_Beta2**t))+k_Epsilon))+sMLPW1*k_Lambda)
                                sMLPW2-= k_Alpha*(((self.admt_MLPW2/(1-k_Beta1**t))/(cp.sqrt(self.advt_MLPW2/(1-k_Beta2**t))+k_Epsilon))+sMLPW2*k_Lambda)
                                sMLPb1 -= k_Alpha*(((self.admt_MLPb1/(1-k_Beta1**t))/(cp.sqrt(self.advt_MLPb1/(1-k_Beta2**t))+k_Epsilon))+sMLPb1*k_Lambda)
                                sMLPb2 -= k_Alpha*(((self.admt_MLPb2/(1-k_Beta1**t))/(cp.sqrt(self.advt_MLPb2/(1-k_Beta2**t))+k_Epsilon))+sMLPb2*k_Lambda)
                                sLNGain -= k_Alpha*(((self.admt_LNGain/(1-k_Beta1**t))/(cp.sqrt(self.advt_LNGain/(1-k_Beta2**t))+k_Epsilon))+sLNGain*k_Lambda)
                                sLNBias-= k_Alpha*(((self.admt_LNBias/(1-k_Beta1**t))/(cp.sqrt(self.advt_LNBias/(1-k_Beta2**t))+k_Epsilon))+sLNBias*k_Lambda)
                                sLW -= k_Alpha*(((self.admt_LW/(1-k_Beta1**t))/(cp.sqrt(self.advt_LW/(1-k_Beta2**t))+k_Epsilon))+sLW*k_Lambda)
                                sLB -= k_Alpha*(((self.admt_LB/(1-k_Beta1**t))/(cp.sqrt(self.advt_LB/(1-k_Beta2**t))+k_Epsilon))+sLB*k_Lambda)
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
                        token_stream=token_stream[curr_start:]




        cp.savez("./Weights/weights.npz", sWe=sWe, sWpos=sWpos, sWq=sWq, sWk=sWk, sWv=sWv, sMLPW1=sMLPW1, sMLPW2=sMLPW2, sMLPb1=sMLPb1, sMLPb2=sMLPb2, sLNGain=sLNGain, sLNBias=sLNBias, sLW=sLW, sLB=sLB, sWo = sWo)

    def is_hex(self, s):
        return all(c in string.hexdigits for c in s)

    def run_model(self):
        embed=self.embed
        svocabDict=self.svocabDict
        k_ContextLength=self.k_ContextLength
        decode=self.decode
        vocab_list=self.vocab_list
        is_hex=self.is_hex
        fowardprop=self.fowardprop
        BYTE_LOOKUP = [f"{i:02x}" for i in range(256)]

        while(True):
            q = input("input_llm part of a word, a char, or something: ")
            q = embed(svocabDict, q, BYTE_LOOKUP)
            k = len(q)
            print(q) 
            while k < k_ContextLength:
                E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E, E_conc_cache, temp1, temp2, temp3= fowardprop([q], svocabDict, 1)
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

                print(final_text, end='', flush=True)
                k+=1
            print("")
            # print(q)



