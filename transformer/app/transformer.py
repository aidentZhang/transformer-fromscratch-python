import numpy as np
class transformer():
    def __init__(self, weights, rule_list, vocab_list, svocabDict, num_times):
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

        self.k_VocabSize=len(self.sWe)
        self.k_DModel=len(self.sWe[0])
        self.k_AttBlocks=len(self.sWk)
        self.k_Attheads=len(self.sWk[0])
        self.k_ContextLength=len(self.sWpos)
        self.k_Dquery=len(self.sWk[0][0][0])
        self.k_DKey = self.k_Dquery

        self.rule_list=rule_list
        self.svocabDict=svocabDict
        self.num_times=num_times


    def layerNorm(self, E, attLayer, prePostMLP):
        sLNBias = self.sLNBias
        sLNGain = self.sLNGain
        temp = np.nan_to_num((E-np.mean(E, axis = -1, keepdims = True))/np.sqrt((np.nan_to_num(np.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001)))
        return sLNBias[attLayer, prePostMLP] + temp * (sLNGain[attLayer, prePostMLP]), temp

    def softmax(self, E):
        exp_ = np.exp(E-np.max(E, axis=-1, keepdims=True))
        return exp_/(np.sum(exp_, axis=-1, keepdims=True))

    def relu(self, E):
        return np.maximum(0, E)

    def relu_deriv(self, E):
        return np.minimum(1, E)

    def decode(self, E, svocabList):
        temp = np.argmax(E, axis = -1)
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
    def embed(self, rule_list, case):
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

    def fowardprop(self, input_llm, svocabDict):
        layerNorm=self.layerNorm
        softmax=self.softmax
        relu=self.relu
        k_AttBlocks = self.k_AttBlocks
        k_ContextLength=self.k_ContextLength
        k_DModel=self.k_DModel
        k_Attheads=self.k_Attheads
        k_VocabSize=self.k_VocabSize
        num_times = self.num_times
        k_DKey = self.k_DKey
        sWe=self.sWe
        sWpos=self.sWpos
        sWq=self.sWq
        sWk=self.sWk
        sWv=self.sWv
        sMLPW1=self.sMLPW1
        sMLPW2=self.sMLPW2
        sMLPb1=self.sMLPb1
        sMLPb2=self.sMLPb2
        sLNGain=self.sLNGain
        sLNBias=self.sLNBias
        sLW=self.sLW
        sLB=self.sLB
        #------------------
        #CACHEING
        E_preln_cache = np.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
        E_midln_cache = np.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
        E_postln_cache = np.zeros((k_AttBlocks, 2, k_ContextLength, k_DModel))
        E_soft_cache = np.zeros((k_AttBlocks, k_Attheads, k_ContextLength, k_ContextLength))
        E_lin_cache = np.zeros((k_ContextLength, k_DModel))
        E_relu_cache = np.zeros((k_AttBlocks, k_ContextLength, k_DModel*4))
        
        sSoftmaxMask = np.where(np.triu(np.ones((k_ContextLength, k_ContextLength)), k=1),-np.inf,0.0)
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
            E_preln_cache[currAttBlock, 0] = np.array(E)
            E_ln, E_midln_cache[currAttBlock, 0] = layerNorm(E, currAttBlock, 0)
            E_postln_cache[currAttBlock, 0] = np.array(E_ln)
            currAttHead = 0
            while(currAttHead<k_Attheads):
                E_soft_cache[currAttBlock, currAttHead] = softmax(1/np.sqrt(k_DKey) * E_ln@sWq[currAttBlock, currAttHead]@(E_ln@sWk[currAttBlock, currAttHead]).T+sSoftmaxMask+padMask)
                E+= E_soft_cache[currAttBlock, currAttHead]@(E_ln@sWv[currAttBlock, currAttHead])
                currAttHead+=1
            E_preln_cache[currAttBlock, 1] = np.array(E)
            E_ln, E_midln_cache[currAttBlock, 1] = layerNorm(E, currAttBlock, 1)
            E_postln_cache[currAttBlock, 1] = np.array(E_ln)
            E_relu_cache[currAttBlock] = relu(E_ln@sMLPW1[currAttBlock]+sMLPb1[currAttBlock])
            E += E_relu_cache[currAttBlock]@sMLPW2[currAttBlock]+sMLPb2[currAttBlock]
            currAttBlock+=1
        E_lin_cache = np.array(E)
        E=E@sLW+sLB
        E=softmax(E)

        return E, E_midln_cache, E_soft_cache, E_lin_cache, E_relu_cache, E_postln_cache, E_preln_cache, We_to_E