import numpy as np
# print(self.k_VocabSize)
# print(self.k_DModel)
# print(self.k_ContextLength)
# print(self.k_DKey)
# print(self.k_AttBlocks)
# print(self.k_Attheads)
# print(self.k_Dquery)



def load_model():
    num_times = 5000

    weights = np.load("./Weights/weights.npz")
    with open('bpe_rules.txt', 'r') as f:
        rule_list = []
        i = 0
        while(i < num_times):
            
            rule_list.append((next(f)[:-1].replace('\\n', '\n'), next(f)[:-1].replace('\\n', '\n')))
            i+=1
        
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
    
    model = gpt(weights, rule_list, svocabDict, vocab_list)

    return model

class gpt:
    def __init__(self, weights, rule_list, svocabDict, vocab_list):  
        self.vocab_list = vocab_list
        self.rule_list = rule_list
        self.svocabDict = svocabDict
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
    
    def layerNorm(self, E, attLayer, prePostMLP):
        temp = np.nan_to_num((E-np.mean(E, axis = -1, keepdims = True))/np.sqrt((np.nan_to_num(np.var(E, axis = -1, keepdims = True), nan = 0.)+0.00001)))
        return self.sLNBias[attLayer, prePostMLP] + temp * (self.sLNGain[attLayer, prePostMLP])

    def softmax(self, E):
        return np.nan_to_num(np.exp(E)/(np.exp(E)@np.ones((E.shape[1],1))), nan = 0)

    def relu(self, E):
        return np.maximum(0, E)

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
        #------------------
        #CACHEING

        sSoftmaxMask = np.where(np.triu(np.ones((self.k_ContextLength, self.k_ContextLength)), k=1),-np.inf,0.0)
        padMask = np.zeros((self.k_ContextLength, self.k_ContextLength))
        padMask[:, len(input_llm)+1:self.k_ContextLength] = -np.inf

        temp = np.zeros(self.k_VocabSize)
        temp[2] = 1
        We_to_E = np.zeros((self.k_ContextLength, self.k_VocabSize))
        We_to_E[0, 0] = 1
        i = 1
        for j in input_llm:
            if j in svocabDict:
                We_to_E[i, [svocabDict[j]]] = 1
            else:
                We_to_E[i, self.k_VocabSize-1] = 1
            i+=1
        while(i<self.k_ContextLength):
            We_to_E[i, 2] = 1
            i+=1
        E = We_to_E@self.sWe

        i = 0
        while(i<len(input_llm)+1):
            E[i]+=self.sWpos[i]
            i+=1

        currAttBlock = 0
        while(currAttBlock < self.k_AttBlocks):
            E_ln = self.layerNorm(E, currAttBlock, 0)
            currAttHead = 0
            while(currAttHead<self.k_Attheads):
                E+= self.softmax(1/np.sqrt(self.k_DKey) * E_ln@self.sWq[currAttBlock, currAttHead]@(E_ln@self.sWk[currAttBlock, currAttHead]).T+sSoftmaxMask+padMask)@(E_ln@self.sWv[currAttBlock, currAttHead])
                currAttHead+=1
            E_ln = self.layerNorm(E, currAttBlock, 1)
            E += self.relu(E_ln@self.sMLPW1[currAttBlock]+self.sMLPb1[currAttBlock])@self.sMLPW2[currAttBlock]+self.sMLPb2[currAttBlock]
            currAttBlock+=1

        E=E@self.sLW+self.sLB
        E=self.softmax(E)

        return E

    def gen(self, q):
        q = list(q)
        q = self.embed(self.rule_list, q)
        k = len(q)
        # print(q)
        while k < self.k_ContextLength:
            E = self.fowardprop(q, self.svocabDict)
            prediction = self.decode(E, self.vocab_list)
            q.append(prediction[k])
            # print(prediction[k], end='')
            k+=1
        # print("")
        return ''.join(q[:-1])
