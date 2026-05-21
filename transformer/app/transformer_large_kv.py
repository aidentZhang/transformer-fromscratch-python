import numpy as np
import numpy as cp
from pathlib import Path
import time
import itertools
import string
import re


class transformer_inf_large:
    def __init__(self, weights, vocab_list, svocabDict, train_params):
        # PARAM DECLARATIONS
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

        self.sSoftmaxMask = cp.nan_to_num(
            -cp.inf
            * cp.triu(
                cp.ones((self.k_ContextLength, self.k_ContextLength), dtype=cp.float32),
                k=1,
            ),
            nan=0.0,
        )

        self.svocabDict = svocabDict
        self.vocab_list = vocab_list

        #print("successfully initialized!")

    def layerNorm(self, E, attLayer, prePostMLP):
        temp = (E - cp.mean(E, axis=-1, keepdims=True)) / cp.sqrt(
            (cp.nan_to_num(cp.var(E, axis=-1, keepdims=True), nan=0.0) + 0.00001)
        )
        return (
            self.sLNBias[attLayer, prePostMLP]
            + temp * (self.sLNGain[attLayer, prePostMLP]),
            temp,
        )

    def softmax(self, E):
        exp_ = cp.exp(E - cp.max(E, axis=-1, keepdims=True))
        return cp.nan_to_num(exp_ / (cp.sum(exp_, axis=-1, keepdims=True)))

    def relu(self, E):
        return cp.maximum(0, E)

    def relu_deriv(self, E):
        return (E > 0).astype(cp.float32)

    #decode currently is only designed for individual queries, no batches yet
    def decode_inf(self, E, svocabList, ind, min_p = 0):
        start = time.perf_counter()
        rw = E[ind]
        rw[rw < (np.max(rw))*min_p] = 0
        
        rw/=np.sum(rw)

        end = time.perf_counter()
        #print(f"top-k time: {end - start:.4f} seconds")
            # 6. Map the sampled IDs back to your vocabulary strings
        start = time.perf_counter()

        i = np.random.choice(a=self.k_VocabSize, p=rw)
        if i > 3:
            if i - 4 >= len(svocabList):
                return "<NA>"
            else:
                return svocabList[i - 4]
        elif i == 3:
            return " "
        elif i == 1:
            return "<END>"
        elif i == 2:
            return "<PAD>"
        else:
            return "<STA>"


    def prefill(self, input_llm):
        k_BatchSize = self.k_BatchSize
        k_ContextLength = self.k_ContextLength
        k_DModel = self.k_DModel
        k_VocabSize = self.k_VocabSize
        sWpos = self.sWpos
        sWe = self.sWe
        svocabDict = self.svocabDict
        E = cp.zeros((k_BatchSize, k_ContextLength, k_DModel), dtype=cp.float32)

        for i in range(k_BatchSize):
            temp = cp.zeros(k_VocabSize)
            temp[2] = 1
            We_to_E = cp.zeros((k_ContextLength, k_VocabSize))
            We_to_E[0, 0] = 1
            o = 1
            for j in input_llm[i]:
                if j in svocabDict:
                    We_to_E[o, svocabDict[j]] = 1
                else:
                    We_to_E[o, k_VocabSize - 1] = 1
                o += 1

            while o < k_ContextLength:
                We_to_E[o, 2] = 1
                o += 1
            E[i] = We_to_E @ sWe * cp.sqrt(k_DModel)
            E[i] += sWpos
        return E

    # ------------------
    # FORWARD PROPAGATE
    def fowardprop(self, embeddings, padMask, last_token, k_temp):
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
        k_BatchSize = self.k_BatchSize
        k_DKey = self.k_DKey

        sMLPW1 = self.sMLPW1
        sMLPW2 = self.sMLPW2
        sMLPb1 = self.sMLPb1
        sMLPb2 = self.sMLPb2
        sLW = self.sLW
        sLB = self.sLB
        sSoftmaxMask = self.sSoftmaxMask

        relu = self.relu
        layerNorm = self.layerNorm
        softmax = self.softmax
        E = np.array(embeddings)
        currAttBlock = 0
        while currAttBlock < k_AttBlocks:
            E_ln, *_ = layerNorm(E, currAttBlock, 0)
            if(self.fill_cache):
                self.k_cache[currAttBlock] = cp.transpose(
                    cp.reshape(
                        E_ln
                        @ cp.reshape(
                            cp.transpose(sWk[currAttBlock], [1, 0, 2]),
                            [k_DModel, k_DKey * k_Attheads],
                        ),
                        [k_BatchSize, k_ContextLength, k_Attheads, k_DKey],
                    ),
                    [0, 2, 1, 3],
                )
                self.v_cache[currAttBlock] = cp.transpose(
                    cp.reshape(
                        E_ln
                        @ cp.reshape(
                            cp.transpose(sWv[currAttBlock], [1, 0, 2]), [k_DModel, k_DModel]
                        ),
                        [k_BatchSize, k_ContextLength, k_Attheads, k_DModel // k_Attheads],
                    ),
                    [0, 2, 1, 3],
                )
            start = time.perf_counter()

                #set up aliases
            K = self.k_cache[currAttBlock]
            V = self.v_cache[currAttBlock]

            K[:, :, last_token, :] = np.transpose(
                    cp.reshape(
                        E_ln[:, last_token, :]
                        @ cp.reshape(
                            cp.transpose(sWk[currAttBlock], [1, 0, 2]),
                            [k_DModel, k_DKey * k_Attheads],
                        ),
                        [k_BatchSize, 1, k_Attheads, k_DKey],
                    ),
                    [0, 2, 1, 3],
                ).squeeze(2)
            V[:, :, last_token, :] = cp.transpose(
                    cp.reshape(
                        E_ln[:, last_token, :]
                        @ cp.reshape(
                            cp.transpose(sWv[currAttBlock], [1, 0, 2]), [k_DModel, k_DModel]
                        ),
                        [k_BatchSize, 1, k_Attheads, k_DModel // k_Attheads],
                    ),
                    [0, 2, 1, 3],
                ).squeeze(2)
            # for reference:
                    # self.k_cache = np.zeros(
                    #         [self.k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_DKey],
                    # )

                    # self.v_cache = np.zeros(
                    #     (
                    #         [self.k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, self.k_DModel // k_Attheads],
                    #     )
                    # )
            Q = cp.transpose(
                cp.reshape(
                    E_ln
                    @ cp.reshape(
                        cp.transpose(sWq[currAttBlock], [1, 0, 2]),
                        [k_DModel, k_DKey * k_Attheads],
                    ),
                    [k_BatchSize, k_ContextLength, k_Attheads, k_DKey],
                ),
                [0, 2, 1, 3],
            )

            end = time.perf_counter()
            #print(f"QKV calculation time for block {currAttBlock}: {end - start:.4f} seconds")

            start = time.perf_counter()
            # #print(cp.shape(cp.reshape(cp.transpose(E_soft_cache[currAttBlock]@(V), [0, 2, 1, 3]), [k_BatchSize, k_ContextLength, k_DModel])))
            E += (
                cp.reshape(
                    cp.transpose(
                        softmax(
                            1 / cp.sqrt(k_DKey) * Q @ cp.transpose(K, [0, 1, 3, 2])
                            + sSoftmaxMask
                            + cp.expand_dims(padMask, axis=1)
                        )
                        @ (V),
                        [0, 2, 1, 3],
                    ),
                    [k_BatchSize, k_ContextLength, k_DModel],
                )
                @ sWo[currAttBlock]
            )
            end = time.perf_counter()
            #print(f"Attention calculation time for block {currAttBlock}: {end - start:.4f} seconds")

            start = time.perf_counter()
            E_ln, *_ = layerNorm(E, currAttBlock, 1)
            E += (
                relu(E_ln @ sMLPW1[currAttBlock] + sMLPb1[currAttBlock])
                @ sMLPW2[currAttBlock]
                + sMLPb2[currAttBlock]
            )
            end = time.perf_counter()
            #print(f"layernorm + MLP calculation time for block {currAttBlock}: {end - start:.4f} seconds")
            currAttBlock += 1

        E = E @ sLW + sLB
        E = softmax(E / k_temp)
        self.fill_cache = False

        return E

    # ------------------

    # ---------------------------------
    # Data processing functions
    def embed(self, svocabDict, case, BYTE_LOOKUP):
        embeded = []

        words = re.findall(r"\w+|[^\w\s]|\n", case)

        processed_text = [
            [BYTE_LOOKUP[b] for b in word.encode("utf-8")] + ["</w>"] for word in words
        ]
        case = processed_text
        # with tqdm(total=len(case)) as pbar:
        for word in case:
            # pbar.update(1)
            i = 0
            last = len(word)
            while i != len(word):
                # #print(word, " ", i,  " ", last, " ", ''.join(word[i:last]))
                if "".join(word[i:last]) in svocabDict:
                    embeded.append("".join(word[i:last]))
                    i = last
                    last = len(word)
                else:
                    last -= 1
                    if last <= i:
                        embeded.append("</UNKOWN>")
                        break
        return embeded

    def is_hex(self, s):
        return all(c in string.hexdigits for c in s)

    def run_model(self, q):
        #print("running inference ...")
        embed = self.embed
        svocabDict = self.svocabDict
        k_ContextLength = self.k_ContextLength
        decode_inf = self.decode_inf
        vocab_list = self.vocab_list
        is_hex = self.is_hex
        fowardprop = self.fowardprop
        prefill = self.prefill
        k_BatchSize = self.k_BatchSize
        sWe = self.sWe
        k_Attheads=self.k_Attheads
        k_DKey = self.k_DKey
        BYTE_LOOKUP = [f"{i:02x}" for i in range(256)]

        self.k_cache = np.zeros(
                [self.k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, k_DKey]
        )

        self.v_cache = np.zeros(
                [self.k_AttBlocks, k_BatchSize, k_Attheads, k_ContextLength, self.k_DModel // k_Attheads]
        )

        self.fill_cache = True
        q = [embed(svocabDict, q, BYTE_LOOKUP)]
        #q supports batching if needed to be added in the future
        #print("generating padmask ...")
        padMask = cp.zeros((k_BatchSize, k_ContextLength, k_ContextLength))
        for i in range(k_BatchSize):
            padMask[i, :, len(q[i]) + 1 : k_ContextLength] = -cp.inf

        embeddings = prefill(q)
        #print(q)
        k = len(q[0])
        import time

# ... your code here ...

        while k < k_ContextLength:
            start = time.perf_counter()

            padMask = cp.zeros((k_BatchSize, k_ContextLength, k_ContextLength))
            for i in range(k_BatchSize):
                padMask[i, :, k + 1 : k_ContextLength] = -cp.inf
            start2 = time.perf_counter()
            E = fowardprop(embeddings, padMask, k, 0.7)
            end = time.perf_counter()
            #print(f"Forwardprop time: {end - start2:.4f} seconds")
            start = time.perf_counter()
            prediction = decode_inf(E[0], vocab_list, k)
            end = time.perf_counter()
            #print(f"prediction time: {end - start:.4f} seconds")
            # #print(embeddings[0][1])
            if(k != k_ContextLength-1):
                embeddings[0][k+1] = (cp.sqrt(self.k_DModel) * sWe[svocabDict[prediction]]) + self.sWpos[k+1];
            text = prediction.split("</w>")
            post_processed = [
                bytes.fromhex(word).decode("utf-8") if is_hex(word) else word
                for word in text
            ]

            final_text = " ".join(post_processed)
            if final_text == "<EOS>":
                break
            # #print(final_text)
            yield (final_text)


            k += 1

            end = time.perf_counter()
            #print(f"Execution time: {end - start:.4f} seconds")
