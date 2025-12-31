#PARAM DECLARATIONS
k_DModel = 128 #32
k_ContextLength = 128#8
k_VocabSize = 567+5 #plus four for start, end, pad, and space      last one is no idea
k_Attheads = 2
k_AttBlocks = 4
k_DQuery = 64 #key and value space are same
num_times = 500



k_BatchSize = 16
k_Alpha = 0.0001
k_Beta1 = 0.9
k_Beta2 = 0.98
k_Epsilon = 0.00000001
