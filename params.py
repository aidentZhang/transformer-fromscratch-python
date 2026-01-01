#PARAM DECLARATIONS
k_DModel = 384
k_ContextLength = 64
k_VocabSize = 1074+5 #plus five for start, end, pad, and space      last one is no idea
k_Attheads = 12
k_AttBlocks = 12
k_DQuery = 64
num_times = 1000

k_BatchSize = 16
k_Alpha = 0.0002
k_Beta1 = 0.9
k_Beta2 = 0.98
k_Epsilon = 0.00000001
