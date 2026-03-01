#PARAM DECLARATIONS
k_DModel = 384
k_ContextLength = 64
k_VocabSize = 59060+5 #plus five for start, end, pad, and space at start      last one is no idea, must manually set
k_Attheads = 4
k_AttBlocks = 4
k_DQuery = 96
num_times = 25000

k_ShiftFactor = 2
k_BatchSize = 32
k_Alpha = 0.001
k_Beta1 = 0.9
k_Beta2 = 0.999
k_Epsilon = 0.00000001
k_Lambda = 0.01
