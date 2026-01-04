#PARAM DECLARATIONS
k_DModel = 384
k_ContextLength = 96
k_VocabSize = 4985+5 #plus five for start, end, pad, and space at start      last one is no idea, must manually set
k_Attheads = 4
k_AttBlocks = 4
k_DQuery = 96
num_times = 5000

k_ShiftFactor = 2
k_BatchSize = 16
k_Alpha = 0.0005
k_Beta1 = 0.9
k_Beta2 = 0.999
k_Epsilon = 0.00000001
