# transformer-fromscratch-python
transformer from scratch without PyTorch or tensorflow just numpy

# THE MOST UPDATE TO DATE BRANCH IS mac-mlx-accel 
But this one will only work on mac. It was the code that ran the fastest for me, since I have a mac, even beating out a 2080 super using Cupy. Other branches don't have the entire shakespeare corpus and suffer from extreme undertraining. 

windows-cuda-accel is a branch optimzed for Nvidia's CUDA. It does not include shakespeare's full corpus and require Cupy and Cuda to be installed.

char-level-embedded-transformer is a branch built for embedding on the character level. When trained on Romeo and Juliet, this model can generate words given a seed character

main is a branch written in plain numpy. It include the bype-pair encoding but not the entire training dataset.

Parameters can be changed in params.py. k_vocabsize is dependent on the length of bpe_vocab.py and must be updated every time bpe.py is run.

Run run_transformer.py to run the current loaded model. If you wish to change to a different model, go in into the Weights folder and drag a weight.npz from 384-xx to directly under /Weights. 

The model has around 2 million parameters. With shakespeare's entire corpus totaling around 1.5-2million tokens, this model is undertrained. However, it does generate text with semebelance of structure. 

Ex:

seed = "Romeo:"
> what he does it will
> 
> I begin to love to know
> 
> to know
> 
> From the duke's aid him, and say it will
> 
> From the duke's aid it will serve it.
>
> 
>
> KING.
> 
>I would have been in mine own fily, sir, that it not; it will serve it END.
> 
>
>BERTRAM


seed = "a"
>
> questakens is his filt-pretarched it.
> 
>
> KING.
>
> I would have it at his wife: if I be a knave to say I begin to squite of it at his own fancest
>
>  to take it at his fancy of recovered fed
> 
This is giberish, but it sort of follows a shakespearean play structure. 



