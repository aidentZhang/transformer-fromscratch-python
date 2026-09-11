# transformer-fromscratch-python
transformer from scratch without PyTorch or tensorflow just numpy.

main is slightly out-of-date. Take a look at `lambda-deployment` for the most up-to-date inference pipeline, or `win-cuda-accel-wikipedia` for the EC2 training pipeline + Fast BPE

This project is deployed at aidenzhang.dev/#/llm

`win-cuda-accel-wikipedia` contains the full inference pipeline I used to train the model. This was deployed on an EC2 instance with an Nvidia A40, ran for around 200 million tokens, and whose weights were eventually deployed. 

`lambda-deployment` contains the fastAPI connector and an updated inference engine, leading to a 10x inference speedup. The Docker image generated created here was deployed on an AWS Lambda. 

`max-mlx-accel` : It was the code that ran the fastest for me locally, since I have a mac, even beating out a 2080 super using Cupy. Other branches don't have the entire shakespeare corpus and suffer from extreme undertraining. 

`char-level-embedded-transformer` is a legacy branch built for embedding on the character level. When trained on Romeo and Juliet, this model can generate words given a seed character

`main` is a branch written in plain numpy. It include the bype-pair encoding but not the entire training dataset.

Parameters can be changed in params.py. k_vocabsize is dependent on the length of bpe_vocab.py and must be updated every time bpe.py is run.
# Aiden GPT-2.5
Same underlying weights as GPT-2.0, but includes a repeat penalty, KV-cache, min_p instead of top_k, and runs around 10x faster.

# Aiden GPT-2.0
trained on wikipedia this time with 27 million parameters. This used my fast-BPE algorithm (1 million wiki articles in 30 minutes on my MacBook), and was trained on around 200 million tokens. 

# Aiden GPT-1.0
The model has around 2 million parameters. With shakespeare's entire corpus totaling around 1.5-2million tokens, this model is undertrained. However, it does generate text with semebelance of structure. 

GPT-1.0 Example:

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



