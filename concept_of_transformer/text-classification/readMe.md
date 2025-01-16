
## Trial and Error

>[!IMPORTANT]
> Description : <https://velog.io/@kksj0216/Transformer-TroubleShooting-checkminversion>
> Task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.



#### Issue 1 
> context : Transformer version is latest version 

> Follow below the code then you can solve.
~~~sh
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
~~~

#### Issue 2 
> message : Systemexit : 2

That is wrong environment to develop. In this case, (*_line 332_*) argparser can't be used.