# Get to the bottom of Transformer 


>[!IMPORTANT]
>코드없는 프로그래밍 채널을 통해 Transformer 개념을 이해하기 위한 RNN 개념 이해 및 간단한 모델 설계 및 Classification, Generation 모델을 만들어 본다.
> 이후 Transformer Encoder 와 Decoder 까지 typing 하는 것을 목표로 한다.

[![코드없는 프로그래밍](http://img.youtube.com/vi/xrq2yN4K_-M/0.jpg)](https://youtu.be/xrq2yN4K_-M)

### 코드없는 프로그래밍 코드
> 링크 : <https://github.com/NoCodeProgram/deepLearning/tree/main>

### Tokenizer 개념 이해를 돕기 위한 링크
> 링크 : <https://tiktokenizer.vercel.app/?model=gpt2>

Tokenizer는 Word를 Computation 하기 위한 숫자 값으로 Embedding 한 Lookup Table 과 같은 개념으로 이해하면 된다. ( Word2Vec 개념 )

Word2Vec Model을 사용하기 위해서는 gensim package를 이용해 사용가능
~~~sh
!pip install gensim
~~~

~~~py
import gensim.downloader as api
for model_name, model_data in sorted(api.info()['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )

model = api.load("word2vec-google-news-300")

print(model.most_similar("cat"))

print(model.most_similar_cosmul(positive=['Seoul', 'France'], negative=['Paris']))


print(model.most_similar_cosmul(positive=['father','woman'], negative=['man']))
print(model.most_similar_cosmul(positive=['brother','woman'], negative=['man']))


print(model.most_similar_cosmul(positive=['soju','mexico'], negative=['korea']))
print(model.most_similar_cosmul(positive=['soju','russia'], negative=['korea']))


~~~
이해를 돕기 위해 Vector Representation 방식으로는 one-hot encoding 을 주로 설명한다.

