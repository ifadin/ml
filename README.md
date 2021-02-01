# ML Notebooks

Setup:

```shell script
    make init
```

## Sentiment analysis

### DBPedia

[Blazingtext](https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia.ipynb)
.

### IMDB

#### BlazingText

Unpack data:

```shell script
  mkdir -p imdb
  wget -O imdb/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz    
  tar -zxf imdb/aclImdb_v1.tar.gz -C imdb
  mv imdb/aclImdb imdb/data 
  rm imdb/aclImdb_v1.tar.gz
```

Prepare data (cleanup, contractions expansion, lemmatization):

```shell script
    source setup.rc
    python -m imdb.blz.prepare_data
```

Train model:

```shell script
    python -m imdb.blz.train_model
```

#### PyTorch with LSTM

Extract IMDB dataset to `imdb/data`.

Activate virtual env:

```shell
    source setup.rc
```

Format data (tokens to indices):

```shell
    python3 -m imdb.pyt.prepare_data
```

Train locally:

```shell
    python3 imdb/pyt/code/train.py --epochs=5
```

On SageMaker:

```shell
    python3 -m imdb.pyt.run --epochs=5
```

## Jupyter

```shell script
    jupyter notebook --ip=0.0.0.0 --port=8080
```
