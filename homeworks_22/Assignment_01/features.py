from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        words_set = set(' '.join(X).split())
        all_text = " ".join(X)
        bow_vocabulary = sorted([(word, all_text.count(word)) for word 
                                 in words_set], key = lambda item: -item[1])
        self.bow = [item[0] for item in bow_vocabulary[:self.k]]
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = None
        tokens = tokenizer.tokenize(text.lower())
        result = [tokens.count(token) for token in self.bow]
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow

    

class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        super(self, TfIdf).__init__()
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()
        self.symbols = '!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n""1234''567890'
        self.vocab = None
        self.counter = None
        self.all_docs = None
        self.size = None
        
    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        self.all_docs = " ".join(X)
        self.size = len(X)
        
        for symbol in self.symbols: 
            self.all_docs = self.all_docs.replace(symbol, " ").lower()
        
        self.vocab = list(set(self.all_docs.split(" ")))
        
        if not self.k:
            self.k = len(self.vocab)
            
        self.counter = {word : self.all_docs.count(word) for word in self.vocab if len(word)>1}
        self.counter = sorted([(word, count) for word, count in self.counter.items()], 
                              key = lambda item: -item[1])[:self.k]
        # fit method must always return self
        
        self.vocab = [word for (word, count) in self.counter]    
        self.idf = {word : np.log((self.size +1)/(count + 1)) + 1 for word, count in self.counter}
        
        return self

    def get_words_count(self, word):
        """
        return how many documents contains the word 
        """
        count = 0
        for text in self.all_docs:
            count += int(word in text.lower().split())
        return count
    
    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        result = np.zeros(self.k)
        for symbol in self.symbols:
            text  =  text.replace(symbol, " ").lower()
        words_list = text.split()
        
        for word in words_list:
                
                if word in self.vocab:
                    index = self.vocab.index(word)
                    tf = text.count(word)/len(text)
                    
                    result[index] = tf * self.idf[word]
                    
        if self.normalize:
            result = (result - result.mean())/(result.std() + 0.001)
            
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
