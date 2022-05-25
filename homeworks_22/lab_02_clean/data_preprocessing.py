import torchtext
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator
from nltk.tokenize import WordPunctTokenizer

TRAIN_SIZE = 0.8
VAL_SIZE = 0.15
TEST_SIZE = 0.05
MIN_FREQ = 3

tokenizer_W = WordPunctTokenizer()
def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def get_dataset(path_do_data, 
                train_size=TRAIN_SIZE, 
                val_size=VAL_SIZE, 
                test_size=TEST_SIZE, 
                min_freq = MIN_FREQ):
    """
    input: path to text data, train_data size, val_data size, test_data size, minimum frequency of the word for adding to vocabulary 
    return: (tokenized_train_data,tokenized_val_data,tokenized_test_data), (source_vocab, target_vocab)
    """
    SRC = Field(tokenize=tokenize,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

    TRG = Field(tokenize=tokenize,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

    dataset = torchtext.legacy.data.TabularDataset(
        path=path_do_data,
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    )
    
    train_data, valid_data, test_data = dataset.split(split_ratio=[train_size, val_size, test_size])
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    train_valid_test_data = (train_data, valid_data, test_data)
    
    SRC.build_vocab(train_data, min_freq = min_freq)
    TRG.build_vocab(train_data, min_freq = min_freq)
    print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    
    return train_valid_test_data, SRC, TRG
    
    