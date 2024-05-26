import panda as pd
import seaborn as sb
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Setup NLTK resources
nltk.download('punkt')

# Reading data files
test_filename = 'test.csv'
train_filename = 'train.csv'
valid_filename = 'valid.csv'

train_news = pd.read_csv(train_filename)
test_news = pd.read_csv(test_filename)
valid_news = pd.read_csv(valid_filename)

# Data observation
def data_obs():
    print("Training dataset size:")
    print(train_news.shape)
    print(train_news.head(10))

    print("Test dataset size:")
    print(test_news.shape)
    print(test_news.head(10))
    
    print("Validation dataset size:")
    print(valid_news.shape)
    print(valid_news.head(10))

# Distribution of classes for prediction
def create_distribution(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='hls')

create_distribution(train_news)
create_distribution(test_news)
create_distribution(valid_news)

# Data integrity check
def data_quality_check():
    print("Checking data qualities...")
    print(train_news.isnull().sum())
    train_news.info()
        
    print(test_news.isnull().sum())
    test_news.info()

    print(valid_news.isnull().sum())
    valid_news.info()

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

# Process the data
def process_data(data, exclude_stopword=True, stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = stem_tokens(tokens, porter)
    if exclude_stopword:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords]
    return tokens_stemmed


