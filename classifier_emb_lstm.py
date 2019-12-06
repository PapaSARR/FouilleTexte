import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Input, Dense, Dropout, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, \
    Bidirectional
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors as kv



def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)


def mytokenize_nltk(text):
    """Customized tokenizer.
    Here you can add other linguistic processing and generate more normalized features
    """
    tokens = word_tokenize(text, language='french')
    tokens = [t.lower() for t in tokens if t not in "',;.?!:()#_-+/\\"]
    # tokens = [t for t in tokens if t not in self.stopset]
    return tokens

def load_embs():
    emb_filename = "../resources/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
    emb = kv.load_word2vec_format(emb_filename, binary=True, encoding='UTF-8', unicode_errors='ignore')
    emb_dim = emb.vector_size

    # pretrained word embeddings
    padid = 0
    oovid = 1
    oovVector = np.zeros(emb_dim)
    padVector = np.zeros(emb_dim)
    emb_weights : np.ndarray = emb.vectors
    emb_weights = np.insert(emb_weights, padid, padVector, axis=0)
    emb_weights = np.insert(emb_weights, oovid, oovVector, axis=0)
    word2idx = {word:(emb.vocab[word].index + 2) for word in emb.vocab}
    word2idx['<PAD>'] = padid
    word2idx['<OOV>'] = oovid

    return word2idx, emb_weights, emb_dim, padid, oovid

# load word embeddings
word2idx, emb_weights, emb_dim, padid, oovid = load_embs()
vocab_size = len(word2idx)


def create_model(input_size, ouput_size):
    """Create a neural network model and return it.
    Here you can modify the architecture of the model (network type, number of layers, number of neurones)
    and its parameters"""
    n_timesteps = None
    lstm_size = 100

    # Define input vector, its size = number of features of the input representation
    input = Input(shape=(n_timesteps,))
    # Define output: its size is the number of distinct (class) labels (class probabilities from the softmax)
    layer = input
    # embedding layer
    # Embeddings for words
    layer = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                    input_length=n_timesteps, weights=[emb_weights],
                    trainable=False, mask_zero=True)(layer)
    # layer = GlobalAveragePooling1D(input_shape=(n_timesteps,))(layer)

    layer = Bidirectional(LSTM(lstm_size, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                                   input_shape=(n_timesteps, emb_dim))(layer)

    # layer = Dense(200, activation='relu')(layer)
    # layer = Dropout(.95)(layer)
    output = Dense(ouput_size, activation='softmax')(layer)
    # create model by defining the input and output layers
    model = Model(inputs=input, outputs=output)
    # compile model (pre
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


set_reproducible()
datadir = "../data/"
trainfile =  datadir + "frdataset1_train.csv"
devfile =  datadir + "frdataset1_dev.csv"

# Download data: list of texts (sentences) with reference labels
headers = ['polarity', 'text']
# train data
train_data = pd.read_csv(trainfile, encoding="utf-8", sep='\t', names=headers)
train_texts = train_data['text']
train_labels = train_data['polarity']
# validation data
dev_data = pd.read_csv(devfile, encoding="utf-8", sep='\t', names=headers)
val_texts = dev_data['text']
val_labels = dev_data['polarity']

# hyperparameters
epochs = 200
batchsize = 16
max_features = 8000
#
label_binarizer = LabelBinarizer()
# create the vectorizer
vectorizer = CountVectorizer(
    max_features=max_features,
    strip_accents=None,
    analyzer="word",
    tokenizer=mytokenize_nltk,
    stop_words=None,
    ngram_range=(1, 2),
    binary=False,
    preprocessor=None
)
# fit the vectorizer and the label binarizer
vectorizer.fit(train_texts)
label_binarizer.fit(train_labels)

# create model
input_size = len(vectorizer.vocabulary_) + emb_dim
# input_size = emb_dim
output_size = len(label_binarizer.classes_)
model = create_model(input_size, output_size)


def vectorize(texts):
    # tokenize all texts
    tokenized_texts = [ mytokenize_nltk(text) for text in texts]
    # get max length
    max_len = max([len(tokenized_text) for tokenized_text in tokenized_texts])
    # transform tokens to indices using the embeddings word2idx index (if token is unknown, ind=1
    texts_inds = [ [word2idx[token] if token in word2idx else 1 for token in tokenized_text]
                   for tokenized_text in tokenized_texts ]
    # pad all sequences to max_len
    padded_sequences = pad_sequences(texts_inds, maxlen=max_len, padding='post')
    # create the input matrix, shape = (N, max_len) where N is the number of texts
    X = np.array(padded_sequences)
    return X



# TRAINING
# create the binary output vectors from the correct labels
Y_train = label_binarizer.transform(train_labels)
Y_val = label_binarizer.transform(val_labels)
# create the vectors representing the input texts
X_train = vectorize(train_texts)
X_val = vectorize(val_texts)
# early stopping (optional)
my_callbacks = []
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
my_callbacks.append(early_stopping)
# Train the model now!
print("1. Training the classifier...")
model.fit(
    X_train, Y_train,
    epochs=epochs,
    batch_size=batchsize,
    callbacks=my_callbacks,
    validation_data=(X_val, Y_val),
    verbose=2)

print("\n2. Evaluation on the dev dataset...")
# get the predicted output vectors: each vector will contain a probability for each class label
Y_predicted = model.predict(X_val)
# from the output probability vectors, get the labels that got the best probability scores
predicted_labels = label_binarizer.inverse_transform(Y_predicted)

# compute per-class precision, recall and f1-score
labelset = ["positive", "negative", "neutral"]
precision, recall, f1score, support = precision_recall_fscore_support(val_labels, predicted_labels, labels=labelset, average=None)
print()
for i, label in enumerate(labelset):
    print(f"{label:10}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1score[i]:.4f}\t(n={support[i]})")
# compute average accuracy
precision, _, _, _ = precision_recall_fscore_support(val_labels, predicted_labels, average='micro')
print(f"Micro-averaged accuracy: {precision:.4f}")





















