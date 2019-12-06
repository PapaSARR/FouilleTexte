import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Input, Dense, Dropout
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

#Les embbedings sont un ensemble de mots qui sont représentés en vecteurs denses (pas de 0 dans le vecteur) de même taille
#Pour passer de la représentation des mots vers la représentation des phrases (ie obtenir les vecteurs des phrases), il faut faire la moyenne des vecteurs des mots (ce qu'on fait ici) ou utiliser les RNN (LSTM ou GRU) ou BiRNN (ce qu'on fait dans le fichier emb_lstm)
def load_embs():
    emb_filename = "../resources/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
    emb = kv.load_word2vec_format(emb_filename, binary=True, encoding='UTF-8', unicode_errors='ignore') #Les embbedings sont chargés en mémoire
    emb_dim = emb.vector_size #On récupère la taille des vecteurs

    # pretrained word embeddings
    padid = 0 #indice des mots artificiels, les mots artificiels permettent d'avoir des vecteurs de même dimension pour toutes les phrases
    oovid = 1 #indice des mots inconnus, out of vocabulary
    oovVector = np.zeros(emb_dim) #Vecteur des mots inconnues initialisé à null (que des zéros)
    padVector = np.zeros(emb_dim) #Vecteur des mots artificiels initialisé à null (que des zéros)
    emb_weights = emb.vectors	#On récupère la matrice ( vecteur ayant comme élément les vecteurs de mots embbeding
    emb_weights = np.insert(emb_weights, padid, padVector, axis=0) #On insère le Vecteur des mots artificiels à l'indice 0
    emb_weights = np.insert(emb_weights, oovid, oovVector, axis=0) #On insère le Vecteur des mots inconnues à l'indice 1
    word2idx = {word:(emb.vocab[word].index + 2) for word in emb.vocab} #On parcoure tous les mots de la matrice
    word2idx['<PAD>'] = padid
    word2idx['<OOV>'] = oovid
    
    return word2idx, emb_weights, emb_dim, padid, oovid


def create_model(input_size, ouput_size):
    """Create a neural network model and return it.
    Here you can modify the architecture of the model (network type, number of layers, number of neurones)
    and its parameters"""

    # Define input vector, its size = number of features of the input representation
    input = Input((input_size,))
    # Define output: its size is the number of distinct (class) labels (class probabilities from the softmax)
    layer = input
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

# load word embeddings
word2idx, emb_weights, emb_dim, padid, oovid = load_embs()

# create model
input_size = len(vectorizer.vocabulary_) + emb_dim
output_size = len(label_binarizer.classes_)
model = create_model(input_size, output_size)


def vectorize(texts):
    X_v1 = vectorizer.transform(texts)
    X_v1 = X_v1.toarray()
    X_v2 = np.zeros((len(texts),emb_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = mytokenize_nltk(text)
        token_ids = [word2idx[token] if token in word2idx else oovid for token in tokens]
        token_vectors = [ emb_weights[token_id] for token_id in token_ids]
        text_vector = np.mean(token_vectors, axis=0)
        X_v2[i] = text_vector
    # return X_v2
    return np.concatenate([X_v1, X_v2], axis=1)
    #return X_v1



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





















