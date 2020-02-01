import numpy as np
import embeddings
np.random.seed(15)

from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from datatools import load_dataset
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer() 
word2idx, emb_weights, emb_dim, padid, oovid = embeddings.load_embs()


class Classifier:
    """The Classifier"""
    
    def __init__(self):
        self.labelset = None 
        self.label_binarizer = LabelBinarizer()
        self.model = None
        self.epochs = 200       #Nombre d'itérations
        self.batchsize = 64     #Taille du batch à 64 
        self.max_features = 16000   #Nombre max de traits fixé à 16000
        # create the vectorizer
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            strip_accents="unicode", #On tient compte des caractères accentués
            analyzer="word",    #Analyse par mot
            tokenizer=self.mytokenize_nltk,
            stop_words=None,
            ngram_range=(1, 2),  #Unigramme et bigramme
            binary=True, #On considère les valeurs binaires
            preprocessor=None
        )

    def mytokenize_nltk(self, text):
        """Customized tokenizer.
        Here you can add other linguistic processing and generate more normalized features
        """
        tokens = word_tokenize(text, language='french')
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens if t not in "',;.?!:()#_-+/\\"]
        return tokens
    
    def feature_count(self):
        return len(self.vectorizer.vocabulary_) + emb_dim

    def create_model(self):
        """Create a neural network model and return it.
        Here you can modify the architecture of the model (network type, number of layers, number of neurones)
        and its parameters"""

        # Define input vector, its size = number of features of the input representation
        input = Input((self.feature_count(),))
        layer = input
        # Define output: its size is the number of distinct (class) labels (class probabilities from the softmax)
        #layer = Dense(60, activation='relu')(layer)  #couche intermédiaire de 60 neurones
        #layer = Dropout(.5)(layer) 
        output = Dense(len(self.labelset), activation='softmax')(layer) #fonction softmax car classification multi-classes monolabel
        # create model by defining the input and output layers
        model = Model(inputs=input, outputs=output)
        # compile model (pre
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',  #car classification multi-classes monolabel
                      metrics=['accuracy'])
        model.summary()
        return model

    def vectorize(self, texts):
        X_v1 = self.vectorizer.transform(texts)
        X_v1 = X_v1.toarray()
        X_v2 = np.zeros((len(texts),emb_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self.mytokenize_nltk(text)
            token_ids = [word2idx[token] if token in word2idx else oovid for token in tokens]
            token_vectors = [ emb_weights[token_id] for token_id in token_ids]
            text_vector = np.mean(token_vectors, axis=0)
            X_v2[i] = text_vector
        return np.concatenate([X_v1, X_v2], axis=1)
        
    def train_on_data(self, texts, labels, valtexts=None, vallabels=None):
        """Train the model using the list of text examples together with their true (correct) labels"""
        # create the binary output vectors from the correct labels
        Y_train = self.label_binarizer.fit_transform(labels) #Y_train est une Matrice contenant le nbre de phrases à classifier comme nbre de lignes et trois colonnes (parcequ'il y a 3 labels)
        # get the set of labels
        self.labelset = set(self.label_binarizer.classes_)
        print("LABELS: %s" % self.labelset)
        # build the feature index (unigram of words, bi-grams etc.)  using the training data
        self.vectorizer.fit(texts)
        # create a model to train
        self.model = self.create_model()
        # for each text example, build its vector representation
        X_train = self.vectorize(texts)
        #
        my_callbacks = []
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None)
        my_callbacks.append(early_stopping)
        if valtexts is not None and vallabels is not None:
            X_val = self.vectorize(valtexts)
            Y_val = self.label_binarizer.transform(vallabels)
            valdata = (X_val, Y_val)
        else:
            valdata = None
        # Train the model!
        self.model.fit(
            X_train, Y_train,
            epochs=self.epochs,
            batch_size=self.batchsize,
            callbacks=my_callbacks,
            validation_data=valdata,
            verbose=2)

    def predict_on_X(self, X):
        return self.model.predict(X)

    def predict_on_data(self, texts):
        """Use this classifier model to predict class labels for a list of input texts.
        Returns the list of predicted labels
        """
        X = self.vectorize(texts)
        # get the predicted output vectors: each vector will contain a probability for each class label
        Y = self.model.predict(X)
        # from the output probability vectors, get the labels that got the best probability scores
        return self.label_binarizer.inverse_transform(Y)


    ####################################################################################################
    # IMPORTANT: ne pas changer le nom et les paramètres des deux méthode suivantes: train et predict
    ###################################################################################################
    def train(self, trainfile, valfile=None):
        df = load_dataset(trainfile)
        texts = df['text']
        labels = df['polarity']
        if valfile:
            valdf = load_dataset(valfile)
            valtexts = valdf['text']
            vallabels = valdf['polarity']
        else:
            valtexts = vallabels = None
        self.train_on_data(texts, labels, valtexts, vallabels)


    def predict(self, datafile):
        """Use this classifier model to predict class labels for a list of input texts.
        Returns the list of predicted labels
        """
        items = load_dataset(datafile)
        return self.predict_on_data(items['text'])




