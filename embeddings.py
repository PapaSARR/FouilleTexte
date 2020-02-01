import numpy as np
from gensim.models import KeyedVectors as kv

def load_embs():
        #Utilisation de plongements lexicaux statiques
        emb_filename = "../resources/frWac_no_postag_no_phrase_700_skip_cut50.bin"
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