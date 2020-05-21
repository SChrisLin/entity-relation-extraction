'''
    text description
'''
import pickle


from keras.layers import LSTM, Bidirectional, Embedding
from keras.models import Sequential
from keras_contrib.layers import CRF

EMBED_DIM = 200
BIRNN_UNITS = 200



def create_model(train=True):
    '''
        function description
    '''
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BIRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        out = (model, (train_x, train_y), (test_x, test_y))
    else:
        out = (model, (vocab, chunk_tags))
    return out
