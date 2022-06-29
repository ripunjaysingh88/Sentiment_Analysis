import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

model = load_model('lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
    
def get_prediction(sentence):

    x_sample_encoded = tokenizer.texts_to_sequences(sentence)
    x_sample_padded = pad_sequences(x_sample_encoded, maxlen=100, padding='pre', truncating='pre')
    prediction = model.predict(x_sample_padded)
    result = prediction[0][0]
    result = round(result,2)
    return result

