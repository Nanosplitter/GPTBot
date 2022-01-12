from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
class NN:
    def __init__(self, modelFilePath):
        

        # Create a sorted list of the characters
        self.chars = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '\\', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ì', 'í', '\u200d', '’', '☀', '☝', '☠', '☹', '♀', '♂', '✅', '✊', '✌', '❤', '️', '🌞', '🌵', '🍉', '🍕', '🍺', '🍻', '🎂', '🎆', '🐢', '👀', '👋', '👌', '👍', '👎', '👻', '💀', '💙', '💩', '💪', '💯', '💸', '📑', '📓', '📔', '📕', '📗', '📘', '📙', '📚', '🔖', '🔥', '🔫', '🖕', '🖖', '😀', '😁', '😂', '😃', '😄', '😆', '😇', '😉', '😊', '😋', '😍', '😎', '😏', '😐', '😑', '😒', '😔', '😕', '😛', '😜', '😠', '😢', '😥', '😦', '😩', '😬', '😭', '😮', '😯', '😲', '😴', '😶', '🙁', '🙂', '🙃', '🙄', '🙏', '🤑', '🤔', '🤖', '🤘', '🤙', '🤞', '🤟', '🤢', '🤣', '🤦', '🤨', '🤪', '🤫', '🤭', '🤮', '🤷', '🥫', '🥲', '🥳', '🥺', '🧐', '🧟']
        # Create a dictionary where given a character, you can look up the index and vice versa
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of self.maxlen characters
        self.maxlen = 15 # The window size

        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', run_eagerly=False)

        self.model.load_weights(modelFilePath)
    
    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def getResponse(self, seed):
        sentence = ((" "*self.maxlen) + seed)[-15:]
        x = np.zeros((1, self.maxlen, len(self.chars)))
        for t, char in enumerate(sentence):
            x[0, t, self.char_indices[char]] = 1.
        
        variance = 0.2
        generated = ''
        original = seed
        window = sentence
        for i in range(280):
            x = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(window):
                x[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, variance)
            next_char = self.indices_char[next_index]

            generated += next_char
            window = window[1:] + next_char

        return generated