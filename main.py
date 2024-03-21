import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense, LSTM

SEQ_LENGTH = 40
STEP_SIZE = 3 

## MODEL RELATED SETTINGS:
BATCH_SIZE = 256
EPOCHS = 5
#Some global stuff
filepath = tf.keras.utils.get_file('metal stuff.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #reads the text file in lowercase
characters = sorted(set(text)) #finds and sorts each character found in the dataset
char_to_index = dict((c,i) for i, c in enumerate(characters)) #creates a dictionary of all uniqe characters in the dataset
index_to_char = dict((i, c) for i, c in enumerate(characters)) #this converts it back


def generate_model(text, characters, char_to_index, index_to_char):
    sentences = []
    next_characters = []

    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i: i+SEQ_LENGTH])
        next_characters.append(text[i+SEQ_LENGTH])

    x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
    y = np.zeros((len(sentences), len(characters)), dtype=bool)

    for i, sentence, in enumerate(sentences):
        for t, character, in enumerate(sentence):
            x[i, t, char_to_index[character]] = 1
        y[i, char_to_index[next_characters[i]]] = 1


    ##nerual network stuff
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
    model.add(Dense(len(characters)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01))
    model.fit(x,y, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save('textgenerator.keras')

    return text, characters, char_to_index, index_to_char, model


## Character prediction. higher temps pick more risky characters, more creative, more randomish
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def textgeneration(length, temperature, model,):
    start_index = random.randint(0, len(text) - SEQ_LENGTH)
    generated = ''
    setence = text[start_index: start_index + SEQ_LENGTH]
    generated += setence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(setence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        setence = setence[1:] + next_character
        with open('output.txt', 'w') as file:
            file.write(generated)

    print(f"Generated text saved to {'output.txt'}")
    return generated



##### Main menu stuff

def load_model():
    print("Loading Model")
    model = tf.keras.models.load_model('textgenerator.keras')
    print("Model Loaded")
    return model

def generate_new_model():
        generate_model(text, characters, char_to_index, index_to_char)
        return load_model

def run_model(model):
    print("Enter length (int)")
    user_input_length = input()
    # Check if the input can be converted to an integer
    try:
        user_input_length = int(user_input_length)
    except ValueError:
        print("Enter an integer!")
        return  # Return to the main menu if input is not an integer

    print("Great, enter a temperature (float), default = 1.0")
    user_input_temperature = input()
    # Check if the input can be converted to a float
    try:
        user_input_temperature = float(user_input_temperature)
    except ValueError:
        print("Enter a float!")
        return  # Return to the main menu if input is not a float
    textgeneration(user_input_length, user_input_temperature, model)



def main_menu():
    print("Main Menu:")
    print("1. Generate a new model (Takes long!)")
    print("2. Load the model")
    print("3. Exit")
    choice = input("Enter your choice: ")

    if choice == '1':
        return generate_new_model()
    elif choice == '2':
        return load_model()
    elif choice == '3':
        print("Exiting program.")
        exit()
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
        return None

def main():
    model = None
    while True:
        model = main_menu()
        if model is not None:
            run_model(model)

if __name__ == "__main__":
    main()