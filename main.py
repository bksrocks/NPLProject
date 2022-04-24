import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

tokenizer = Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False,
    oov_token=None
)

data = open('wilde.txt').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

# We can go from word to index using "word_index"
# But we need to be able to do the opposite (from index to word)
#
index_to_word = [''] * (len(tokenizer.word_index) + 1)
for curr_key in tokenizer.word_index.keys():
    index_to_word[tokenizer.word_index[curr_key]] = curr_key

input_sequences = []
for line in corpus:
    # This converts a line of words, to a line of numbers
    #
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=10, verbose=1)
print(model)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


def return_best_words(seed_text_input, num_of_words):
    token_list = tokenizer.texts_to_sequences([seed_text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted = np.argsort(predicted)
    predicted = predicted.tolist()[0]

    # We can reverse predicted by doing
    predicted = predicted[::-1]

    returned_list = []
    for el in range(num_of_words):
        returned_list.append(index_to_word[predicted[el]])

    return returned_list


plot_graphs(history, 'accuracy')

seed_text = input("Start with something: ")
next_words = 100

for _ in range(input("Length of sentence:")):
    best_three = []  # Assume seed is "I thought that"
    best_words = return_best_words(seed_text, num_of_words=3)  # [today, tomorrow, yesterday]

    seed_text_0_0 = seed_text + " " + best_words[0]  # I thought that today [b0_0, b0_1, b0_2]
    b0_0, b0_1, b0_2 = return_best_words(seed_text_0_0, num_of_words=3)

    seed_text_0_1 = seed_text + " " + best_words[1]  # I thought that tomorrow [b1_0, b1_1, b1_2]
    b1_0, b1_1, b1_2 = return_best_words(seed_text_0_1, num_of_words=3)

    seed_text_0_2 = seed_text + " " + best_words[2]  # I thought that tomorrow [b1_0, b1_1, b1_2]
    b2_0, b2_1, b2_2 = return_best_words(seed_text_0_1, num_of_words=3)

    # best_three.append(b0_0)
    # best_three.append(b0_1)
    # best_three.append(b0_2)

    print("Option 1:", seed_text_0_0)
    print(f"0.{b0_0}, 1. {b0_1}, 2. {b0_2}")

    print("Option 2:", seed_text_0_0)
    print(f"0.{b1_0}, 1. {b1_1}, 2. {b1_2}")

    print("Option 3:", seed_text_0_0)
    print(f"0.{b2_0}, 1. {b2_1}, 2. {b2_2}")

    choice = input("Select:")

    if choice == 1:
        best_three.append(b0_0)
        best_three.append(b0_1)
        best_three.append(b0_2)
        print(f"0.{b0_0}, 1. {b0_1}, 2. {b0_2}")
        user_val = int(input("Pick a number: "))
        seed_text = seed_text_0_0 + " " + best_three[user_val]
        print(seed_text)

    elif choice == 2:
        best_three.append(b1_0)
        best_three.append(b1_1)
        best_three.append(b1_2)
        print(f"0.{b1_0}, 1. {b1_1}, 2. {b1_2}")
        user_val = int(input("Pick a number: "))
        seed_text = seed_text_0_1 + " " + best_three[user_val]
        print(seed_text)

    elif choice == 3:
        best_three.append(b2_0)
        best_three.append(b2_1)
        best_three.append(b2_2)
        print(f"0.{b2_0}, 1. {b2_1}, 2. {b2_2}")
        user_val = int(input("Pick a number: "))
        seed_text = seed_text_0_2 + " " + best_three[user_val]
        print(seed_text)

    else:
        print("Invalid choice")
        seed_text = seed_text

print(seed_text)
