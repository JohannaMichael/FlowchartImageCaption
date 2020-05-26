from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import load_model

max_length = 21  # maximum length of image captions (see NeuralNet.py, max_length)
print(__name__)
model = load_model('./model_weights/model_30.h5')

images = './FlowchartData/Images/withText/'

with open("./pickle/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

with open("./pickle/wordtoix.pkl", "rb") as encoded_pickle:
    wordtoix = load(encoded_pickle)

with open("./pickle/ixtoword.pkl", "rb") as encoded_pickle:
    ixtoword = load(encoded_pickle)


def greedy_search(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


for testImg in range(10):
    pic = list(encoding_test.keys())[testImg]
    image = encoding_test[pic].reshape((1, 2048))
    x = plt.imread(images + pic)
    plt.imshow(x)
    plt.show()
    print("Greedy: " + str(testImg) + " ", greedy_search(image))

