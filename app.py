# import tensoflow_datasets as tfds
import os
import tqdm
import re
import tensorflow as tf
from flask import Flask,  render_template, request
import logging
import numpy as np
# preparing input to our model
from keras.models import  load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical





app = Flask(__name__)
padding_size = 1000
model = load_model('M_model.h5')
model.load_weights('M_weights.h5')

IMAGE_FOLDER = os.path.join('static', 'img_pool')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# my_file = open(os.path.join('', 'custom_word_embedding (2).txt'), encoding='utf-8')

@app.route('/')
def home():
    f = os.path.join(app.config['UPLOAD_FOLDER'], 'SENTIMENTs.jpg')
    return render_template('index.html', image = f)

@app.route('/seclassifier', methods = ['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        text = [text]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=500)
        # predictions = predict_fn(text, padding_size)
        pred = model.predict(padded)

        class_names = ['positive', 'negative', 'neutral']
        preds = np.argmax(pred)
        preds= class_names[preds]
        if preds=='positive':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.jpg')
        elif preds=='neutral':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.jpg')
        else:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.jpg')
        return  render_template('index.html',  prediction_text="sentiment of this text is: {} ".format(preds), user_image = filename)

    return render_template("index.html")
if __name__ == "__main__":
    app.run(host='127.0.0.1', port='8000', debug=True)
