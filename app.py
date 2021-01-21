# import tensoflow_datasets as tfds
import os
import tqdm
import re
import tensorflow as tf
from flask import Flask,  render_template, request
from flask_cors import cross_origin
import logging
import numpy as np
# preparing input to our model
from keras.models import  load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle




app = Flask(__name__)
app.debug = True
app.run()

padding_size = 300
model = load_model('M.h5')
model.load_weights('w (1).h5')

IMAGE_FOLDER = os.path.join('static', 'img_pool')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# my_file = open(os.path.join('', 'custom_word_embedding (2).txt'), encoding='utf-8')

@app.route('/')
@cross_origin()
def home():
    f = os.path.join(app.config['UPLOAD_FOLDER'], 'SENTIMENTs.jpg')
    return render_template('index.html', image = f)

@app.route('/seclassifier', methods = ['GET','POST'])
@cross_origin()
def predict_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        text = [text]
        
        tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        
        tokenizer.fit_on_texts(text)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=300)
        # predictions = predict_fn(text, padding_size)
        pred = model.predict(padded)

        class_names = ['positive', 'negative']
        preds = np.argmax(pred)
        preds= class_names[preds]
        if preds=='positive':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.jpg')
        else:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.jpg')
        return  render_template('index.html',  prediction_text="sentiment of this text is: {} ".format(preds), user_image = filename)

    return render_template("index.html")
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)
