import os
import numpy as np
import pandas as pd
import utils

from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def get_model(vocab, input_length, embedding_dim=256, rdropout=0.2, dropout=0.2):
    model = Sequential([
        Embedding(vocab, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=rdropout, dropout=dropout)),
        Dense(1, activation="sigmoid")
        ])
    return model

def compile_model(model, lr, loss='binary_crossentropy', metrics=['accuracy']):
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr),
                  metrics=metrics)
    return model

def get_callbacks(patience=3, verbose=1):
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=patience, restore_best_weights=True, verbose=verbose)
    lrp = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, verbose=verbose)
    return (es, lrp)

def save_chart(history, metric='accuracy'):

    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss vs. epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    fig.add_subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'{metric} vs. epoch')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('chart.png')

def train(epoch=20, batch_size=64, char_size=27, maxlen=15, embedding_dim=256, rdropout=0.2, dropout=0.2):
    
    # setting up data directory
    parent_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(parent_path, 'data/name_gender.csv')
    model_path = os.path.join(parent_path, 'model/saved_model.h5')
    
    # loading data
    df = pd.read_csv(data_path)
    df = utils.preprocess(df)

    # instantiate model
    model = get_model(vocab=char_size, input_length=maxlen, embedding_dim=embedding_dim, rdropout=rdropout, dropout=dropout)
    model = compile_model(model=model, lr=0.001)

    # splitting of data in to train and val set
    X = np.asarray(df['name'].values.tolist())
    y = np.asarray(df['gender'].values.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # model training
    early_stopping, learning_rate_reduction = get_callbacks()
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epoch, validation_data=(X_test, y_test), callbacks=[early_stopping, learning_rate_reduction])

    # Save the model
    model.save(model_path)
    save_chart(history)
    
    return model, history
