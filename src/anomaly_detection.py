import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pvlib import location
from pvlib import irradiance
from timezonefinder import TimezoneFinder
from pytz import timezone
from mpl_toolkits.mplot3d import Axes3D
import pvlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Attention, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Function to sample from the latent space
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    

def vae_lstm_atten_train(X_train, X_test, y_train, y_test, feat_target, sat_data, feat_pred, y_label,epochs=100):
    # Hyperparameters
    input_dim = 1   
    latent_dim = 20
    
    # Encoder
    inputs = Input(shape=(timesteps, input_dim),name='encoder_input')
    lstm = LSTM(64, return_sequences=True, name='encoder_lstm1')(input_layer)
    lstm = Dropout(0.2, name='encoder_dropout1')(lstm)
    lstm = LSTM(64, return_sequences=True, name='encoder_lstm2')(lstm)
    lstm = Dropout(0.2, name='encoder_dropout2')(lstm)
    attention = Attention(name='encoder_attention')(lstm)
    concat = Concatenate(name='encoder_concat')([lstm, attention])
    lstm_out = LSTM(64, name='encoder_lstm3')(concat)
    lstm_out = Dropout(0.2, name='encoder_dropout3')(lstm_out)
    z_mean = Dense(latent_dim, name='z_mean')(lstm_out)
    z_log_var = Dense(latent_dim, name='z_log_var')(lstm_out)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


    # Decoder
#    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    decoder_repeat = RepeatVector(timesteps, name='decoder_repeat')(decoder_input)
    decoder_lstm1 = LSTM(64, return_sequences=True, name='decoder_lstm1')(decoder_repeat)
    decoder_lstm2 = LSTM(64, return_sequences=True, name='decoder_lstm2')(decoder_lstm1)
    decoder_output = TimeDistributed(Dense(input_dim), name='decoder_output')(decoder_lstm2)

    
    # VAE model
    vae = Model(input_layer, decoder_output, name='vae')
    encoder = Model(input_layer, z_mean, name='encoder')


    # Loss function
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(decoder_output))
    reconstruction_loss *= timesteps * input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + 0.25*kl_loss)
    vae.add_loss(vae_loss)

    batch_size = 64

    checkpoint_callback = ModelCheckpoint(
        filepath='/home/ahilan/gen-research/neoen-site/models/anomaly_detection/power_loss/model_checkpoint_{epoch:02d}.h5',  # Save with epoch and batch number in the filename
        save_weights_only=False,  # Set to True if you only want to save the model weights
        save_freq=5 * len(train_sequences) // batch_size,  # Change this to 'batch' to save after every batch
        verbose=1  # Set to 1 for verbose logging, 0 for silent
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', # Monitor the validation loss
        patience=10, # Stop after 10 epochs of no improvement
        verbose=1
    )

    # Compile model
    optimizer = Adam(learning_rate=1e-3)
    vae.compile(optimizer=optimizer)
    vae.summary()


    # Train the VAE with early stopping
    history = vae.fit(train_sequences[:,:,1], epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping_callback, checkpoint_callback])


    return vae
