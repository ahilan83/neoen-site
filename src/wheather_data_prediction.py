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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Attention, Concatenate, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense
from keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional, LSTM, Layer, Dot, Activation, GlobalAveragePooling1D

def calculate_POA(ghi,dni,dhi,albedo,lat,lon,azimuth_angle,start_date='2023-05-17', end_date='2023-05-20'):

    # Find the timezone name using TimezoneFinder
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)

    # Use pytz to get the timezone object
    tz = timezone(tz_name)

    # Create location object to store lat, lon, timezone
    site_location = location.Location(lat, lon, tz=tz)

    times = pd.date_range(start=start_date, end=end_date, freq='H')
    
    solar_position = site_location.get_solarposition(times=times)

    # Extract zenith and azimuth angles
    zenith_angles = solar_position['zenith']
    azimuth_angles = solar_position['azimuth']
    tilt = solar_position['elevation'].clip(lower=0) # for single axis tracker north-south

    aoi = calculate_dynamic_aoi(tilt, azimuth_angle, solar_position['azimuth'], 90-solar_position['zenith'])
    POA = calculate_poa_irradiance(dni, dhi, ghi, aoi, tilt, albedo)

    '''
    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot zenith angle
    plt.subplot(3, 1, 1)
    plt.plot(zenith_angles.index, zenith_angles, label='Zenith Angle')
    plt.xlabel('Time')
    plt.ylabel('Zenith Angle (degrees)')
    plt.title('Solar Zenith Angle over Time')
    plt.legend()
    plt.grid(True)

    # Plot azimuth angle
    plt.subplot(3, 1, 2)
    plt.plot(azimuth_angles.index, azimuth_angles, label='Azimuth Angle', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Azimuth Angle (degrees)')
    plt.title('Solar Azimuth Angle over Time')
    plt.legend()
    plt.grid(True)

    # Plot tilt angle
    plt.subplot(3, 1, 3)
    plt.plot(tilt.index, tilt, label='Tilt Angle', color='red')
    plt.xlabel('Time')
    plt.ylabel('Tilt Angle (degrees)')
    plt.title('Tilt Angle over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
    return POA



def calculate_dynamic_aoi(tilt, azimuth_angle, solar_azimuth, solar_zenith):
    # Convert angles from degrees to radians
    azimuth_angle_rad = np.radians(azimuth_angle)
    solar_azimuth_rad = np.radians(solar_azimuth)
    solar_zenith_rad = np.radians(solar_zenith)
    
    # For single-axis tracker, tilt is approximately the solar zenith angle
    tilt_rad = np.radians(tilt)
    
    # Calculate AOI
    aoi = np.degrees(np.arccos(
        np.cos(solar_zenith_rad) * np.cos(tilt_rad) +
        np.sin(solar_zenith_rad) * np.sin(tilt_rad) * np.cos(solar_azimuth_rad - azimuth_angle_rad)
 ))
    
    return aoi


def calculate_poa_irradiance(dni, dhi, ghi, aoi, tilt, albedo):
    """
    Calculate the plane of array (POA) irradiance.
    
    Parameters:
    dni (float): Direct Normal Irradiance (W/m^2)
    dhi (float): Diffuse Horizontal Irradiance (W/m^2)
    ghi (float): Global Horizontal Irradiance (W/m^2)
    aoi (float): Angle of Incidence (degrees)
    tilt (float): Tilt angle of the panel (degrees)
    albedo (float): Ground reflectance (default is 0.2)
    
    Returns:
    float: Total POA irradiance (W/m^2)
    """   
    # Convert angles from degrees to radians
    aoi_rad = np.radians(aoi)
    tilt_rad = np.radians(tilt)
    
    # Direct component
    poa_direct = dni * np.cos(aoi_rad)

    # Diffuse component (using the isotropic sky model)
    poa_diffuse = dhi * (1 + np.cos(tilt_rad)) / 2

    # Ground-reflected component
    poa_ground_reflected = ghi * albedo * (1 - np.cos(tilt_rad)) / 2

    # Total POA irradiance
    poa_total = poa_direct + poa_diffuse + poa_ground_reflected
    
    return poa_total



def lstm_atten_train(X_train, X_test, y_train, y_test, feat_target, sat_data, feat_pred, y_label,batch_size=32, epochs=100):
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_without_attention(input_shape)

    # Compile the model
#    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])


    checkpoint_callback = ModelCheckpoint(filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_lstm_attention_one_year_data_best_model.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                    save_weights_only=False,
                                   verbose=1,
                                   mode='min')

#    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, 
#                                            mode='min',restore_best_weights=True)

    # Learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Define the final filename format
    final_filepath_format = '/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_lstm_attentionone_year_data_best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
    # Instantiate the custom callback
    rename_callback = RenameBestModelCallback(temp_filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_lstm_attention_one_year_data_best_model.h5', 
                                              final_filepath_format=final_filepath_format)

    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                           epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback,   
                                                                            reduce_lr, 
                                                                            rename_callback])

#    history = model.fit(X_train, y_train, 
#                           epochs=epochs, batch_size=batch_size)

    
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

def transformer_train(X_train, X_test, y_train, y_test, feat_target, sat_data, feat_pred, y_label,epochs=100):
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_transformer_model(input_shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

    # Train the model with early stopping
#    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    checkpoint_callback = ModelCheckpoint(filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_transformer_one_year_data_best_model.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                    save_weights_only=False,
                                   verbose=1,
                                   mode='min')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min',restore_best_weights=True)

    # Define the final filename format
    final_filepath_format = '/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_transformer_one_year_data_best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
    # Instantiate the custom callback
    rename_callback = RenameBestModelCallback(temp_filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/POA_transformer_one_year_data_best_model.h5', final_filepath_format=final_filepath_format)

    
    history = model.fit(X_train, y_train, validation_split=0.2, 
                           epochs=epochs, batch_size=32, callbacks=[checkpoint_callback, early_stopping_callback, rename_callback])

    return model



class RenameBestModelCallback(Callback):
    def __init__(self, temp_filepath, final_filepath_format):
        super(RenameBestModelCallback, self).__init__()
        self.temp_filepath = temp_filepath
        self.final_filepath_format = final_filepath_format

    def on_train_end(self, logs=None):
        # Get the final epoch number
        final_epoch = self.model.stop_training and len(self.model.history.epoch) or 0
        final_val_loss = min(self.model.history.history['val_loss'])
        final_filepath = self.final_filepath_format.format(epoch=final_epoch, val_loss=final_val_loss)
        
        # Rename the temporary best model file
        if os.path.exists(self.temp_filepath):
            os.rename(self.temp_filepath, final_filepath)
            print(f"Best model saved as: {final_filepath}")
        else:
            print("Temporary best model file does not exist.")
            

def create_lstm_without_attention(input_shape):
    input_layer = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm_out_2 = LSTM(64, kernel_regularizer=l2(0.001))(lstm)
    lstm_out_2 = Dropout(0.3)(lstm_out_2)
    outputs = Dense(1)(lstm_out_2)
    model = Model(input_layer, outputs)
    return model
    


def create_lstm_with_attention(input_shape):
    input_layer = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(lstm)
    lstm = Dropout(0.3)(lstm)
    attention = Attention()([lstm, lstm])
    concat = Concatenate()([lstm, attention])
    lstm_out_2 = LSTM(64,kernel_regularizer=l2(0.001))(concat)
    lstm_out_2 = Dropout(0.3)(lstm_out_2)
    outputs = Dense(1)(lstm_out_2)
    model = Model(input_layer, outputs)
    return model



def create_lstm_with_multihead_attention_v0(input_shape):
    input_layer = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(lstm)
    lstm = Dropout(0.3)(lstm)
    attention1 = Attention()([lstm, lstm])
    attention2 = MultiHeadAttention(num_heads=1,key_dim=16)
    attention2_output = attention2(lstm, lstm)
    concat = Concatenate()([lstm, attention1, attention2_output])
    lstm_out_2 = LSTM(64, kernel_regularizer=l2(0.001))(concat)
    lstm_out_2 = Dropout(0.3)(lstm_out_2)
    outputs = Dense(1)(lstm_out_2)
    model = Model(input_layer, outputs)
    return model


def create_lstm_with_multihead_attention_v1(input_shape):
    input_layer = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(lstm)
    lstm = Dropout(0.3)(lstm)
    attention1 = Attention()([lstm, lstm])
    attention2 = MultiHeadAttention(num_heads=8,key_dim=64)
    attention_output = attention2(lstm, lstm, lstm)
    concat = Concatenate()([lstm, attention1, attention_output])
    lstm_out_2 = LSTM(64, kernel_regularizer=l2(0.001))(concat)
    lstm_out_2 = Dropout(0.3)(lstm_out_2)

#    # Use a more complex output layer
    outputs = Dense(64, activation='relu')(lstm_out_2)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(1)(outputs)

    model = Model(input_layer, outputs)
    
    return model



def create_transformer_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for _ in range(2):  # num_layers = 2
        x = TransformerEncoder(
            num_heads=8,
            hidden_dim=input_shape[-1],  # assuming hidden_dim = input_dim
            ff_dim=input_shape[-1],
            dropout=0.2)(x)
    multi_head_attention = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.2)(x, x)
    concat = x + multi_head_attention
    pooling = GlobalAveragePooling1D()(concat)
    outputs = Dense(1)(pooling)
    model = Model(input_layer, outputs)    
    return model

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, hidden_dim, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads, hidden_dim, dropout=dropout
        )
        self.dense_proj = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attention_output = self.self_attention(inputs, inputs, training=training)
        attention_output = self.dropout(attention_output, training=training)
        attention_output = attention_output + inputs
        attention_output = self.layer_norm(attention_output)
        ffn_output = self.dense_proj(attention_output)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = ffn_output + attention_output
        ffn_output = self.layer_norm(ffn_output)
        return ffn_output

def cnn_lstm_atten_train(X_train, X_test, y_train, y_test, feat_target, sat_data, feat_pred, y_label,epochs=100):
    # Create the model
    hybrid_model = create_hybrid_model((X_train.shape[1], X_train.shape[2]))

    # Compile the model
    hybrid_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = hybrid_model.fit(X_train, y_train, validation_split=0.2, 
                           epochs=epochs, batch_size=32, callbacks=[early_stopping])

    return hybrid_model
    

def create_hybrid_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Define the hybrid model
    # CNN layers
    cnn = Conv1D(filters=64, kernel_size=4, activation='relu')(input_layer)
    cnn = MaxPooling1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)

    # LSTM layers
    lstm = LSTM(64, return_sequences=True)(input_layer)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(64, return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    
    # Attention mechanism
    attention = Attention()([lstm, lstm])
    attention = Flatten()(attention)

    # Combine CNN and LSTM outputs
    combined = Concatenate()([cnn, attention])

    # Fully connected layer
    output_layer = Dense(1)(combined)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def gan_train(gan, generator, discriminator, real_data, input_data, epochs=10000, batch_size=64):
    
    # Initialize models
    generator = build_generator(latent_dim, input_dim, output_dim)
    discriminator = build_discriminator(input_dim, output_dim)
    gan = build_gan(generator, discriminator)

    for epoch in range(epochs):
        # Generate synthetic future data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict([noise, input_data])

        # Real data samples
        real_samples = real_data[np.random.randint(0, real_data.shape[0], batch_size)]
        input_samples = input_data[np.random.randint(0, input_data.shape[0], batch_size)]

        # Labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(np.hstack((input_samples, real_samples)), real_labels)
        d_loss_fake = discriminator.train_on_batch(np.hstack((input_samples, generated_data)), fake_labels)

        # Train GAN (generator tries to fool the discriminator)
        gan_loss = gan.train_on_batch([noise, input_samples], real_labels)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss_real[0] + d_loss_fake[0]}, GAN Loss: {gan_loss}")
            

    return model

# Generator model to generate synthetic long-term degradation data
def build_generator(latent_dim, input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim + input_dim),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='tanh')  # 'tanh' for output in range [-1, 1]
    ])
    return model

# Discriminator model to differentiate between real and synthetic data
def build_discriminator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim + output_dim),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # 'sigmoid' for binary classification (real or fake)
    ])
    return model

# GAN model combining generator and discriminator
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False  # Freeze the discriminator when training the GAN

    gan_input_noise = tf.keras.Input(shape=(latent_dim,))
    gan_input_real = tf.keras.Input(shape=(input_dim,))
    generated_data = generator(tf.keras.layers.Concatenate()([gan_input_noise, gan_input_real]))
    gan_output = discriminator(tf.keras.layers.Concatenate()([gan_input_real, generated_data]))

    gan = tf.keras.Model([gan_input_noise, gan_input_real], gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan