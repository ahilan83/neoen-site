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



def predict_linear_reg(X, y, feat_target, sat_data, feat_pred, y_label):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define a pipeline with scaler and linear regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Define the parameter grid for grid search
    param_grid = {
        'regressor__fit_intercept': [True, False],
        'regressor__copy_X': [True, False],
        'regressor__n_jobs': [None, -1]
    }

    # Set up the grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')

    # Use the best model to make predictions
    best_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f'Train MSE: {train_mse}')
    print(f'Test MSE: {test_mse}')
    print(f'Train R2: {train_r2}')
    print(f'Test R2: {test_r2}')

    X = np.concatenate((X_train,X_test),axis=0)
    y = np.concatenate((y_train,y_test),axis=0)
    y_pred = np.concatenate((y_train_pred,y_test_pred),axis=0)

    # Plot the results
    plt.figure(figsize=(5, 4))

    plt.scatter(X[:,1], y, color='blue', label=feat_target)
    plt.scatter(X[:,1], y_pred, color='red', label=feat_pred)
    plt.title('Data')
    plt.xlabel(sat_data)
    plt.ylabel(feat_target)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return y_pred


def lstm_atten_train(X_train, X_test, y_train, y_test, feat_target, sat_data, feat_pred, y_label,epochs=100):
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_with_attention(input_shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

    # Train the model with early stopping
#    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    checkpoint_callback = ModelCheckpoint(filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/Reverse_temp_lstm_atten_best_model.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                    save_weights_only=False,
                                   verbose=1,
                                   mode='min')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min',restore_best_weights=True)

    # Define the final filename format
    final_filepath_format = '/home/ahilan/gen-research/neoen-site/models/weather_prediction/Reverse_temp_lstm_atten_best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
    # Instantiate the custom callback
    rename_callback = RenameBestModelCallback(temp_filepath='/home/ahilan/gen-research/neoen-site/models/weather_prediction/Reverse_temp_lstm_atten_best_model.h5', final_filepath_format=final_filepath_format)

    
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
            
            


def create_lstm_with_attention(input_shape):
    input_layer = Input(shape=input_shape)
    lstm = LSTM(64, return_sequences=True)(input_layer)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(64, return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    attention = Attention()([lstm, lstm])
    concat = Concatenate()([lstm, attention])
    lstm_out_2 = LSTM(64)(concat)
    lstm_out_2 = Dropout(0.2)(lstm_out_2)
    outputs = Dense(1)(lstm_out_2)
    model = Model(input_layer, outputs)
    return model
    

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
