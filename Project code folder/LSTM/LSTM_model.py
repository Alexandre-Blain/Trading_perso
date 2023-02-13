from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import optuna

#best class archi
#params={'learning_rate': 0.0008857250003024605, 'activation': 'tanh', 'dense_units': 48, 'lstm_units': 22, 'n_layer': 2, 'dropout_rate': 0.03251418561585747}

weights_vec = np.array([3, 3, 1, 3, 3])

fat = 0.6
med = 0.3
low = 1.2
opo = 14.0
same = 1.0
opo_less = 10.5

weights_matrix = np.array([[fat, same, opo_less, opo_less, opo],
                [same, med, opo_less, opo_less, opo],
                [opo, 1.3, low, 1.3, opo],
                [opo, opo_less, opo_less, med, same],
                [opo, opo_less, opo_less, same, fat]])

#loss function

def canal_high_loss(malus = 0.1):
    malus = K.constant(malus)
    def loss(y_true, y_pred):
        diff = y_pred - y_true
        loss = K.mean(K.square(diff))
        penalty = K.relu(diff)
        loss = loss + malus * K.mean(penalty)
        return loss
    return loss

def trend_loss(y_true, y_pred):
    err = []
    reward = 0.01
    malus = 5
    for i in range(1,len(y_true)):
        # Calculate difference between true value and previous true value
        trend = y_true[i] - y_true[i-1]
        # Calculate difference between predicted value and previous true value
        diff = y_pred[i] - y_true[i-1]
        # Check if prediction is in line with trend
        if (trend > 0 and diff > 0) or (trend < 0 and diff < 0):
            # Add reward to loss
            err.append(reward)
        else:
            # No reward for incorrect predictions
            err.append(malus)
    # Calculate MSE
    error = y_true-y_pred    
    sqr_error = K.square(error)
    # Multiply errors by weights and sum
    weights_apply = tf.math.multiply(sqr_error, err)
    mean_sqr_error = K.mean(weights_apply)

    loss = tf.math.reduce_mean(mean_sqr_error)
    return loss

def weighted_categorical_crossentropy_vec(weights_vec):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights_vec = K.variable(weights_vec)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights_vec
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_categorical_crossentropy_matrix(weights_matrix):
    weights_matrix = K.constant(weights_matrix)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred)
        true_class = K.argmax(y_true, axis=-1)
        pred_class = K.argmax(y_pred, axis=-1)
        indices = tf.stack([true_class, pred_class], axis=-1)
        trend_factor = tf.gather_nd(weights_matrix, indices)
        trend_factor = tf.cast(trend_factor, tf.float32)
        loss = -K.sum(loss, -1) * trend_factor
        return loss
    return loss

#models
def lstm_reg_model(trial, timestep, features, dense_units=50, lstm_units=100, dropout_rate=0.2, learning_rate=0.001, activation = 'relu', n_layer = 1, loss  = trend_loss):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(timestep, features), dropout=dropout_rate))
    for i in range(n_layer):
        activation = trial.suggest_categorical(f'activation_{i}', ['sigmoid', 'relu', 'tanh', 'elu', 'gelu', None])
        dense_units = trial.suggest_int(f"dense_units_{i}", 1, 150)
        model.add(Dense(units=dense_units, activation=activation))
    model.add(Dense(1))
    model.compile(loss = loss, optimizer=Adam(learning_rate), metrics=['mean_squared_error'])
    return model

def lstm_class_model(output_dim, timestep, features, dense_units=50, lstm_units=100, dropout_rate=0.2, learning_rate=0.001, activation = 'relu', n_layer = 1, loss = weighted_categorical_crossentropy_vec(weights_vec)):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(timestep, features), dropout=dropout_rate))
    for i in range(n_layer):
        model.add(Dense(units=dense_units, activation = activation))
    model.add(Dense(output_dim, activation = 'softmax'))
    model.compile(loss = loss, optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model





