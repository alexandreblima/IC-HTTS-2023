# State-of-Charge (SOC) estimation of Panasonic 18650PF Li-ion Battery
# One to six Hidden Layers
# CROSS-VALIDATION
# Data driven estimation using machine learning - neural nets
# data: https://data.mendeley.com/datasets/wykht8y7tg/1
# NN - 25degC
# contributor:  
# Original author: Alexandre Barbosa de Lima
# Adapted by: HELOISA THERESA TEIXEIRA SALIBA
# FCET - PUC/SP
# Date: 31/08/2023
 
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K

# Verify the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load training and test data
train_data_dic = scipy.io.loadmat("C:\\Users\\alexa\\OneDrive\\Área de Trabalho\\SOC-estimation-DL\\simulations\\NN\\25DegC\\train_data.mat")
train_data = train_data_dic['train_data']

test_data_dic = scipy.io.loadmat("C:\\Users\\alexa\\OneDrive\\Área de Trabalho\\SOC-estimation-DL\\simulations\\NN\\25DegC\\test_data.mat")
test_data = test_data_dic['test_data']

# Load training and test labels
train_label_dic = scipy.io.loadmat("C:\\Users\\alexa\\OneDrive\\Área de Trabalho\\SOC-estimation-DL\\simulations\\NN\\25DegC\\train_label.mat")
train_targets = train_label_dic['SOC_label_train']

test_label_dic = scipy.io.loadmat("C:\\Users\\alexa\\OneDrive\\Área de Trabalho\\SOC-estimation-DL\\simulations\\NN\\25DegC\\test_label.mat")
test_targets = test_label_dic['SOC_label_test']

# Hyperparameters
num_epochs = 50
batch_size = 256
n_hidden = 64  # Number of units per layer
lr = 0.001  # Learning rate

# Loss function
mse = tf.keras.losses.MSE

# VALIDATION PHASE

# Build neural net for cross-validation
def build_model(n_hidden_layers=1):
    model = models.Sequential()
    model.add(layers.Dense(n_hidden, activation='relu', input_shape=(train_data.shape[1],)))
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_hidden, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss=mse, metrics=['mae'])
    return model

# Prepare for K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(train_data)

# Store histories for each number of hidden layers
all_histories = {}

for n_hidden_layers in range(1, 7):  # From 1 to 6 layers
    all_mae_histories = []
    print(f"Testing model with {n_hidden_layers} hidden layers.")
    
    for train_index, val_index in kf.split(train_data):
        print("Processing fold #", len(all_mae_histories) + 1)

        # Split the data into training and validation for this fold
        partial_train_data = train_data[train_index]
        partial_train_targets = train_targets[train_index]
        val_data = train_data[val_index]
        val_targets = train_targets[val_index]

        # Build and compile the model
        model = build_model(n_hidden_layers)

        # Train the model
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=batch_size, verbose=1)

        # Evaluate the model and store the metrics for future analysis
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)

    # Calculate the average MSE metrics across folds
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    # Store in a dictionary
    all_histories[f"{n_hidden_layers}_hidden_layers"] = average_mae_history

# Print available keys in "history.history"
print("Available keys in history.history: ", history.history.keys())

# Plotting all configurations
for n_hidden_layers, history in all_histories.items():
    plt.plot(range(1, len(history) + 1), history, label=f"{n_hidden_layers}")

plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()

# TRAINING PHASE

def scaled_sigmoid(x):
    return 100 / (1 + K.exp(-x))

def build_model():
    # modelo equivalente usando a API funcional
    input_tensor = tf.keras.Input(shape=(train_data.shape[1],))
    x = layers.Dense(n_hidden, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_tensor) # hidden layer 1
    #x = layers.Dense(n_hidden, activation='relu')(x) # hidden layer 2
    #x = layers.Dense(n_hidden, activation='relu')(x) # hidden layer 3
    #x = layers.Dense(n_hidden, activation='relu')(x) # hidden layer 4
    #x = layers.Dense(n_hidden, activation='relu')(x) # hidden layer 5
    #x = layers.Dense(n_hidden, activation='relu')(x) # hidden layer 6
    output_tensor = layers.Dense(1, activation = scaled_sigmoid)(x)
    model = tf.keras.Model(input_tensor, output_tensor)
    #model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.RMSprop(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.Adamax(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mse'])
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss = mse, metrics = ['mae'])
    #model.compile(tf.keras.optimizers.Adadelta(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.Adagrad(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.Nadam(learning_rate=lr), loss='mse', metrics=['mae'])
    #model.compile(tf.keras.optimizers.Ftrl(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

model = build_model()
model.summary()

# Train the model on the entire training data - REAL training phase !!!
real_train = model.fit(train_data, train_targets, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Print available keys in "history.history"
print("Available keys in history.history: ", real_train.history.keys())

# Keep a record of how well the model did at each epoch at REAL training phase
# save the per-epoch validation score log:
loss_hist = real_train.history['loss']
mae_hist = real_train.history['mae']
    
# Evaluate the model on the test data
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_pred = model.predict(test_data)
# Converting the array to float64
test_pred64 = test_pred.astype(np.float64)
print("Depois da conversão:", test_pred64.dtype)

# Performance on the test data
print('Test MAE score', test_mae_score) 
print('Test MSE score', test_mse_score) 

# 1) Visualization of the learning curves during training
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_hist) + 1), loss_hist)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(mae_hist) + 1), mae_hist)
plt.title('Training Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

# 2) Compare the model's predictions with the actual labels on a chart
plt.figure(figsize=(8, 8))
plt.scatter(test_targets, test_pred64)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.plot([-100, 100], [-100, 100], color='red')
plt.show()

# Test MAE score 2.25565242767334
# Test MSE score 8.98123836517334
