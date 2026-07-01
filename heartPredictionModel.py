import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#seed random number generator
keras.utils.set_random_seed(42)

#Read in heat disease data into Panda dataframe
df = pd.read_csv("heart.csv")

#Sanity check
print(f"Sanity check. Data frame shape (rows, columns): {df.shape}")
print(f"Heart Disease (1) vs No Heart Disease (0): {df.target.value_counts(normalize=True, dropna=False)}")


#Preprocessing
#Place categorical variables and numerical variables in a list
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
numerics = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']


#one-hot encode categorical variables and normalize numerical variables as neural networks require all inputs to be numeric
df = pd.get_dummies(df, columns = categorical_variables, dtype = int)

print(f"Subset of data in the table shown below")
print(f"\n: {df.head}")
input("Press Enter to continue")


#split data into 80% training and 20% test data. Will normalize only training data separately from test data in order not to influence training data
test_df = df.sample(frac=0.2, random_state=42)
train_df = df.drop(test_df.index)


print(f"Rows & Columns of test set after splitting: {test_df.shape}")
print(f"Rosws & Columns of training set after splitting {train_df.shape}")

#calculate mean and standard deviation for normalization/standardization purposes
means = train_df[numerics].mean()
sd = train_df[numerics].std()

#normalize the data
train_df[numerics] = (train_df[numerics] - means)/sd
test_df[numerics] = (test_df[numerics] - means)/sd


#convert dataframes to Numpy arrays - easiest way to feed data to Keras/Tensorflow
train = train_df.to_numpy()
test = test_df.to_numpy()

#Separate features X from dependent variable Y. Y is on column 6 of the table
train_X = np.delete(train,6, axis=1)
test_X = np.delete(test, 6, axis=1)

print(f"Rows & Columns of dependent variables only in Training data set: {train_X.shape}")
print(f"Rows & Columns of dependent variables only in Test data set: {test_X.shape}")

#select 6th column
train_Y = train[:, 6]
test_Y = test[:, 6]

print(train_Y[:5])

print(f"Rows of Target variable in Training data set: {train_Y.shape}")
print(f"Rows of Target variable in Test data set: {test_Y.shape}")
input("Press enter to continue")


###   Building the Model  ####

#Define the model
num_columns = train_X.shape[1]

#Define the input layer
input = keras.Input(shape = (num_columns,))

#Feed input vector to hidden layer. Using 16 neurons for the hidden layer
h = keras.layers.Dense(16, activation = "relu", name = "Hidden")(input)

#Feed the output of the hidden layer to the output layer
output = keras.layers.Dense(1, activation = "sigmoid", name = "Output")(h)

#Input and Output Pair form the model
model = keras.Model(input, output)

print(f"Here's the model summary")
print(f"{model.summary()}")

#keras.utils.plot_model(model, show_shapes=True)

#Set optimization parameters. Indicate loss function, optimizer, and metrics
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Train the model
history = model.fit(train_X,                #array with input X columns 
                    train_Y,                #array with input Y column             
                    epochs = 235,            #number of epochs to run
                    batch_size = 32,         #number of samples per batch
                    verbose = 1,             #verbosity during training 
                    validation_split=0.2)   #percent of data to be used for validation. 20% in this case    


#Evaluate model
print(f"Loss vs Accuracy evaluation: {model.evaluate(test_X, test_Y)}")