import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
import matplotlib.pyplot as plt
import random


# Loading the dataset
df = pd.read_csv('housing.csv')

# Splitting the data into features (X) and target (Y)
X = df.drop("median_house_value", axis=1)
Y = df["median_house_value"]

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Preprocessing

# Identifying numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = ["ocean_proximity"]

# Creating a pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handling missing values by imputing the median
    ('scaler', StandardScaler())  # Scaling the numeric features
])

# Creating a pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handling missing values by imputing a constant value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding the categorical features
])

# Combining the numeric and categorical transformers into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # Applying numeric transformer to numeric features
        ('cat', categorical_transformer, categorical_features)  # Applying categorical transformer to categorical features
    ]
)


# Applying the preprocessing pipeline to the training and testing data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Building the neural network model
model = keras.Sequential([
    keras.layers.Dense(20, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 20 neurons and ReLU activation
    keras.layers.Dropout(0.2),  # Dropout layer to prevent overfitting
    keras.layers.Dense(16, activation='relu'),  # Hidden layer with 16 neurons and ReLU activation
    keras.layers.Dense(12, activation='relu'),  # Hidden layer with 12 neurons and ReLU activation
    keras.layers.Dense(1)  # Output layer with 1 neuron (for regression)
])


# Defining parameters to be used
epoch = 100
batch_size = 128
learning_rate = 0.001

# Setting up the optimizer
optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

# Compiling the model
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

# Training the model
history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2)

# Making predictions on the test set
output_prediction = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(Y_test, output_prediction)
mae = mean_absolute_error(Y_test, output_prediction)
r2 = r2_score(Y_test, output_prediction)

# Printing the parameters used for testing
print("\nParameters Used for testing")
print(f"Epochs: {epoch}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Optimizer: {optimizer.__class__.__name__}\n") 

# Printing the evaluation metrics
print(f"Mean Squared Error (MSE)= {mse:.2f}")
print(f"Mean Absolute Error (MAE)= {mae:.2f}")
print(f"R-squared (R2)= {r2:.4f}\n")

# Print the actual and predicted values for the random samples
random_indexes = random.sample(range(len(Y_test)), 10)
for i, index in enumerate(random_indexes):
    actual = Y_test.iloc[index]
    pred = output_prediction[index]
    print(f"Example Test {i+1}: Actual Price= ${actual:.2f}, Predicted Price= ${pred[0]:.2f}")
    
# Plotting the training history

# Setting the figure size
plt.figure(figsize=(12, 5))

# Plotting Mean Squared Error (MSE) vs Epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('MSE VS Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

# Plotting Mean Absolute Error (MAE) vs Epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE VS Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

# Adjusting layout and saving the plot
plt.tight_layout()
plt.savefig('mse_mae_epoch_comparisons.png')  
plt.show()

# Regression figure
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, output_prediction, alpha=0.7)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual VS Predicted Prices')
plt.savefig('actual_&_predicted_price.png')  
plt.legend()
plt.show()
