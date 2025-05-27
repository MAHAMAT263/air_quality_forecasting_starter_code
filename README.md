## Air Quality Forecasting Using RNN-Based Deep Learning Models

# 1. Introduction
Air pollution, specifically PM2.5 concentration, has become a serious environmental concern due to its adverse effects on human health and the environment. This project aims to forecast future PM2.5 levels using historical air quality and weather data. We approach this as a time series regression problem, leveraging deep learning techniques particularly Recurrent Neural Networks (RNNs) such as LSTM and GRU known for their effectiveness in sequence modeling.

# 2. Data Exploration & Preprocessing
The dataset consists of timestamped air quality metrics including PM2.5, temperature, pressure, wind direction, and other environmental features.
Key Steps:
Missing Data Handling: Forward filling for time continuity.


Datetime Feature Engineering: Extracted hour, day, and month from timestamps.


Feature Scaling: Used StandardScaler on features to standardize the input range.


Sequence Creation: Transformed time-series data into sequences of length 24 (sliding window) for input to RNNs.


Test Set Padding: Padded the start of the test set with zeros to ensure correct shape for prediction.
Model architecture
Compile
Fit
Plot
Predict using the test



# 3. Model Design
Best Model: LSTM
We obtained the best performance using a LSTM model, which allows the network to learn both past and future temporal dependencies.

model = Sequential([
    # First GRU layer with return_sequences=True to feed the next GRU layer
    LSTM(16, return_sequences=True, activation='relu', kernel_regularizer=l2(1e-3), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),


    # Second GRU layer, outputs sequence reduced, no return_sequences
    LSTM(8, kernel_regularizer=l2(1e-3), activation='relu'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    # Final dense layer for regression output
    Dense(1)
])


Optimizer: Adam


Loss: Mean Squared Error


Regularization: L2 (0.001)


Dropout: To reduce overfitting


We also tried GRU, LSTM, TCN, and Bidirectional LSTM had the most balanced performance.



# 4. Experiment Table
Experiment
Model
Parameters
Train RMSE
Val RMSE
Notes












1
LSTM
LR=0.001, 2 layers, Dropout=0.3, L2=1e-3
0.59
0.67
Best overall performance
2
GRU
LR=0.001, 2 layers, Dropout=0.3, L2=1e-3
0.6
0.8
Slightly lower than LSTM
3
Bidirectional LSTM
LR=0.001, units=32+16, Dropout=0.3, L2=0.001, Adam
0.9
1.4
The last
5
TCN
Dilations=[1,2,4,8], LR=0.001, Dropout=0.2
0.8
0.9
A bit good

# 5. Results and Evaluation
The  LSTM model showed the lowest validation RMSE, indicating better generalization to unseen data. Although GRU, they underperformed in long-range temporal pattern learning. The model successfully learned the short-term and seasonal fluctuations of PM2.5.
Final Evaluation:
Train RMSE: 0.59


Validation RMSE: 0.67


Test Predictions: Submitted in correct format with row ID and predicted pm2.5.


# 6. Conclusion
We built and compared multiple deep learning models to forecast PM2.5 air quality levels. The LSTM outperformed other architectures by effectively learning both forward and backward temporal dependencies. With better hyperparameter tuning and more feature engineering, performance could further improve.

