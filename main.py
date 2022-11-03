import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('cars.csv')


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
df['Number of doors'] = labelencoder_X.fit_transform(df['Number of doors'])
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df[['Drive type']]).toarray())
df = df.drop(['Drive type'], axis=1)

for i in range(3):
    df.insert(3 + i,i, enc_df[i])

X = df.iloc[:, 1:-1].values
Y = df.iloc[:, 9].values
print(Y)

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_Train, Y_Train)

Y_Pred = regressor.predict(X_Test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(Y_Test, Y_Pred)
mae = mean_absolute_error(Y_Test,Y_Pred)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)
print("Accuracy percentage : ", 100 - mae*100 / np.average(Y_Test))

plt.scatter(reg.predict(X_Test), reg.predict(X_Test) - Y_Test, color = "blue", s = 10, label = 'Test data')
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()
    
