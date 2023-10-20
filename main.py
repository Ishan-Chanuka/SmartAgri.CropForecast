import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('crop_trends.csv')

df['Month'] = df['Month-Year'].apply(lambda x: int(x.split('-')[0]))
df['Year'] = df['Month-Year'].apply(lambda x: int(x.split('-')[1]))

# print(df.head())

encoder = OneHotEncoder(drop='first')
encoded_yield = encoder.fit_transform(df[['Yield Type']])
encoded_yield_df = pd.DataFrame(encoded_yield.toarray(), columns=encoder.get_feature_names_out(['Yield Type']))

df_encoded = pd.concat([df, encoded_yield_df], axis=1)

# print(df_encoded.head())

df_encoded = df_encoded.drop(columns=['Month-Year', 'Yield Type'])
# print(df_encoded.head())

X = df_encoded.drop(columns=['Trend Score'])
y = df_encoded['Trend Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.head())

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)

# print('Random Forest MAE: ', rf_mae)
# print('Random Forest MSE: ', rf_mse)

# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(encoder, 'encoder.pkl')

with open('x_train_columns.txt', 'w') as f:
    for column in X_train.columns:
        f.write("%s\n" % column)
