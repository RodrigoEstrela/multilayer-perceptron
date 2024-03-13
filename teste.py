import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 
# read the dataset
df = pd.read_csv('Downloads/data.csv')
 
# get the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
 
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

print(f'train size {len(X_train)} test size {len(X_test)}')

pd.concat([pd.concat([X_train, y_train], axis=1)]).to_csv('train.csv')
pd.concat([pd.concat([X_test, y_test], axis=1)]).to_csv('test.csv')
