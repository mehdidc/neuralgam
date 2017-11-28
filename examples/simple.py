import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from neuralgam import build_fully_connected_gam
from keras.optimizers import Adam, SGD

data = load_boston()
X = data.data
y = data.target
y = np.expand_dims(y, axis=1)
names = data.feature_names

nb_train = int(len(X) * 0.8)
X_train = X[0:nb_train]
y_train = y[0:nb_train]
X_test = X[nb_train:]
y_test = y[nb_train:]

X_mu = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)
y_mu = y_train.mean()
y_std = y_train.std()

X_train = (X_train - X_mu) / X_std
y_train = (y_train - y_mu) / y_std
X_test = (X_test - X_mu) / X_std
y_test = (y_test - y_mu) / y_std

model = build_fully_connected_gam(
    input_dim=X.shape[1], 
    output_dim=1, 
    hidden_units=[100],
    hidden_activation='tanh',
    output_activation='linear',
)
model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
model.fit(X_train, y_train, verbose=0, epochs=200)

err_train = np.abs(model.predict(X_train) - y_train).sum(axis=1).mean()
err_test = np.abs(model.predict(X_test) - y_test).sum(axis=1).mean()
print('MAE train : {:.4f}'.format(err_train))
print('MAE test  : {:.4f}'.format(err_test))
nb = 1000
xfull = np.zeros((nb, X.shape[1]))
for i in range(X.shape[1]):
    xfull[:, i] = np.linspace(X[:, i].min(), X[:, i].max(), nb)
yfull = model.predict_components(xfull)
for i in range(X.shape[1]):
    plt.plot(xfull[:, i], yfull[i][:, 0] * y_std + y_mu)
    plt.title(names[i])
    plt.show()
