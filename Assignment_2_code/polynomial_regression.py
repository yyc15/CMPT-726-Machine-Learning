import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

targets = values[:, 1]
x = values[:, 7:]

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


def polynomial_features(x, order):
    features = np.hstack([np.power(x, i) for i in range(1, order + 1)])
    return features


def least_squares(x, y):
    xTx = x.T.dot(x)
    xTx_inv = np.linalg.inv(xTx)
    w = xTx_inv.dot(x.T.dot(y))
    return w


def avg_loss(x, y, w):
    y_hat = x.dot(w)
    loss = np.sqrt(np.mean(np.power((y - y_hat), 2)))
    return loss


train_losses = []
test_losses = []
for i in range(1, 9):
    features = polynomial_features(x_train, i)
    w = least_squares(features, t_train)
    train_loss = avg_loss(features, t_train, w)
    train_losses.append(train_loss)

    features = polynomial_features(x_test, i)
    test_loss = avg_loss(features, t_test, w)
    test_losses.append(test_loss)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(train_losses) + 1), train_losses, 'b')
plt.plot(np.arange(1, len(test_losses) + 1), test_losses, 'r')
plt.ylabel('RMS')
plt.semilogy()
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, without normalization')
plt.xlabel('Polynomial degree')
plt.show()

x = a2.normalize_data(x)
N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_losses = []
test_losses = []
degree = 8
for i in range(1, degree + 1):
    features = polynomial_features(x_train, i)
    w = least_squares(features, t_train)
    train_loss = avg_loss(features, t_train, w)
    train_losses.append(train_loss)

    features = polynomial_features(x_test, i)
    test_loss = avg_loss(features, t_test, w)
    test_losses.append(test_loss)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(train_losses) + 1), train_losses, 'b')
plt.plot(np.arange(1, len(test_losses) + 1), test_losses, 'r')
plt.ylabel('RMS')
plt.semilogy()
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, with normalization')
plt.xlabel('Polynomial degree')
plt.show()

