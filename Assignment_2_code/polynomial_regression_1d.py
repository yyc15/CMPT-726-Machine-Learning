import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt


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


(countries, features, values) = a2.load_unicef_data()

targets = values[:, 1]
x = values[:, 7:]
# x = a2.normalize_data(x)


N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_losses = []
test_losses = []
for i in range(0, 8):
    features = polynomial_features(x_train[:, i], 3)
    w = least_squares(features, t_train)
    train_loss = avg_loss(features, t_train, w)
    train_losses.append(train_loss)

    features = polynomial_features(x_test[:, i], 3)
    test_loss = avg_loss(features, t_test, w)
    test_losses.append(test_loss)

plt.figure(figsize=(10, 5))
plt.bar([i + 0.2 for i in np.arange(8, len(train_losses) + 8)], train_losses, color='b', width=0.4)
plt.bar([i - 0.2 for i in np.arange(8, len(test_losses) + 8)], test_losses, color='r', width=0.4)
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Training error and Test error (in RMS error) versus feature degree - not normalised')
plt.xlabel('Feature')
plt.show()

for i in (3, 4, 5):
    # Select a single feature.
    x_train = x[0:N_TRAIN, i]
    t_train = targets[0:N_TRAIN]
    x_test = x[N_TRAIN:, i]
    t_test = targets[N_TRAIN:]

    features = polynomial_features(x_train, 3)
    w = least_squares(features, t_train)
    train_loss = avg_loss(features, t_train, w)
    train_losses.append(train_loss)

    features = polynomial_features(x_test, 3)
    test_loss = avg_loss(features, t_test, w)
    test_losses.append(test_loss)

    x_ev = np.linspace(min(x_train).item(), max(x_train).item(), num=500)
    values = x_ev.reshape(500, 1)
    values = polynomial_features(values, 3)

    y_ev = np.random.random_sample(x_ev.shape)
    y_ev = values * w

    plt.figure(figsize=(10, 5))
    plt.plot(x_ev, y_ev, 'r.-')
    plt.plot(x_train, t_train, 'bo')
    plt.plot(x_test, t_test, 'go')
    plt.title('A visualization of the fits for degree 3 polynomials for features ' + str(
        i + 8) + ' with training data points and testing data points')
    plt.show()
    