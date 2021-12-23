import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def polynomial_features(x, order):
    features = np.hstack([np.power(x,i) for i in range(1, order+1)])
    reform_feature = np.concatenate((features, np.ones((features.shape[0], 1))), axis=1)
    return reform_feature


def least_squares(x, y, k):
    xTx = x.T.dot(x)
    xTx += k * np.identity(xTx.shape[0])
    xTx_inv = np.linalg.inv(xTx)
    w = xTx_inv.dot(x.T.dot(y))
    return w


def avg_loss(x, y, w):
    y_hat = x.dot(w)
    loss = np.sqrt(np.mean(np.power((y - y_hat),2)))
    return loss

(countries, features, values) = a2.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a2.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:, :]
t_test = targets[N_TRAIN:]


k_set = (0, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)

degree = 2
average_train_losses = []
average_test_losses = []
for k in k_set:

    start_ranges = [i * 10 for i in range(10)] # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    end_ranges = [i * 10 for i in np.arange(1,11,1)] # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    test_losses = []
    train_losses = []

    for i in range(10):
        start_index = start_ranges[i]
        end_index = end_ranges[i]
        
        x_train_validation = np.concatenate((x_train[:start_index], x_train[end_index:]))
        t_train_validation = np.concatenate((t_train[:start_index], t_train[end_index:]))
        x_test_validation = x_train[start_index:end_index]
        t_test_validation = t_train[start_index:end_index]        
        
        features = polynomial_features(x_train_validation, degree)
        w = least_squares(features, t_train_validation, k)
#         train_loss = avg_loss(features, t_train_validation, w)
#         train_losses.append(train_loss)

        features = polynomial_features(x_test_validation, degree)
        test_loss = avg_loss(features, t_test_validation, w)
        test_losses.append(test_loss)
        
#     average_train_losses.append(np.average(train_losses))
    average_test_losses.append(np.average(test_losses))
    

df = pd.DataFrame([average_test_losses], columns = k_set , index= ['Average validation losses'])
df.reset_index(inplace=False)
display(df)
plt.figure(figsize=(10, 5))
plt.semilogx(k_set, average_test_losses)
plt.title('Average validation set error versus λ')
plt.show()
