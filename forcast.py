# suppress the possible cpu architecture warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.interpolate import spline

df = pd.read_csv('btc_history.csv', parse_dates=['Date'])
#df.index = pd.to_datetime(df.index)

num_periods = 20
f_horizon = 1

df = df[['Date','Close']]

org_num = len(df)
sample_per_period = org_num / num_periods
spp = int(sample_per_period);

# print (org_num)
df = df.set_index('Date')
#df = df.resample('5D').mean()

ts = df.reindex(index=df.index[::-1])

tsnp = np.array(ts)

x_data = tsnp[:(len(tsnp)-(len(tsnp) % num_periods ))]
x_batches = x_data.reshape(-1, num_periods, 1)
y_data = tsnp[1:(len(tsnp)-(len(tsnp) % num_periods)) +  f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)

each_batch_contains = len(x_batches)
print(len(x_batches))
print("x_batches.shape")
print(x_batches.shape)
print("x_batches")
print(x_batches)
print("y_batches")
print(y_batches)
print("y_batches.shape")
print(y_batches.shape)


def test_data(series, forecast, nums):
    test_x_setup = series[-(num_periods + forecast):]
    testx = test_x_setup[:nums].reshape(-1, nums, 1)
    testy = series[-nums:].reshape(-1, nums, 1)
    return testx, testy


x_test, y_test = test_data(tsnp, f_horizon, num_periods)

print("ytest")
print(x_test.shape)
print(y_test)
print("xtest")
print(x_test.shape)
print(x_test)

###################################################################################
tf.reset_default_graph()
###################################################################################
inputs = 1  # number of inputs submitted
hidden = 100 # number of neurons
output = 1 # number of output vectors

x = tf.placeholder(tf.float32, [None, num_periods, inputs]) # variables
y = tf.placeholder(tf.float32, [None, num_periods, output]) #

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.elu) # RNN hyper activate function
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32) # dynamic

learning_rate = 0.0001   # small learning rate so we wont overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden]) # change into tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)  # nn layer type is dense
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output]) # shape of results

loss = tf.reduce_sum(tf.square(outputs - y)) # cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss) # train the result of the application of cost function
###################################################################################
init = tf.global_variables_initializer()    # initialize all the variables
###################################################################################
epoch = 1000    # number of iterations or training cycles, includes both the feedforward and back propogations

with tf.Session() as sess :

    init.run()
    for ep in range(epoch):
        sess.run(training_op, feed_dict={x: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={x: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)

    y_pred = sess.run(outputs, feed_dict={x: x_test})

    print("ypred")
    print(y_pred)


plt.figure(figsize=(20, 12))
plt.title("Currency: BTC")

ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_right()

#plt.ylim(200, 90)
plt.xlim(1000, 1750)

plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

yytest = pd.Series(np.ravel(y_test))

yypred = pd.Series(np.ravel(y_pred))

xold = len(tsnp) - num_periods + yytest.index.get_values()

plt.axvline(x=len(tsnp) - num_periods)
#xnew = np.linspace(xold.min(),xold.max(),100)

#yytest_smooth = spline(xold, yytest, xnew)
#yypred_smooth = spline(xold, yypred, xnew)

plt.plot(xold, yytest, "gx", markersize=5, label="Actual")
plt.plot(xold, yypred, "r.", markersize=5, label="Forecast")

plt.plot(tsnp, label="All Actual", alpha=0.5)

plt.legend(loc="upper left")
plt.xlabel("Time Seq")

ax.annotate('Recurrent Neural Net Platform for Crypto-Forecast RnnPCF\n'
            'Author: Milad Kiaee, All rights reserved'
            '\n Powered by: Google Tensorflow',
            xy=(80, 10), xycoords='figure pixels')

plt.savefig("tmp.png", bbox_inches="tight")