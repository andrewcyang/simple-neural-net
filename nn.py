# simple demo neural net for classification

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# read in file, assumes label column is at end, and one-hot encodes label
df = pd.read_csv('filename')
df = pd.get_dummies(data=df,
                    columns=['labelname'])
num_features = 5
num_outputs  = 4

X = df.iloc[:, :num_features]
y = df.iloc[:, num_features:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

X_train = np.array(X_train, dtype='float32')
X_test  = np.array(X_test, dtype='float32')
y_train = np.array(y_train, dtype='float32')
y_test  = np.array(y_test, dtype='float32')

X_data   = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32)

hidden_nodes = 12
iterations = 500
interval = 50
alpha = 0.001 # learning rate

# two hidden layers
W1 = tf.Variable(tf.random_normal(shape=[num_features, hidden_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_nodes]))
W2 = tf.Variable(tf.random_normal(shape=[hidden_nodes, hidden_nodes]))
b2 = tf.Variable(tf.random_normal(shape=[hidden_nodes]))
W3 = tf.Variable(tf.random_normal(shape=[hidden_nodes, num_outputs]))
b3 = tf.Variable(tf.random_normal(shape=[num_outputs]))

hidden_in    = tf.add(tf.matmul(X_data, W1), b1)
hidden_out   = tf.nn.relu(hidden_in)

hidden_in_2  = tf.add(tf.matmul(hidden_out, W2), b2)
hidden_out_2 = tf.nn.relu(hidden_in_2)

output_in    = tf.add(tf.matmul(hidden_out_2, W3), b3)
output_out   = tf.nn.softmax(output_in)

loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(output_out), axis=0))
grad_optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train neural net
for i in range(iterations):
    sess.run(grad_optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if (i+1) % interval == 0:
        correct = 0
        for j in range(len(X_test)):
            pred = np.argmax(np.rint(sess.run(output_out, feed_dict={X_data: [X_test[j]]})))
            actual = np.argmax(y_test[j])
            if pred == actual:
                correct += 1
        acc = correct / len(X_test)
        _loss = sess.run(loss, feed_dict={X_data: X_train, y_target: y_train})
        print('Iteration', i+1, '====>', 'Loss: {:.3f}'.format(_loss), 'Accuracy: {:.3f}'.format(acc))

sess.close()
