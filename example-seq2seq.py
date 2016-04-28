# A basic autoencoder/translator using seq2seq


import numpy as np
import tensorflow as tf

#%matplotlib inline
#import matplotlib.pyplot as plt

import tempfile
logdir = tempfile.mkdtemp()
print logdir


from tensorflow.models.rnn import rnn_cell, seq2seq
import seq2seq_new

#tf.ops.reset_default_graph()
sess = tf.Session()




'''

The main encoder/decoder graph

'''


# max length of sentences
seq_length = 5
# num of sentences (one word per sentence) processed in each iteration 
batch_size = 64

# number of symbols used in sentences
vocab_size = 7
#encoding the symbols into 50 dim vectors
embedding_dim = 50

# rnn hidden memory size
memory_dim = 100


# encode inputs -- as many as seq_length

enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))


# seq 2 seq 

cell = rnn_cell.GRUCell(memory_dim)


#encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols,  embedding_size

dec_outputs, dec_state, enc_state = seq2seq_new.embedding_rnn_seq2seq_new(
    enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)

#print dec_outputs[0], len(dec_outputs), tf.shape(dec_state)
print '** enc memory ', enc_state.get_shape()
print '** dec memory ', dec_state.get_shape()

# Objective 1

# loss function - mean cross entropy across sequence
loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)


learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)


# Objective 2

# Set model weights

W = tf.Variable(tf.random_normal([100, 1], stddev=0.35),
                      name="weights")
b = tf.Variable(tf.zeros([1]), name="biases")


# Construct a linear model
activation = tf.add(tf.matmul(enc_state,W), b)

Ynum = tf.placeholder(tf.float32)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Ynum, 2))/(2*memory_dim) 
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate)
train_op2 = optimizer2.minimize(cost) #Gradient descent


#initialize
sess.run(tf.initialize_all_variables())

# train


def train_seq2seq(batch_size):
	# generate dummy data
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]

    #print X

    Y = X[:]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    #print X.shape, Y.shape

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t



num_iterations = 10	

for t in range(num_iterations):
    loss_t = train_seq2seq(batch_size)
    print 'loss = ', loss_t



def train_price(batch_size):
	# generate dummy data
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]

    sumX = np.count_nonzero(X)
    #print 'sumX is ', sumX
    X = np.array(X).T

    #print X.shape, Y.shape

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({Ynum: sumX})

    _, loss_t = sess.run([train_op2, cost], feed_dict)
    return loss_t

num_iterations = 100

for t in range(num_iterations):
    loss_t = train_price(batch_size)
    print 'loss = ', loss_t


sess.close()



