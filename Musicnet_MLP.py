# from https://homes.cs.washington.edu/~thickstn/mlp.html

import numpy as np                                       # fast vectors and matrices
# import matplotlib.pyplot as plt                          # plotting
from scipy.fftpack import fft

from intervaltree import Interval,IntervalTree

from time import time

import tensorflow as tf

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# %matplotlib inline


d = 2048        # input dimensions
m = 128         # number of notes
fs = 44100      # samples/second
features = 0    # first element of (X,Y) data tuple
labels = 1      # second element of (X,Y) data tuple


# Warning: the full dataset is over 40GB. Make sure you have enough RAM!
# This can take a few minutes to load
train_data = dict(np.load(open('../musicnet.npz','rb')))


# split our the test set
test_data = dict()
for id in (2303,2382,1819): # test set
    test_data[str(id)] = train_data.pop(str(id))
    
train_ids = train_data.keys()
test_ids = test_data.keys()
    
print len(train_data)
print len(test_data)


# create the test set
Xtest = np.empty([3*7500,d])
Ytest = np.zeros([3*7500,m])
for i in range(len(test_ids)):
    for j in range(7500):
        index = fs+j*512 # start from one second to give us some wiggle room for larger segments
        Xtest[7500*i + j] = test_data[test_ids[i]][features][index:index+d]
        
        # label stuff that's on in the center of the window
        for label in test_data[test_ids[i]][labels][index+d/2]:
            Ytest[7500*i + j,label.data[1]] = 1



tf.reset_default_graph()
tf.set_random_seed(999)

k = 500

x = tf.placeholder(tf.float32, shape=[None,d])
y_ = tf.placeholder(tf.float32, shape=[None, m])

wscale = .001
w = tf.Variable(wscale*tf.random_normal([d,k],seed=999))
beta = tf.Variable(wscale*tf.random_normal([k,m],seed=999))

zx = tf.nn.relu(tf.matmul(x,w))
y = tf.matmul(zx,beta)
R = tf.nn.l2_loss(w) + tf.nn.l2_loss(beta)
L = tf.reduce_mean(tf.nn.l2_loss(y-y_)) #+ 1*R

init = tf.initialize_all_variables()


square_error = []
average_precision = []
sess = tf.Session()
sess.run(init)


lr = .0001
opt = tf.train.GradientDescentOptimizer(lr)
train_step = opt.minimize(L)
Xmb = np.empty([len(train_data),d])
np.random.seed(999)
start = time()
print 'iter\tsquare_loss\tavg_precision\ttime'
for i in xrange(250):
    if i % 1000 == 0 and (i != 0 or len(square_error) == 0):
        square_error.append(sess.run(L, feed_dict={x: Xtest, y_: Ytest})/Xtest.shape[0])
        
        Yhattestbase = sess.run(y,feed_dict={x: Xtest})
        yflat = Ytest.reshape(Ytest.shape[0]*Ytest.shape[1])
        yhatflat = Yhattestbase.reshape(Yhattestbase.shape[0]*Yhattestbase.shape[1])
        average_precision.append(average_precision_score(yflat, yhatflat))
        
        if i % 10000 == 0:
            end = time()
            print i,'\t', round(square_error[-1],8),\
                    '\t', round(average_precision[-1],8),\
                    '\t', round(end-start,8)
            start = time()
    
    Ymb = np.zeros([len(train_data),m])
    for j in range(len(train_ids)):
        s = np.random.randint(d/2,len(train_data[train_ids[j]][features])-d/2)
        Xmb[j] = train_data[train_ids[j]][features][s-d/2:s+d/2]
        for label in train_data[train_ids[j]][labels][s]:
            Ymb[j,label.data[1]] = 1
    
    sess.run(train_step, feed_dict={x: Xmb, y_: Ymb})


print(square_error)
# fig, ((ax1, ax2)) = plt.subplots(1, 2)
# fig.set_figwidth(12)
# fig.set_figheight(5)
# ax1.set_title('average precision')
# ax1.plot(average_precision[1:],color='g')
# ax2.set_title('square loss')
# ax2.plot(square_error[1:],color='g')



# window = 2048
# f, ax = plt.subplots(20,2, sharey=False)
# f.set_figheight(20)
# f.set_figwidth(20)
# for i in range(20):
#     ax[i,0].plot(w.eval(session=sess)[:,i], color=(41/255.,104/255.,168/255.))
#     ax[i,0].set_xlim([0,d])
#     ax[i,0].set_xticklabels([])
#     ax[i,0].set_yticklabels([])
#     ax[i,1].plot(np.abs(fft(w.eval(session=sess)[:,0+i]))[0:200], color=(41/255.,104/255.,168/255.))
#     ax[i,1].set_xticklabels([])
#     ax[i,1].set_yticklabels([])

# for i in range(ax.shape[0]):
#     for j in range(ax.shape[1]):
#         ax[i,j].set_xticks([])
#         ax[i,j].set_yticks([])