import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from loadDomains import mnist

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_h3y, w_h4y, w_h3d, w_h4d, w_oy, w_od, p_drop_input, p_drop_hidden):
    
    # Model architecture
    # h, h2 shared
    # h3y, h4y for character recognition
    # h3d, h4d for domain confusion
    
    X = dropout(X, p_drop_input)

    h = rectify(T.dot(X, w_h))
    h = dropout(h, p_drop_hidden)

    h2 = rectify(T.dot(h, w_h2))
    h2 = dropout(h2, p_drop_hidden)
    
    h3y = rectify(T.dot(h2, w_h3y))
    h3y = dropout(h3y, p_drop_hidden)

    h4y = rectify(T.dot(h3y, w_h4y))
    h4y = dropout(h4y, p_drop_hidden)

    h3d = rectify(T.dot(h2, w_h3d))
    h3d = dropout(h3d, p_drop_hidden)

    h4d = rectify(T.dot(h3d, w_h4d))
    h4d = dropout(h4d, p_drop_hidden)

    py_x = softmax(T.dot(h4y, w_oy))
    pd_x = softmax(T.dot(h4d, w_od))
    return h, h2, h3y, h4y, h3d, h4d, py_x, pd_x


# To use GPU. 
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
theano.config.floatX = 'float32'

trX,teX,trY,teY,trD,teD,trD_fake,teD_fake = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()
D = T.fmatrix()
D_random = T.fmatrix()

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_h3y = init_weights((625, 625))
w_h4y = init_weights((625, 625))
w_h3d = init_weights((625, 625))
w_h4d = init_weights((625, 625))
w_oy = init_weights((625, 10))
w_od = init_weights((625, 3))

noise_h, noise_h2, noise_h3y, noise_h4y, noise_h3d, noise_h4d, noise_py_x, noise_pd_x = model(X, w_h, w_h2, w_h3y, w_h4y, w_h3d, w_h4d, w_oy, w_od, 0., 0.)

h, h2, h3y, h4y, h3d, h4d, py_x, pd_x = model(X, w_h, w_h2, w_h3y, w_h4y, w_h3d, w_h4d, w_oy, w_od, 0., 0.)

y_x = T.argmax(py_x, axis=1)
d_x = T.argmax(pd_x, axis=1)

cost_1 = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) + T.mean(T.nnet.categorical_crossentropy(noise_pd_x, D_random)) 
params_1 = [w_h, w_h2, w_h3y, w_h4y, w_oy]
updates_1 = RMSprop(cost_1, params_1, lr=0.001)

cost_2 = T.mean(T.nnet.categorical_crossentropy(noise_pd_x, D_random)) 
params_2 = [w_h, w_h2]
updates_2 = RMSprop(cost_2, params_2, lr=0.001)

cost_3 = T.mean(T.nnet.categorical_crossentropy(noise_pd_x, D)) 
params_3 = [w_h3d, w_h4d, w_od]
updates_3 = RMSprop(cost_3, params_3, lr=0.001)

train_1 = theano.function(inputs=[X, Y, D_random], outputs=cost_1, updates=updates_1, allow_input_downcast=True)
train_2 = theano.function(inputs=[X, D_random], outputs=cost_2, updates=updates_2, allow_input_downcast=True)
train_3 = theano.function(inputs=[X, D], outputs=cost_3, updates=updates_3, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
predict_d = theano.function(inputs=[X], outputs=d_x, allow_input_downcast=True)


epochs = 100

for e in range(epochs):

    # In this step we update the theta_f and theta_y

    for i in range(1):
        for start1, end1, start2, end2 in zip(range(0, len(trX)/2, 64), range(64, len(trX)/2, 64), range(0, len(teX), 64), range(64, len(teX), 64)):
            cost_1 = train_1(trX[start1:end1,:], trY[start1:end1,:], trD_fake[start1:end1,:])
            cost_1 = train_1(trX[start1+len(trX)/2:end1+len(trX)/2,:], trY[start1+len(trX)/2:end1+len(trX)/2,:], trD_fake[start1+len(trX)/2:end1+len(trX)/2,:])
            cost_2 = train_2(teX[start2:end2,:], teD_fake[start2:end2,:])


    # In this step we update theta_d

    for i in range(1):
        for start1, end1, start2, end2 in zip(range(0, len(trX)/2, 64), range(64, len(trX)/2, 64), range(0, len(teX), 64), range(64, len(teX), 64)):
            cost_3 = train_3(trX[start1:end1,:], trD[start1:end1,:])
            cost_3 = train_3(trX[start1+len(trX)/2:end1+len(trX)/2,:], trD[start1+len(trX)/2:end1+len(trX)/2,:])
            cost_3 = train_3(teX[start2:end2,:], teD[start2:end2,:])

	print '............................................................'
	print 'accuracy on training set:',np.mean(np.argmax(trY, axis=1) == predict(trX))
	print 'accuracy on test set:',np.mean(np.argmax(teY, axis=1) == predict(teX))
	print 'accuracy on domain classification, training set:',np.mean(np.argmax(trD, axis=1) == predict_d(trX))
	print 'accuracy on domain classification, test set:',np.mean(np.argmax(teD, axis=1) == predict_d(teX))
	 









    





