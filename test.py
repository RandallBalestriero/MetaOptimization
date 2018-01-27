from pylab import *
import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

batch_size = 50
DATASET = 'CIFAR'


if(DATASET=='MNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        input_shape   = (batch_size,28,28,1)
	x_train = transpose(x_train,[0,2,3,1])
	x_test  = transpose(x_test,[0,2,3,1])
	c = 10
        n_epochs = 150

elif(DATASET == 'CIFAR'):
        batch_size = 50
        TRAIN,TEST = load_cifar(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150

elif(DATASET == 'CIFAR100'):
	batch_size = 100
        TRAIN,TEST = load_cifar100(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=100
        n_epochs = 200

elif(DATASET=='IMAGE'):
	batch_size=200
        x,y           = load_imagenet()
	x = x.astype('float32')
	y = y.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=20000,stratify=y)
        input_shape   = (batch_size,64,64,3)
	c=200
        n_epochs = 200

else:
        batch_size = 50
        TRAIN,TEST = load_svhn()
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
	c=10
        n_epochs = 150




x_train          -= x_train.mean((1,2,3),keepdims=True)
x_train          /= abs(x_train).max((1,2,3),keepdims=True)
x_test           -= x_test.mean((1,2,3),keepdims=True)
x_test           /= abs(x_test).max((1,2,3),keepdims=True)
x_train           = x_train.astype('float32')
x_test            = x_test.astype('float32')
y_train           = array(y_train).astype('int32')
y_test            = array(y_test).astype('int32')
 


accu_nothing = []
accu_diagonal = []

n_epochs = 5

for i in xrange(1):
	model1  = DNNClassifier(input_shape,smallCNN(1),optimizer=meta_sgd,n=2,Q=0,lr=0.01)
#	K=model1.get_templates(x_train[:input_shape[0]])
#        imshow(K[0,0,:,:,0])
#        show()
	train_loss,test_loss = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
	accu_nothing.append(test_loss)

#K=model1.get_templates(x_train[:input_shape[0]])
#imshow(K[0,0,:,:,0])
#show()



subplot(121)
for l in accu_nothing:
	plot(l,color='b')
subplot(122)
for l in accu_diagonal:
	plot(l,color='b')
show()



