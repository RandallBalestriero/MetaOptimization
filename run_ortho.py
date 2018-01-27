from pylab import *
import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-4]
lr      = float(sys.argv[-3])

if(int(sys.argv[-2])==0):
	m = smallCNN
	m_name = 'smallCNN'
elif(int(sys.argv[-2])==1):
	m = largeCNN
	m_name = 'largeCNN'

elif(int(sys.argv[-2])==2):
        m = resnet_large
        m_name = 'resnetLarge'



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
 


for kk in xrange(10,20):
	all_train = []
	all_test  = []
	all_W     = []
	for coeff in linspace(0,2,10):
        	name = DATASET+'_'+m_name+'_lr'+str(lr)+'_run'+str(kk)+'_c'+str(coeff)
		model1  = DNNClassifier(input_shape,m(1,c,g=0,p=2),lr=lr,n=int(sys.argv[-1]),Q=coeff)
		train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
		all_train.append(train_loss)
		all_test.append(test_loss)
		all_W.append(W)
		f = open('../../SAVE/QUADRATIC/'+name,'wb')
		cPickle.dump([all_train,all_test,all_W],f)
		f.close()




