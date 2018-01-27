import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

DATASET = 'CIFAR100'
#models = ['smallCNN']
#lrs = ['0.0001','0.0005','0.001']
models = ['largeCNN']
lrs = ['0.0005']
C= linspace(0,2,15)
C=(C*100).astype('int32').astype('float32')/100.0
print C

def load_files(DATASET,model,lr):
	"""Returns [array of shape (C,RUNS,BATCHES),(C,RUNS,EPOCHS)]"""
	all_train = []
        all_test  = []
	Cs        = []
	files     = sort(glob.glob('../../SAVE/QUADRATIC/'+DATASET+'*'+model+'_lr'+lr+'_run0_c*'))
	for f,cc in zip(files,xrange(len(files))):
		trainc = []
		testc  = []
                Cs.append(float(f.split('c')[-1][:4]))
		subfiles = glob.glob(f.replace('run0','run*'))
		print subfiles
		print int(subfiles[0].split('run')[1].split('_')[0])
		if(not DATASET=='CIFAR100'):
			lists = [fff for fff in subfiles if(int(fff.split('run')[1].split('_')[0])>9)]
		else:
                        lists = subfiles
                for ff in lists:
			print ff
			fff = open(ff,'rb')
                        content = cPickle.load(fff)
			train = content[0]
			test  = content[1]
                        fff.close()
			print shape(train),shape(test)
                        trainc.append(train[cc])#find(Cs[-1]==C)[0]])
                        testc.append(test[cc])#find(Cs[-1]==C)[0]])
		all_train.append(asarray(trainc))
                all_test.append(asarray(testc))
		print shape(all_train[-1])
	return all_train,all_test,Cs


def compute_mean_std_max(data):
	return asarray([d[:,-3:].mean() for d in data]),asarray([d[:,-3:].mean(1).std() for d in data]),asarray([d.max() for d in data])



for model in models:
	if(1):#len(lrs)>1):
		all_train = []
		all_test  = []
		figure(figsize=(18,8))
		cpt=1
		for lr in lrs:
			train,test,Cs = load_files(DATASET,model,lr)
			all_train.append([d.mean(0) for d in train])
			all_test.append([d.mean(0) for d in test])
			#
			dmean,dstd,dmax = compute_mean_std_max(test)
			print dmean
                	subplot(1,len(lrs),cpt)
                	plot(Cs,100*dmax,'bo')
                	plot(Cs,100*dmean,'ko')
                	fill_between(Cs,100*dmean+100*dstd,100*dmean-100*dstd,alpha=0.5,facecolor='gray')
                        title('Learning Rate:'+lr,fontsize=20)
                	xlabel(r'$\gamma$',fontsize=19)
                	if(lr==lrs[0]):
                	        ylabel('Test Accuracy',fontsize=21)
                	cpt+=1
        	suptitle(DATASET+' '+model,fontsize=18)
#		savefig(DATASET+'_'+model+'_histo.png')
#		close()
		figure(figsize=(18,8))
		cpt=1
                for lr,i in zip(lrs,xrange(len(lrs))):
			subplot(2,len(lrs),cpt)
			semilogy(all_train[i][0],'b',alpha=0.5)
                        semilogy(all_train[i][7],'k',alpha=0.5)
			xlabel('Batch',fontsize=19)
                        if(lr==lrs[0]):
                                ylabel(r'$\log (\mathcal{L}_{CE})$',fontsize=21)
			title('Learning Rate:'+lr,fontsize=20)
                        subplot(2,len(lrs),len(lrs)+cpt)
                        plot(all_test[i][0],color='b',alpha=0.5)
                        plot(all_test[i][7],color='k',alpha=0.5)
			if(lr==lrs[0]):
                                ylabel('Test Accuracy',fontsize=21)
                        xlabel('Epoch',fontsize=19)
			cpt+=1
                suptitle(DATASET+' '+model,fontsize=18)
#                savefig(DATASET+'_'+model+'_loss.png')
#		close()
	show()	




