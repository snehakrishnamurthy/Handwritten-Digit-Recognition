import cPickle
import gzip
import time
import numpy as np
from PIL import Image
import numpy as np
import os

count=0
image=np.empty((0,784))
target=[]
image=[]
Tup=np.zeros((19999,10))
for i in range(0 ,10):
    path1 = "USPSdata/Numerals/"+str(i)+"/"
    listing = os.listdir(path1)
    for j in listing:
        if(j!="Thumbs.db"):
            if(j!="2.list"):
                target.append(i)
                count+=1
                im = Image.open(path1 + j)
                im = im.getdata()
                im = im.resize((28,28))
                im = np.array(list(im))
                image.append(im.reshape((1,784)))
image=np.array(image)

for i in range(0,19999):
    Tup[i,target[i]]=1

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
x_train=training_data[0]
t_train=training_data[1]
x_val=validation_data[0]
t_val=validation_data[1]
x_test=test_data[0]
t_test=test_data[1]
minm=float("inf")

n=50000
n1=10000
k=10

w=np.random.rand(10,784)*0
for eta in np.arange(0.0068,.0069):
    aj=0
    b=np.ones((10,1))
    yk=np.zeros((10,1))
    ykval=np.zeros((10,1))
    T=np.zeros((n,10))
    Tval=np.zeros((n1,10))
    Ttest=np.zeros((n1,10))

    for i in range(0,n):
        T[i,t_train[i]]=1
    for i in range(0,n1):
        Tval[i,t_val[i]]=1
        Ttest[i,t_test[i]]=1

    for i in range(0,n):
        a=np.dot(w,x_train.T[:,i])
        a_new=a.reshape(10,1)
        a=a_new+b
        aj=np.sum(np.exp(a))
        for m in range(0,k):
            yk[m]=np.exp(a[m])/aj
        T_new=T[i].reshape(10,1)
        temp=yk-T_new
        x_tr=x_train[i,:].reshape(1,784)
        gradientE = np.dot(temp,x_tr)
        w=w-eta*gradientE

    aj1=0
    count =0
    for i in range(0,n1):
        a1=np.dot(w,x_val[i])
        a_new=a1.reshape(10,1)
        a1=a_new+b
        aj1=np.sum(np.exp(a1))
        for m in range(0,k):
            ykval[m]=np.exp(a1[m])/aj1
        T_new=Tval[i].reshape(10,1)
        sa=np.argmax(ykval)
        if(T_new[sa]!=1):
            count=count+1
    error=count/10000.0
    if error < minm :
        count_s=count
        etas=eta
        minm=error
        w_save=w
    acc=((10000-count)/10000.0)*100
    print "Logistic Regression (MNIST data-set)"
    print "Validation data-set Accuracy : ", acc,"%"
    print "------------------------------------------\n"

aj2=0
tcount=0
yktest=np.zeros((10,1))
for i in range(0,n1):
    a2=np.dot(w_save,x_test.T[:,i])
    a_new=a2.reshape(10,1)
    a2=a_new+b
    aj2=np.sum(np.exp(a2))
    for m in range(0,k):
        yktest[m]=np.exp(a2[m])/aj2
    T_new=Ttest[i].reshape(10,1)
    sas=np.argmax(yktest)
    if(T_new[sas]!=1):
        tcount=tcount+1
acc=((10000-tcount)/10000.0)*100
print "Logistic Regression (MNIST data-set)"
print "Test data-set Accuracy : ", acc,"%"
print "------------------------------------------\n"


aj21=0
tcount1=0
yktest1=np.zeros((10,1))
for i in range(0,19999):
    im=image[i]
    a21=np.dot(w_save,im.T)
    a_new1=a21.reshape(10,1)
    a21=a_new1+b
    aj21=np.sum(np.exp(a21-np.max(a21)))
    for m1 in range(0,k):
        yktest[m1]=np.exp(a21[m1]-np.max(a21)-1000)/aj21
    T_new=Tup[i].reshape(10,1)
    sau1=np.argmax(yktest1)
    if(T_new[sau1]!=1):
        tcount1=tcount1+1
acc1=((19999-tcount1)/19999.0)*100



#Neural Network

import cPickle
import gzip
import time
import numpy as np

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
x_train=training_data[0]
t_train=training_data[1]
x_val=validation_data[0]
t_val=validation_data[1]
x_test=test_data[0]
t_test=test_data[1]
minm=float("inf")

n1=10000
K=10
M=1000
eta=0.004

b1=np.zeros((500,M))
b2=np.zeros((10,500))
T=np.zeros((50000,10))
Tval=np.zeros((n1,10))
Ttest=np.zeros((n1,10))
w1=.001*np.random.randn(784,M)
w2=.001*np.random.randn(M,10)


for i in range(0,50000):
    T[i,t_train[i]]=1
for i in range(0,n1):
    Tval[i,t_val[i]]=1
    Ttest[i,t_test[i]]=1


#q=50000
#r=q
N=50000
eta=0.002
print eta
for epoc in range(0,40):
    for ba in range(0,50000/500):
        x=x_train[ba*(500):(ba+1)*(500)]
        t=T[ba*(500):(ba+1)*(500)]
        h1=np.dot(x,w1)+b1
        z=1.0/(1.0+np.exp(-h1))
        a=np.dot(z,w2)+b2.T
        c_max = np.max(a,axis = 1).reshape(500,1)
        a_exp = np.exp(a-c_max)
        total_a_exp = np.sum(a_exp,axis = 1)
        for val in range(0,500):
        	a_exp[val] = a_exp[val]/total_a_exp[val]

        y = a_exp
        delk=y-t
        hdash=z*(1.0-z)
        grad2=np.dot(delk.T,z).T
        delj=np.dot(w2,delk.T)
        delj=hdash.T*delj
        w2-=np.multiply(eta,grad2)
        grad1=delj.dot(x).T
        w1-=np.multiply(eta,grad1)
        b1 -= eta*delj.T
        b2 -= eta*delk.T
        wm = np.max(w1,axis = 1).reshape(784,1)
        w1 = w1/wm
#print n
count1=0

for k in range(0,10000/500):

    x = x_val[k*500:(k+1)*500]
    tt = Tval[k*500:(k+1)*500]
    z = np.dot(x,w1)
    z = z + b1
    z = 1.0/(1+np.exp(-z))
    a = np.dot(z,w2)
    a = a + b2.T
    c_max = np.max(a,axis = 1).reshape(500,1)
    a_exp = np.exp(a-c_max)
    total_a_exp = np.sum(a_exp,axis = 1)
    for val in range(0,500):
        a_exp[val] = a_exp[val]/total_a_exp[val]
    y = a_exp
    for i in range(0,500):
   	    if np.argmax(y[i]) != np.argmax(tt[i]):
   			count1 += 1
acc=((10000-count1)/10000.0)*100
print "Neural Network (MNIST data-set)"
print "Validation data-set Accuracy : ", acc,"%"
print "------------------------------------------\n"

count1=0


for k in range(0,10000/500):

    x = x_test[k*500:(k+1)*500]
    tt = Ttest[k*500:(k+1)*500]
    z = np.dot(x,w1)
    z = z + b1
    z = 1.0/(1+np.exp(-z))
    a = np.dot(z,w2)
    a = a + b2.T
    c_max = np.max(a,axis = 1).reshape(500,1)
    a_exp = np.exp(a-c_max)
    total_a_exp = np.sum(a_exp,axis = 1)
    for val in range(0,500):
        a_exp[val] = a_exp[val]/total_a_exp[val]
    y = a_exp
    for i in range(0,500):
   	    if np.argmax(y[i]) != np.argmax(tt[i]):
   			count1 += 1
acc=((10000-count1)/10000.0)*100
print "Neural Network (MNIST data-set)"
print "Test data-set Accuracy : ", acc,"%"
print "------------------------------------------\n"

count1=0
