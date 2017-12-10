import sys
import cPickle
import gzip
import os
import time
import io
#import scipy.io

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from scipy.io import arff
#from cStringIO import StringIO

class dA(object):
  #def __init__(self, numpy_rng, theano_rng=None, n_visible=784, n_hidden=1000,W=None, bhid=None, bvis=None):
  def __init__(self, numpy_rng, theano_rng=None, input=None,n_visible=784, n_hidden=1000,W=None, bhid=None, bvis=None):
    self.n_visible = n_visible
    self.n_hidden = n_hidden
    # create a Theano random generator that gives symbolic random values
    if not theano_rng:
      theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
      # note : W' was written as `W_prime` and b' as `b_prime`
    if not W:
      initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
      W = theano.shared(value=initial_W, name='W', borrow=True)
    
    if not bvis:
      bvis = theano.shared(value=numpy.zeros(n_visible,dtype=theano.config.floatX),borrow=True)
    
    if not bhid:
      bhid = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='b',borrow=True)
    
    self.W = W
    self.b = bhid
    self.b_prime = bvis
    self.W_prime = self.W.T
    self.theano_rng = theano_rng
    
    if input == None:
      self.x = T.dmatrix(name='input')
    else:
      self.x = input
    
    self.params = [self.W, self.b, self.b_prime]
  
  def set_x(self, input):
    self.x = input
  
  def get_corrupted_input(self, input, noisemodel, noiserate):
    if noisemodel == 'dropout':
      return  self.theano_rng.binomial(size=input.shape, n=1, p=1-noiserate, dtype=theano.config.floatX) * input
    else:
      return self.theano_rng.normal(size=input.shape, avg=0.0, std=noiserate, dtype=theano.config.floatX) + input
  
  def get_hidden_values(self, input):
    return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
  
  def get_reconstructed_input(self, hidden):
    return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
  
  def get_cost_updates(self, noisemodel, noiserate, learning_rate):
    tilde_x = self.get_corrupted_input(self.x, noisemodel, noiserate)
    y = self.get_hidden_values(tilde_x)
    z = self.get_reconstructed_input(y)
    L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
    cost = T.mean(L)
    gparams = T.grad(cost, self.params)
    updates = []
    for param, gparam in zip(self.params, gparams):
      updates.append((param, param - learning_rate * gparam))
    
    return (cost, updates)

def mytest_dA(da,train_set_x, learning_rate, noisemodel, noiserange, training_epochs=300, batch_size=20):
  tunning_epochs = range(1,20,2)+range(20, 301, 20);
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  #print train_set_x.get_value(borrow=True).shape[0],n_train_batches
  index = T.lscalar()
  #print index
  #x = T.matrix('x')
  
  for rate in noiserange:
    #rng = numpy.random.RandomState(123)
    #theano_rng = RandomStreams(rng.randint(2 ** 30))
    #if not isupdate:
    #  da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=train_set_x.get_value(borrow=True).shape[1],n_hidden=train_set_x.get_value(borrow=True).shape[1]/3)# n_hidden=100)
    #print "before",da.W.get_value(borrow=True)[0]
    cost, updates = da.get_cost_updates( noisemodel=noisemodel, noiserate=rate,learning_rate=learning_rate)
    train_da = theano.function([index], cost, updates=updates, givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})#, on_unused_input='ignore')
    start_time = time.clock()
    ############
    # TRAINING #
    ############
    # go through training epochs
    for epoch in range(1, training_epochs+1):
      # go through trainng set
      c = []
      for batch_index in xrange(n_train_batches):
        l = train_da(batch_index)
        c.append(l)
      
      #print 'Training epoch %d, cost %f' % (epoch, numpy.mean(c))
      if epoch in tunning_epochs:
        end_time = time.clock()
        training_time = (end_time - start_time)
      #print 'running time = %.2fm' % ((training_time) / 60.)
  #print ' with noise level %.2f' % (rate) + 'learning rate %.6f' %(learning_rate)  + ' finished in %.2fm' % ((training_time) / 60.)
  return {'W': da.W.get_value(borrow=True), 'b':da.b.get_value(borrow=True), 'b_prime':da.b_prime.get_value(borrow=True), 'cost':c}




if __name__ == '__main__':
  #if len(sys.argv) < 4:
  #  print 'Usage: python pret'
  mm = sys.argv[1]  
  lr = float(sys.argv[2])
  nlayer = int(sys.argv[3])
  #print mm,dd,lr,nlayer
  
  if mm == 'dropout':
    noiserange = [0.85]#[0.1, 0.25, 0.40, 0.55, 0.7, 0.85]
  else:
    noiserange = [1.1]
  
  numpy.set_printoptions(threshold=numpy.nan)
  #content = sys.stdin.read().strip()
  content=""
  rawcontent=sys.stdin.readline().strip()
  while rawcontent not in ['break', 'quit']:
    if rawcontent!="":
      content=content+rawcontent      
      #print len(rawcontent),rawcontent[-2:]
      if ";" in rawcontent:
        data_x=[]
        classlabel=[]
        #print content[-10:]
        conlist=content[2:].split("|")
        #print conlist[-1]
        for idx,con in enumerate(conlist):
          conlist=con.replace("[","").replace("]","").replace("\n","").replace(" ","").replace("|","").replace(";","").split(",")
          classlabel.append(conlist[-1])
          data_x.append(conlist[:-1])
          #print idx,len(conlist[:-1])
        
        #print len(data_x)
        data_x=numpy.array(data_x)
        data_x=data_x.astype(numpy.float)
        numpy.clip(data_x,0,1,data_x)
        
        data_y=numpy.array(classlabel)
        data_y=data_y.astype(numpy.float)
        
        if "@0" in content[0:2]: #New model
          xtrain = data_x
          #shared_x = theano.shared(numpy.asarray(xtrain, dtype=theano.config.floatX), borrow=True)
          #x = T.matrix('x')
          rng = numpy.random.RandomState(123)
          theano_rng = RandomStreams(rng.randint(2 ** 30))
          dAElist = [dict() for x in range(nlayer)]
          for lidx in range(nlayer):
            #shared_x.set_value(numpy.asarray(xtrain, dtype=theano.config.floatX))
            dAElist[lidx]['shared_x']=theano.shared(numpy.asarray(xtrain, dtype=theano.config.floatX), borrow=True)
            x = T.matrix('x')
            dAElist[lidx]['da'] = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=xtrain.shape[1], n_hidden=xtrain.shape[1]/3)
            dAElist[lidx]['Weights'] = mytest_dA(dAElist[lidx]['da'], train_set_x=dAElist[lidx]['shared_x'], learning_rate=lr, noisemodel=mm, noiserange = noiserange, training_epochs=30, batch_size=20)
            W=dAElist[lidx]['Weights']['W']
            b=dAElist[lidx]['Weights']['b']
            b_prime=dAElist[lidx]['Weights']['b_prime']
            hx = numpy.dot(xtrain, W) +numpy.tile(b,(xtrain.shape[0],1))
            xr = (1./(1+numpy.exp(-hx)))
            xtrain=xr
            #print "after",W[0]
        elif "@1" in content[0:2]: #Update Model
          xtrain = data_x
          for lidx in range(nlayer):
            #shared_x = theano.shared(numpy.asarray(xtrain, dtype=theano.config.floatX), borrow=True)
            dAElist[lidx]['shared_x'].set_value(numpy.asarray(xtrain, dtype=theano.config.floatX))
            x = T.matrix('x')
            dAElist[lidx]['da'].set_x(x)
            #print dAElist[lidx]['da'].x 
            dAElist[lidx]['Weights'] = mytest_dA(dAElist[lidx]['da'], train_set_x=dAElist[lidx]['shared_x'], learning_rate=lr, noisemodel=mm, noiserange = noiserange, training_epochs=10, batch_size=20)
            W=dAElist[lidx]['Weights']['W']
            b=dAElist[lidx]['Weights']['b']
            b_prime=dAElist[lidx]['Weights']['b_prime']
            hx = numpy.dot(xtrain, W) +numpy.tile(b,(xtrain.shape[0],1))
            xr = (1./(1+numpy.exp(-hx)))
            xtrain=xr
            #print "after",W[0]
        elif "@2" in content[0:2]: #Transform features
          xtrain = data_x
          for lidx in range(nlayer):
            W=dAElist[lidx]['Weights']['W']
            b=dAElist[lidx]['Weights']['b']
            b_prime=dAElist[lidx]['Weights']['b_prime']
            hx = numpy.dot(xtrain, W) +numpy.tile(b,(xtrain.shape[0],1))
            xr = (1./(1+numpy.exp(-hx)))
            xtrain=xr
        
        transdata=numpy.hstack((xr,data_y.reshape(data_y.shape[0],1)))
        transdata=transdata.astype(numpy.float)
        
        #datastring=str(data_x)
        #batchsize=1000
        #for i in range(len(datastring)/batchsize):
        strtxt = io.BytesIO()#StringIO.StringIO()
        numpy.savetxt(strtxt, transdata, fmt='%f,'*(transdata.shape[1]-1)+'%i', delimiter=',', header='', comments='')
        sys.stdout.write(strtxt.getvalue().replace("\n","|")+"\n")
        #print strtxt.getvalue().replace("\n","|")+"\n"
        #sys.stdout.write((numpy.array_repr(transdata)[8:-3]).replace("\n","").replace(" ","")+"\n")
        #print (str(transdata.shape)).replace("\n","").replace(" ","")+"\n"
        #sys.stdout.write((str(transdata.shape)).replace("\n","").replace(" ","")+"\n")
        sys.stdout.flush()
        content=""
      else:
        #print "rec\n"
        sys.stdout.write("rec\n")#data_x.shape)
        sys.stdout.flush()
    rawcontent = sys.stdin.readline().strip()
  '''
  s = sys.stdin.readline().strip()
  while s not in ['break', 'quit']:
    sys.stdout.write(s.upper() + '\n')
    sys.stdout.flush()
    s = sys.stdin.readline().strip()
  '''