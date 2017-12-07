# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:24:32 2017

@author: student
"""
import pickle
import numpy as np
from copy import deepcopy

# activation function:
sigmoid = lambda x: 1/(1+np.exp(-x))
dsigmoid=lambda x: np.exp(x)/( (np.exp(x) + 1)**2)



def load_MLP_from_file(filename):
    f = open(filename,"rb")
    nnmpl = pickle.load(f)       
    
    f.close()
    return nnmpl
class MLP(object):    
    """Neural Network with 1 hidden layer,
    Trained with backpropagation"""
    def __init__(self, nx, nh, ny=1, Wmin=-15, Wmax=15, noinit = False, epsilon = .3):
        super(MLP, self).__init__()
        self.nx = nx
        self.nh = nh
        self.ny = ny
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.epsilon = epsilon
        
        # do random weight initialization
        if not noinit:
            self.init_weights()

        return
    
    def save_to_file(self, filename):
        g = open(filename,"wb")
        pickle.dump(self,g)
        
        g.close
        return
    
    def copy(self,weightless = True):
        '''If weightless==True, the Weigth matrices Whx And Wyh are not assigned
            They can be initialized to random by calling MLP.init_weights()
            if weightless==False, these are copied as well
        '''
        if weightless:
            return MLP(self.nx,self.nh,self.ny,self.Wmin,self.Wmax,noinit=True)
        else:
            mlp = MLP(self.nx,self.nh,self.ny,self.Wmin,self.Wmax,noinit=True)
            #mlp.Whx = self.Whx[:]
            #mlp.Wyh = self.Wyh[:]
            mlp.Whx = deepcopy(self.Whx)
            mlp.Wyh = deepcopy(self.Wyh)
            return mlp
    
    def init_weights(self):
        """
        Init weight matrices
        """
        nx, ny, nh = self.nx, self.ny, self.nh
        #### weight matrices; unirand init::
        Whx = np.random.uniform(self.Wmin,self.Wmax, (nh, nx+1) ) # +1 for the bias input
        Wyh = np.random.uniform(self.Wmin,self.Wmax, (ny, nh+1) ) 
        
        self.Whx = Whx
        self.Wyh = Wyh 
        return


    def ffwd(self, x):
        """Sweep through the network of sigmoidal units; 
        store all activations and outputs and return y (output)
        """

        Whx, Wyh = self.Whx, self.Wyh

        # process x+bias input -> h
        x = np.r_[x, 1]
        ha = Whx.dot(x)
        h = sigmoid(ha)

        # process h+bias input -> y
        h = np.r_[h, 1]
        #ya = np.dot(Wyh, h)
        ya = Wyh.dot(h)
        y = sigmoid(ya)

        # this next line is short for: self.x=x; self.h=h, ....
        #self.__dict__.update( dict(x=x, ha=ha,h=h,ya=ya,y=y ) )
        # return output:
        return y

    def backpropagation(self, t):
        '''print self.Wyh
        print self.Wyh.T
        print self.Whx
        print self.y
        print self.x
        input('waiting')'''
        # store the feedback weights:
        Why = self.Wyh.T[:-1, :] # ignoring the weight attached to the bias..

        # compute error 'vector':
        e = self.y - t
        # the weight update for the output weights is as normal:
        # inp * error * dsigmoid * learning rate
        # here, I'm using some tricks to skip a few of these steps:
        h = self.h[:, np.newaxis]
        e = e[np.newaxis, :]
        '''print'h=',h
        print'e=',e
        print 'sigmeps', dsigmoid(self.ya) * epsilon
        print 'edot',e.dot(h.T)
        time.sleep(5)
        '''
        # the actual weight update
        delta_y = (dsigmoid(self.ya) * e).T
        self.Wyh -= self.epsilon * delta_y * h.T

        # now, propagate the error back, using Why:
        ha = self.ha[:,np.newaxis]
        x =   self.x[:, np.newaxis]
        delta_h = dsigmoid(ha) * Why.dot(delta_y) 
        self.Whx  -= self.epsilon * delta_h * x.T
        return
    '''
    original
        # the actual weight update
        self.Wyh -= e.dot(h.T) * dsigmoid(self.ya) * epsilon

        # now, propagate the error back, using Why:
        ha = self.ha[:, np.newaxis]
        delta_bp = Why.dot(e) * dsigmoid(ha)

        # use this new delta to update hidden weights:

        x = self.x[:, np.newaxis]
        self.Whx -= ( delta_bp.dot(x.T) ) * epsilon
        return'''
    
    '''
    from demo
         
        # the actual weight update
        delta_y = (dsigmoid(self.ya) * e).T
        self.Wyh -= epsilon * delta_y * h.T

        # now, propagate the error back, using Why:
        ha = self.ha[:,np.newaxis]
        x =   self.x[:, np.newaxis]
        delta_h = dsigmoid(ha) * Why.dot(delta_y) 
        self.Whx  -= epsilon * delta_h * x.T
        return
'''