# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:53:46 2017

@author: student
"""
import time

class Stopwatch:
    ctime=0;
    name = ''
    def __init__(self,name=''):
        self.name = name
        self.ctime = time.time()
        
    def reset(self,name=''):   
        self.ctime = time.time()        
        if name!='':
            self.name = name
    def clock(self):
        newtime = time.time()
        if self.name=='':
            print('Operation took %f ms'%((newtime-self.ctime)*1000))
        else:        
            print('%s took %f ms' %(self.name,(newtime-self.ctime)*1000))
            
    def get_time(self):
        return time.time()-self.ctime