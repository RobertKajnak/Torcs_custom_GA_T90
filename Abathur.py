import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from copy import deepcopy
import subprocess
import csv
from stopwatch import Stopwatch
from mlp_evo import MLP
import mlp_evo as mlp
import datetime
import os
# learning rate epsilon set globally:

ICHECK=False

class Abathur:
    def __init__(self,nx,nh,ny,dataset=None,Wmin=-1, Wmax=1, poolsize=100, survivors=80, \
                 leftover_parents=5,individual_mutation_chance=.2, \
                 gene_mutation_chance=.1, mutation_swing = 1, \
                 crossover_chance = .5, discrete=False , cleanSlate=True):
        #the pool should always be kept in an ordered state
        
        self.nx = nx
        self.nh = nh 
        self.ny = ny
        #self.dataset = dataset
        self.generation = 0
        self.poolsize = poolsize
        self.survivors = survivors
        self.leftover_parents = leftover_parents
        self.indmut = individual_mutation_chance
        self.genmut = gene_mutation_chance
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.Wdiff = Wmax-Wmin
        self.mutswing = mutation_swing
        self.discrete = discrete
        self.crosschance = crossover_chance
        
        self.is_log_started = False
        
        self.pool = []    
        if cleanSlate:    
            for i in range(0,poolsize):
                self.pool.append((-1,MLP(nx,nh,ny,Wmin=self.Wmin,Wmax=self.Wmax)))       

            #self.insort(self.pool,MLP(nx,nh,ny,Wmin=self.Wmin,Wmax=self.Wmax))
        else:
            for i in range(0,10):
                net = mlp.load_MLP_from_file(os.path.abspath(os.curdir)+"/genomes/genome" + str(i) + "/thenet.net")
                self.pool.append((-1,net))
                   
        self.get_fitness_and_sort()
        self.display_best_specimens(self.poolsize)
            
        return

    
    def insort(self,array, specimen, fitness = None):
        i = 0;
        if fitness==None:
            key = self.fitness(specimen,self.discrete)
        else:
            key = fitness
        
        value = specimen

        if len(array) == 0:
            array.insert(0,(key,value))
            return
    
        if key<=array[-1][0]:
            array.insert(len(array),(key,value))
            return
        
        for element in array:
            if key>element[0]:
                array.insert(i,(key,value))
                break
            i = i+1
                
        return 
    def run_and_get(self):
        write_genomes(self)    
        run_simulation()
        return get_results()
        
    def get_fitness_and_sort(self):
        res = self.run_and_get()
        newpool =[]
        for v in res:
            self.insort(newpool,self.pool[int(v[0])][1],float(v[1]))
        self.pool = newpool
    
    def fitness(self,specimen,discrete):
        E=0
        for x,t in self.dataset:
            y = specimen.ffwd(x)
            if discrete==1:
                E += np.abs(t-np.floor(y+.5))
            elif discrete==2:
                E += np.abs(t-np.floor(y+.5)) + (t-y) ** 2
            else:
                E += (t-y) ** 2
            
            #print t,y,np.abs(t-int(y+.5))
        #to make it 'the higher the better'
        E = np.sum(E)
        return -E#/len(self.dataset)
    
    '''def mate(self,parent1,parent2):
        child1 = MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax,noinit=True)
        child2 = MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax,noinit=True)
        avgWhx = (parent1.Whx + parent2.Whx)/2
        avgWyh = (parent1.Wyh + parent2.Wyh)/2
        deltaWhx = parent1.Whx - parent2.Whx
        deltaWyh = parent1.Wyh - parent2.Wyh
        
        child1.Whx = avgWhx - deltaWhx*.16
        child2.Whx = avgWhx + deltaWhx*.16
        
        child1.Wyh = avgWyh - deltaWyh*.16
        child2.Wyh = avgWyh + deltaWyh*.16
        
        return child1,child2'''
    
    def mate(self, *parents):
        '''
        Currently only works for 2 parents
        accepts any number of parents, but the basic structure of the first
        will be copied (nx, nh, etc.) and is expected to be identical to Abathur specs
        '''
        # goes through all the parends and does a 'rotation' with the genes

        #afterwards, it is assumed that there are at least 2 parents
        if len(parents) == 1:
            return parents[0].copy()
        
        parentcount = len(parents)
        
        avgWhx = np.zeros((self.nh, self.nx+1))
        avgWyh = np.zeros((self.ny, self.nh+1))
        for parent in parents:
            avgWhx += parent.Whx
            avgWyh += parent.Wyh
        avgWhx /= parentcount
        avgWyh /= parentcount
    
        
        children = [parents[0].copy(weightless=True)] * (parentcount+1)
       
        children[0].Whx = deepcopy(avgWhx)
        children[0].Wyh = deepcopy(avgWyh)
       
        #parent1 gene and parent 2 gene
        for i,(p1g,p2g) in enumerate(zip(parents[0].Whx.flat,parents[1].Whx.flat)):
            if np.random.random()<=self.crosschance:
                children[1].Whx.flat[i] = p1g
                children[2].Whx.flat[i] = p2g
            else:
                children[2].Whx.flat[i] = p1g
                children[1].Whx.flat[i] = p2g
        
        for i,(p1g,p2g) in enumerate(zip(parents[0].Wyh.flat,parents[1].Wyh.flat)):
            if np.random.random()<=self.crosschance:
                children[1].Wyh.flat[i] = p1g
                children[2].Wyh.flat[i] = p2g
            else:
                children[2].Wyh.flat[i] = p1g
                children[1].Wyh.flat[i] = p2g
                
        '''for i,pgs in enumerate([parent.Wyh.flat for parent in parents]):
            if np.random.random()<=self.crosschance:
                for j in range(1,parentcount+1):
                    children[j].Wyh.flat[i] = pgs[j] 
            else:
                children[2].Wyh.flat[i] = p1g
                children[1].Wyh.flat[i] = p2g
        '''
        return children 
        
    #TODO - verify if mutations are done correctly
    #TODO - increase efficiency based on mutation chance
    def mutate(self,pool,nosort=False):
        #i =0 
        
        
        if ICHECK: print ("Total Errors - Starting mutation:",self.check_integrity(pool) )
        newpool = []
        for indituple in pool:
            #TODO it is probably not necessary to use copy
            individual = indituple[1].copy(weightless = False)
            '''for i in range(0,len(individual.Whx)):
                    for j in range(0,len(individual.Whx[0])):
                            individual.Whx.flat[i] += 1'''
            if np.random.random()<=self.indmut:
                #need to modify the individual weights, which are floats,
                #therfore iteration over the space will not work    
                
                for i in range(0,len(individual.Whx.flat)):
                    if np.random.random()<=self.genmut:
                         individual.Whx.flat[i] += self.Wdiff * self.mutswing * \
                                                 (np.random.random() - .5)
                for i in range(0,len(individual.Wyh.flat)):
                    if np.random.random()<=self.genmut:
                         individual.Wyh.flat[i] += self.Wdiff * self.mutswing * \
                                                 (np.random.random() - .5)
                
            if nosort==False:    
                self.insort(newpool,individual)
            else:
                newpool.append((-1,individual))
            
        if ICHECK: print ("Total Errors - Returning:",self.check_integrity(newpool)) 
            #i+=1
        return newpool
    
    def evolve(self):
        if ICHECK: print ("Total Errors - evolve start:",self.check_integrity(self.pool) )
        self.generation += 1
        
        ##########################
        ##The custom version for group fitness
        ####
        
        #print('before')
        #self.display_best_specimens(self.poolsize,log=False)
        
        newpool = self.pool[:self.leftover_parents]
        
        cpm = 3
        nr_matings_req= 1.0*self.survivors/3.0/cpm
        repr_chance = .5
        
        #Select mating pairs
        for i in range (0,self.poolsize):
            nr_matings = 0
            for j in range (i,self.poolsize):
                if (np.random.random()<repr_chance):
                    nr_matings += 1
                    children = self.mate(self.pool[i][1], self.pool[j][1])
                    #print "mating ",i,"with",j
                    for child in children:
                        #self.insort(newpool,child)
                        newpool.append((-1,child))
                    print('nr_matings:%f, nr_matigns_req:%f, len(newpool):%f,self.survivors:%f'\
                            %(nr_matings,nr_matings_req,len(newpool),self.survivors))
                    if nr_matings>=nr_matings_req:
                        #print "children produced:",nr_matings * cpm
                        #print "too many children"
                        break;
                    if len(newpool)>=self.survivors:
                        #print "too many surivors"
                        break
            nr_matings_req = nr_matings_req/2 - 1 
            repr_chance /= 2
            
            if len(newpool)>=self.survivors:
                break
            
        #print('\n\nAfter mating')        
        #self.display_best_specimens(self.poolsize,log=False)
        
        if ICHECK: print ("Total Errors - mating halfway:",self.check_integrity(self.pool) )
        
        print("Newpool length == :" + str(len(newpool)))
        self.pool = newpool
        print("Pool length == :" + str(len(self.pool)))
        if len(self.pool) > self.poolsize:
            self.pool = self.pool[:self.poolsize]
            print('trimming excess parents')
        else:
            print('filling missing parent\'s place')    
            for i in range(len(self.pool),self.poolsize):
                self.pool.append((-1,MLP(self.nx,self.nh,self.ny,Wmin=self.Wmin,Wmax=self.Wmax)))
                
        
        #print('\n\nAfter trimming')        
        #self.display_best_specimens(self.poolsize,log=False)
        
        ###MUTATION        
        self.pool = [self.pool[0]]+self.mutate(self.pool[1:],nosort=True)
        #print('\n\nAfter mutating')        
        #self.display_best_specimens(self.poolsize,log=False)
        
        self.get_fitness_and_sort()
        print('\n\nCycle finished')
        self.display_best_specimens(self.poolsize)
        #self.display_best_specimens(self.poolsize)
    
        
        return
        
    #returns the best performing individual
    def prime_specimen(self):
        return self.pool[0][1]
        
    def get_all_specimens(self):
        return [specimen[1] for specimen in self.pool]
        
    
    def display_best_specimens(self,n=5, log=True):
        print ("Generation:",self.generation)
        print     ("Place -   Score")
        for i in range(0,min(n,len(self.pool))):
            if ICHECK and self.fitness(self.pool[i][1],self.discrete)!=self.pool[i][0]:
                print ("--------\nCORRUPTION DETECTED\n--------")
            print ("%d.   -   %f"%(i,   self.pool[i][0]))
            
        if ICHECK :print ("Total Errors - printing prime specimens:",self.check_integrity(self.pool) )
        
        if log==True:               
            with open("AbathursDiary.txt", "a") as logfile:
                if not self.is_log_started:
                    self.is_log_started = True
                    st = str(datetime.datetime.now()).split('.')[0]
                    logfile.write(st + ' Format: best worst total\n')
                avg=0
                for spec in self.pool:
                    avg+=spec[0]
                logfile.write('%f %f %f\n'%(self.pool[0][0],self.pool[-1][0],avg))



    def check_integrity(self,pool):
        s = 0
        for specimen in pool:
            if self.fitness(specimen[1],self.discrete) != specimen[0]:
                s+=1
        return s
        
        
    '''def __str__(self):
        s =    "Rank    --   Score  \n"
        for i,specimen in enumerate(self.pool):
            s+="%d.     --   %f     \n"%(i,specimen[0])
        return s'''
    
def get_results(default = False):
    if default:
        with open("ratings.csv") as ratingfile:
            reader = csv.reader(ratingfile, delimiter=',')
            ratings = list(reader)
            for rating in ratings:
                rating[0]=rating[0][7]
    else:
        ratings=[]
        for i in range(0,10):
            with open(os.path.abspath(os.curdir)+"/genomes/genome" + str(i) + \
                    "/fitness_parameter.csv") as ratingfile:
                reader = csv.reader(ratingfile, delimiter=',')
                scores = list(reader)
                #print(scores[0][0], scores[0][1], float(scores[0][0])/float(scores[0][1]))
                ratings.append((i,float(scores[0][0])))  
    #print(ratings)
    return ratings

def run_simulation():
    '''try:
        output = subprocess.check_output(["python3", "torcs_tournament.py", "quickrace.yml"]) 
    except subprocess.CalledProcessError as e:
        output = e.output
    print(output)
    '''
    subprocess.check_output(["python3", "../torcs_tournament.py", "quickrace.yml"]) 
    return
    
def write_genomes(abathur):
    abathur.pool[0][1].save_to_file(os.path.abspath(os.curdir)+"/genomes/" + str(abathur.generation) + "thenet.net")
    for i,specimen in enumerate(abathur.get_all_specimens()):
        specimen.save_to_file(os.path.abspath(os.curdir)+"/genomes/genome" + str(i) + "/thenet.net")
    return


    
def let_the_evolution_commence():
    aba = Abathur(8,40,3,poolsize=10,survivors=8, \
                 leftover_parents=2,individual_mutation_chance=.2, \
                 gene_mutation_chance=.3, mutation_swing = 1, \
                 crossover_chance = .5, discrete=False,cleanSlate=True )
    
    t = Stopwatch("Generation");    
    for i in range(0,5):
        
        print("Generation %d started\n---------------------\n\n"%(i))
        t.reset()
        aba.evolve()
        t.clock()
    write_genomes(aba)
    return

if __name__ == '__main__':
    #learn_xor()
    #learn_vowels()
    #learn_all()
    
    #TODO - there is a bug in the metric
    #bs = learn_vowels_by_evolution()
    #bs = learn_all_by_evolution()
    
    t = Stopwatch('Complete simulation')
    let_the_evolution_commence()
    t.clock()

    t = Stopwatch('Getting results')    
    #print(get_results())
    t.clock()
    

    
# pairing mating:
    # vowels: 100gen discrete: 1
    # vowels: 30gen continuous: 2
    # voiwels: 100gen cont: 1
    # voiwels: 100gen combi: 1
    # all: 100gen cont: 15
    # all: 100gen disc: 19
    # qll: 100gen combi: 19

# no mating
    # all: 100gen cont: 22
    # all: 100gen dosc: 22
    # vowels: 100 gen cont: 2
    
#advanced mating:
    #vowel 100gen cont: 0 (0.12913)
    #vowel 100gen combi: 0, 0.00000 (after about 90 gens)
    #all 100gen combi: 14 (27.4314)
    #all 100gen cont: 16 (14.64) 
    #all 100gen disc: 