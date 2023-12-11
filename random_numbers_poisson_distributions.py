# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:30:17 2022

@author: wawrz
"""

import matplotlib.pyplot as plt
import numpy as np
import math # used to define the factorial
import statistics as st  #this is used to find the standard deviation



#here I defined empty arrays that will store the values of the Poisson Distributions
Poisson1 = []
Poisson2 = []
Poisson3 = []
nPoisson1 = []
N = np.int64(50)
#here I build the Poisson distribution
for n in range(N) :
   #First I define the factorial which will be an imput for the next stage.
    def factorial(n):
        factorial = math.factorial(n)
        return factorial
    factorial(n)
    #here taking the factorial in , defined the poisson
    def poisson(n, mean,factorial):
    
          
        poisson = (((mean)**n)/(factorial))*(np.exp(-1*mean))
        return poisson
    
    #sum calculated the mean 
    def npoisson(n, mean,factorial):
    
          
        poisson = n*(((mean)**n)/(factorial))*(np.exp(-1*mean))
        return npoisson
    #called the function for the parameters to generate the poisson distributions 
    Poisson = poisson(n,1,factorial(n))
    Poisson1.append(Poisson)
    Pois5 = poisson(n,5,factorial(n))
    Poisson2.append(Pois5)
    Pois10 = poisson(n,10,factorial(n))
    Poisson3.append(Pois10)
    #np(n) appending 
    nP1 = n*poisson(n,1,factorial(n))
    nPoisson1.append(nP1)
n = np.arange(0,N,1)    
plt.plot(n,Poisson1)
plt.plot(n,Poisson2)
plt.plot(n,Poisson3)
plt.title('Q1 Poissons <n> = 1,5,10')
plt.ylabel('frequency')
plt.xlabel('segment number')
plt.legend(['<n> = 1',' <n> = 5','<n> = 10'])

#here is the start of part 2 
#checking taht the poissons are normalised and finding the means 
print('beginning of answers q2')
def sum_poisson1():
    sum_poisson1 = sum(Poisson1)
    return sum_poisson1
sum_poisson1 =  sum_poisson1()
print('p(n)mean = 1',sum_poisson1 )
def sum_poisson2():
    sum_poisson2 = sum(Poisson2)
    return sum_poisson2
sum_poisson2 =  sum_poisson2()
print('p(n)mean = 5',sum_poisson2 )
def sum_poisson3():
    sum_poisson3 = sum(Poisson3)
    return sum_poisson3
sum_poisson3 =  sum_poisson3()
print('p(n) mean = 10',sum_poisson3 )
 #next part find n*p(n)
def sum_npoisson1():
    sum_npoisson1 = sum( nPoisson1)
    return sum_npoisson1
sum_npoisson1 =  sum_npoisson1()



#q2 
#here printing the mean values   
n = np.arange(0,N,1)
maybe = n*Poisson1
print(sum(maybe),'np(n) <n> =1')
maybe2 = n*Poisson2
npn2 = sum(maybe2)
print(npn2,'np(n) <n> =5')
maybe3 = n*Poisson3
print(sum(maybe3),'np(n) <n> =10')
nsquared = []
for i in n:
    nsquares = i**2
    nsquared.append(nsquares)
#finding the mean squared values 
nsquaredp1 = []

nsquaredp1 = (n**2)*Poisson1
print(sum(nsquaredp1), 'n2p(n) <n> = 1')

nsquaredp2 = (n**2)*Poisson2
print(sum(nsquaredp2), 'n2p(n) <n> = 5')

nsquaredp3 = (n**2)*Poisson3
print(sum(nsquaredp3), 'n2p(n) <n> = 10')
    
# finding the standard deviation
stdevp1 = st.stdev(Poisson1)
print(stdevp1, 'standard deviation of Poisson 1 <n> = 1')
stdevp2 = st.stdev(Poisson2)
print(stdevp2, 'standard deviation of Poisson 2 <n> = 5')
stdevp3 = st.stdev(Poisson1)
print(stdevp3, 'standard deviation of Poisson 3 <n> = 10')
#part 3 
#randomly place darts in regions
L = 100 #number of regions 
N = 50 #number of darts thrown 
ar = np.arange(1,L+1,1)




L = 100
N = 50
T = 10


numb_darts = np.arange(N)
############################################
#part 3 
#define an array of zeros that will hold how many times each section was hit by a dart 
H_n = np.zeros(N)
for i in range(T): #T is the number of trials eqch of which will be added together
    dartboard = np.zeros(L) #this array will register the segments that get hit by darts  
    for j in range(N): #50 hits per trial 
        dartboard[np.random.randint(L)] += 1 #throwing the darts and appending them back into the segments they hit 
        
    Hindv = []
    for i in numb_darts:
        Hindv.append(np.count_nonzero(dartboard == i))   
#this nonzero function goes through the terms of H and counts    
#how many sections have however many darts in them 
#https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html    
    H_n += Hindv #the individual trial outcomes are appended into the big combined one 
print(Hindv)
print(sum(Hindv)) #check if the values are appended as expected. 

 






print('H-N', H_n)
print(sum(H_n))
H_n = np.array(H_n)
# the p sim method for finding the normalized H(n) is incorrect as the H(n) will counts the number of 
#sections with each amount of darts in them so they will add up to 100 
#dividing by L*T = 1000 this would result in 100/1000 which is not normalised, therefore used the method
#below 
#norm = H_n /(L*T)
# normalise the distribution, want sum to add up to 1
#source https://stackoverflow.com/questions/26785354/normalizing-a-list-of-numbers-in-python 
norm = [float(i)/sum(H_n) for i in H_n]
sumb = sum(norm)
print(sumb)

#finding the mean value of the simulation
mean_sim_array = n*norm
mean_sim = sum(mean_sim_array)
print(mean_sim, 'mean of the simulated dart problem')
#calculated that the mean is 0.5



#plotting against the poisson with the mean of the simulation redefining the poisson here 
Poisson_simb = []
for n in range(N) :
    #ns = np.arange(1,21,1)
    #n = ns[n]
    #mean = 1
    def factorial(n):
        factorial = math.factorial(n)
        return factorial
    factorial(n)
    def poisson(n, mean,factorial):
    
        #mean = [1,5,10]
        #for i in mean:  
        poisson = (((mean)**n)/(factorial))*(np.exp(-1*mean))
        return poisson
    
    #sum np*(n)
    def npoisson(n, mean,factorial):
    
        #mean = [1,5,10]
        #for i in mean:  
        poisson = n*(((mean)**n)/(factorial))*(np.exp(-1*mean))
        return npoisson
    
    Poisson_sim = poisson(n,0.5,factorial(n))
    Poisson_simb.append(Poisson_sim)






n = np.arange(N)
#L_sections = np.arange(L)
plt.show()
#q3 plot 
plt.plot(numb_darts, norm)
plt.plot(n,Poisson_simb)
plt.title('part 3 Plot of simulation and Poisson mean = 0.5')
plt.legend(['simulation', 'Poisson'])
plt.ylabel('frequency')
plt.xlabel('segment')
plt.show()


#sim_mean = 0.5

#part 4 

norm = np.array(norm)
#here found the minimum value of the simulation that probes the data. This cannot be = 0 so exclude that  
probe = min(norm[(norm>0)])
print(probe, 'smallest value of which the simulation probes the Poisson')
plt.scatter(numb_darts, norm)
plt.plot(n,Poisson_simb)
plt.plot(n, [probe for i in n])
plt.title('part 4 Log plot for which the numerical simulation probes the poisson ')
plt.yscale('log')
plt.legend(['poisson distribution', 'min value that probes the poisson', 'numerical data'])
plt.ylabel('Probability')
plt.xlabel('segments n')
plt.show()
#q5
# for T = 1000, 10000 we can see that the numerical data perfectly matches the poisson distribution 
#q6

print('Jan Wawrzynek 2035060')