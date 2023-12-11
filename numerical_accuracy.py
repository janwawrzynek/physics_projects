# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:54:51 2022

@author: Jan Wawrzynek 20335060
"""
#Jan Wawrzynek 20335060
#Computer Simulation Numerical Methods Assignment 1 

import numpy as np
under = 1.0
over = 1.0
N =1080 
factor = 2
for x in range(N):
    
    under = under/2.
    over = over*2.
    print('under:','\t\t', under,'\t\t',  'over:', over)
    #under flow = 5e-324 
    #overflow = 8.98846567431158e+307

    M = 60
eps = 1.0
factor = 2
for x in range(M):
    eps = eps/2
    one = 1.0 + eps
    
    print(x, eps, one)

    
#Machine Precision 2.220446049250313e-16 1.0000000000000002
#1.4 Q1a
import matplotlib.pyplot as plt

def differentiation2(t,h ):   
    # True differentiation 
    
    d_true = np.exp(t)
    dt_F = (np.exp(t + h)- np.exp(t))/h
    dt_C = (np.exp(t + h/2) -  np.exp(t- h/2))/h
    
    error_forward = abs((d_true - dt_F)/d_true)
    error_central =  abs((d_true -  dt_C)/ d_true)
    print(dt_F, "Forward")
    print( error_forward, "forward error")
    print( dt_C, "Central")
    print(error_central, "central error")
          
    return error_forward
differentiation2(0.1, 2.22e-16)
error_forward_e = differentiation2(100.,np.logspace(-1, -20, 500) )
t =100.



def differentiation_e_central(t,h ):   
    # True differentiation 
    
    d_true = np.exp(t)
    dt_F = (np.exp(t + h)- np.exp(t))/h
    dt_C = (np.exp(t + h/2) -  np.exp(t- h/2))/h
    
    error_forward = abs((d_true - dt_F)/d_true)
    error_central =  abs((d_true -  dt_C)/ d_true)
    return error_central

error_central_e = differentiation_e_central(100.,np.logspace(-1, -20, 500) )
#Step size h
h = np.logspace(-1, -20, 500)
plt.loglog(h,error_forward_e, label = "forward error")
plt.loglog(h, error_central_e, label = "central error")

#plt.legend('forward error',  "central error")
plt.legend(loc = 0)


plt.title('step size vs Error magnitude exp(x) t = 100.0')
#plt.xlabel("step size")
#plt.ylabel("error size")
plt.show()


#1.4 Q1b
# Cosine Functions. That printed out the forward and central errors 
def differentiation_cosf(t,h ):   
    # True differentiation 
    d_true = -np.sin(t)
    print(d_true, 'd_true')
    dt_F = (np.cos(t + h)- np.cos(t))/h
    print(dt_F, 'dt_F')
    dt_C = (np.cos(t + h/2) -  np.cos(t- h/2))/h
    
    #return dt_F
    #return dt_C
    
    error_forward = abs((d_true - dt_F)/d_true)
    return error_forward

def differentiation_cosc(t,h ):   
    # True differentiation 
    d_true = -np.sin(t)
    print(d_true, 'd_true')
    dt_F = (np.cos(t + h)- np.cos(t))/h
    print(dt_F, 'dt_F')
    dt_C = (np.cos(t + h/2) -  np.cos(t- h/2))/h
    #return dt_F
    #return dt_C
    error_central = abs((d_true - dt_C)/dt_C)
    error_forward = abs((d_true - dt_F)/d_true)
    print(dt_F, "Forward")
    print( error_forward, "forward error")
    print( dt_C, "Central")
    print(error_central, "central error")
    return error_central
#1.4 Q1a 
# This was used to print out the  values of forward, central and errors
differentiation_cosc(0.1, 2.22e-16)


error_forward_e = differentiation_cosf(100.0,np.logspace(-1, -20, 500) )
error_central_e = differentiation_cosc(100.0,np.logspace(-1, -20, 500) )
h = np.logspace(-1, -20, 500)

plt.loglog(h,error_forward_e, label = "forward error")
plt.loglog(h, error_central_e, label = "central error")

#plt.legend('forward error',  "central error")
plt.legend(loc = 0)


plt.title('step size vs Error magnitude cos(t)  t = 100.0')
#plt.xlabel("step size")
#plt.ylabel("error size")
plt.show()


#1.4 Q1b
#Here I worked out the second derivative of cosine using the central theorem.
def double_der(t, h):
    d2t = (np.cos( t + h) + np.cos(t-h) - 2*np.cos(t))/h**2
    d2t_true = -np.cos(t)
    error = abs((d2t_true - d2t)/d2t_true)
    return d2t
    
    #print(h, 'h', d2t_true, 'd2t_true', d2t ,'d2t', error, 'error')

x_values = np.linspace(-2*np.pi,2*np.pi)
secderivative = double_der(x_values, 2.220446049250313e-16)
#print(secderivative)
plt.plot(x_values, secderivative)

#plt.legend('forward error',  "central error")
plt.legend(loc = 0)


plt.title('d2t of cos(t) h = 2.220446049250313e-16')
#plt.xlabel("step size")
#plt.ylabel("error size")
plt.show()
#As you can see I was having issues getting my plots to label the axes. 
#I do not understand what is wrong as I have done this many times before.
#Yet for some reason they do not work.
#I included the code that should plot them.

