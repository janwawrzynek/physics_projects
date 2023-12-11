# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:11:31 2022

@author: wawrz
"""
print('Jan Wawrzynek 20335060')
import numpy as np
import matplotlib.pyplot as plt 

from importlib import reload
plt=reload(plt)
#here defined the space between dots that would result in 25x25 grid 

nt,nx = .2,.24
t = np.arange(0,5,nt) #here set the t to range from t to 5 as required 
x = np.arange(-3,3,nx) # here set x to range from -3 to 3 
T,X = np.meshgrid(t,x) #this funtion creates the 25x25 grid using the t,x imputs
#below defined the ordinary differential equation 
dx = ((1 + T)*X) + 1 - (3*T) + (T**2)
dt = np.ones(dx.shape)
x_zero = 0.0655
start = 0
#step  = 0.04
step = 0.02
#for part 3 update step to 0.02
end = 5
#defined the ODE to be evaluated using the 3 methods 
def f(t,x):
	f = ((1 +t)*x) + 1 - (3*t) + (t**2)
	return f
t = np.arange(start,5, step)

#def seuler(x,t,step):
#	x_new = x + step*f(t,x)
#	return x_new
#when used the simple euler formula provided in the course material
#got a strange plot that did not follow the direction field vectors.
#found altenate representation of the simple euler method at the source below
#https://codeguru.academy/?p=326


#   
def seuler(f,x_zero,t):  #here have the simple euler method   
    x_new = np.zeros(len(t)) #here defined an empty array x values
    x_new[0] = x_zero 
    for n in range(0,len(t)-1):
        x_new[n+1] = x_new[n] + f(x_new[n],t[n])*(t[n+1] - t[n])
        #appended the x values of ODE found using the seuler method to array 
    return x_new # recovered the array  of the evaluated ODE 
seul = seuler(f,x_zero,t) #calling the simple euker over range 
def ieuler(t,x,step): #here have the improved euler method from lecture notes 
	imp_euler = x + 0.5*step*( f(t,x) + f(t+step, x + step*f(t, x)) )
	return imp_euler

def rk(t,x,step): #here have the runga kutta method defined from the notes 
	k1 = f(t,x)
	k2 = f(t + 0.5*step, x + 0.5*step*k1)
	k3 = f(t + 0.5*step, x + 0.5*step*k2)
	k4 = f(t + step, x + step*k3)
	runga = x + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) 
	return runga #returned the runga kutta method 

n = int((end-start)/step)

# pre-allocate arrays
x = np.arange(start,end,step)
# set the arrays that will hold the improved euler and runga kutta approximations of ODE
ieul = np.zeros(n)
ruku = np.zeros(n)



ieul[0] = x_zero
ruku[0] = x_zero


# increment solutions
for i in range(1,n):
	#iterated over the ODE for the range, with n steps.
    #then appended into the earlier defined list 
	ieul[i] = ieuler(x[i-1], ieul[i-1], step)
	ruku[i] = rk(x[i-1], ruku[i-1], step)
plt.xlabel('t')
plt.ylabel('x')	
plt.plot(t, seul, label = 'simple euler')
plt.plot(x, ieul, color = 'black') #plotting the improved euler method 
plt.plot(x, ruku, color = 'green') #runga kutta plot


plt.quiver(T,X,dt,dx, color = 'red', headlength = '5')
plt.ylim(-3,3)



plt.xlim(0,5)

plt.quiver(T,X,dt,dx, color = 'red', headlength = '5')
#this function plt.quiver creates a direction field
plt.legend(['simple euler' , 'improved euler', 'Runga Kutta', 'direction field'] )
plt.title('ODE evaluated with different methods step = 0.02')

plt.show()