import time
import sc
import importlib
import subprocess
import random
import matplotlib.pylab as plt
import numpy as np

def sigmoid(x):
        return 1/(1+np.exp(-x))

def backerror(E,W):
        E1=E*W
        return E1

def neu():
        global wa, wb, wa1, wb1, wc1, wa2, wb2, wc2, wconst1, wconst2, wconst3, w13, w23      
        wa=random.random()
        wb=random.random()
        wa1=random.random()
        wb1=random.random()
        wc1=random.random()
        wa2=random.random()
        wb2=random.random()
        wc2=random.random()
        w13=random.random()
        w23=random.random()
        wconst1=random.random()
        wconst2=random.random()
        wconst3=random.random()
        dwa1=0
        dwb1=0
        dwc1=0
        dwa2=0
        dwb2=0
        dwc2=0
        dw13=0
        dw23=0
        dwconst1=0
        dwconst2=0
        dwconst3=0

#-----------------------------------------------
        
        def neut1(a, b, c, dwa1, dwb1, dwc1, dwconst1):
                global wa1, wb1, wc1, wconst1
                wa1 = wa1 + dwa1
                wb1 = wb1 + dwb1
                wc1 = wc1 + dwc1
                wconst1 = wconst1 + dwconst1
                S = a*wa1 + b*wb1 + c*wc1 + 1*wconst1
                F = sigmoid(S)
                print(F)
                return F
        
        def neut2(a, b, c, dwa2, dwb2, dwc2, dwconst2):
                global wa2, wb2, wc2, wconst2
                wa2 = wa2 + dwa2
                wb2 = wb2 + dwb2
                wc2 = wc2 + dwc2
                wconst2 = wconst2 + dwconst2
                S = a*wa2 + b*wb2 + c*wc2 + 1*wconst2
                F = sigmoid(S)
                print(F)
                return F
          
        def neut3(f1,f2,dw13,dw23,dwconst3,X):
                global w13, w23, wconst3
                w13 = w13 + dw13
                w23 = w23 + dw23
                wconst3 = wconst3 + dwconst3
                S = f1*w13+f2*w23+1*wconst3
                F=sigmoid(S)
                E=X-F
                print(F)
                return F, w13, w23, E
            
#---------------------------------------------------
        
        def neu1(a,b,c):
                global wa1, wb1,wc1,wconst1
                F = sigmoid(a*wa1 + b*wb1 + c*wc1 + wconst1)
                return F
        
        def neu2(a,b,c):
                global wa2, wb2,wc2, wconst2
                F = sigmoid(a*wa2 + b*wb2+ c*wc2 + wconst2)
                return F
        
        def neu3(a,b):
                global w13,w23,wconst3
                f3=sigmoid(a*w13+b*w23+wconst3)
                return f3
        
        print("Training area initialized. Print what u want to do:")
        print("e=exit;l=learn")
        s=input()
        s="l"
        if s=="e":
                print("See you later!")
        if s=="l":
                print("Lets train something")
                print("Enter values")
                i=0
                q=0
                j=0
                x1=[1,-1]
                y1=[0,0]
                x2=[0,0]
                y2=[1,-1]
                plt.plot(x1,y1)
                plt.plot(x2,y2)
                plt.grid()
                plt.ion()
                while i<7000:
                        #-Тренировочные_данные-\/
                        a=[1,1,1,1,0,0,0,0]
                        b=[1,1,0,0,1,1,0,0]
                        c=[1,0,1,0,1,0,1,0]
                        x=[1,1,1,1,0,0,1,0]
                        #-Тренировочные_данные-/\
                        ai = a[q]
                        bi = b[q]
                        ci = c[q]
                        xi = x[q]
                        f1 = neut1(ai,bi,ci,dwa1,dwb1,dwc1,dwconst1)
                        f2 = neut2(ai,bi,ci,dwa2,dwb2,dwc2,dwconst2)
                        f3, w13, w23, E = neut3(f1,f2,dw13,dw23,dwconst3,xi)
                        E1 = backerror(E,w13)
                        E2 = backerror(E,w23)
                        dwa1 = E1*(f1*(1-f1))*ai
                        dwb1 = E1*(f1*(1-f1))*bi
                        dwc1 = E1*(f1*(1-f1))*ci
                        dwconst1 = E1*(f1*(1-f1))*1
                        dwa2 = E2*(f2*(1-f2))*ai
                        dwb2 = E2*(f2*(1-f2))*bi
                        dwc2 = E2*(f2*(1-f2))*ci
                        dwconst2 = E2*(f2*(1-f1))*1
                        dw13 = E*(f3*(1-f3))*f1
                        dw23 = E*(f3*(1-f3))*f2
                        dwconst3 = E*(f3*(1-f3))*1
                        print("------------------")
                        if q==6:
                                q=0
                        else:
                                q=q+1
                        i=i+1
                        try:
                                Ap.remove()
                        except:
                                print("Старт")
                        Ap=plt.scatter(E, E)
                        plt.draw()
                        plt.pause(0.0001)
                plt.ioff()
                plt.show()
                print("Lets train my skills")
                while i<5:
                    i=i+1
                    a=int(input())
                    b=int(input())
                    c=int(input())
                    f1 = neu1(a,b,c)
                    f2 = neu2(a,b,c)
                    f3 = neu3(f1,f2)
                    print(f3)
neu()












