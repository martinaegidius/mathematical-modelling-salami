
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0B2436","#C9A3A4","#D1B899","#BBBBBE","#AEB5AC","#858686","#ADCDEC","#B6BDCA"])


mu = [175.5,162.9]
sigma = (6.7)**2

x = np.linspace(100,200,1000)



PDFbois = (1/(6.7*np.sqrt(2*np.pi))*np.exp((-1/2)*(1/sigma)*((x-mu[0])**2)))
PDFgals = (1/(6.7*np.sqrt(2*np.pi))*np.exp((-1/2)*(1/sigma)*((x-mu[1])**2)))

fig, axs = plt.subplots(2)
fig.suptitle("Probablity density functions for bois and gals")
axs[0].plot(x,PDFbois)
axs[1].plot(x,PDFgals)
axs[1].set_xlabel("height in cm")
plt.show()


x = np.linspace(140,200,1000000)
a = stats.norm.pdf(x,mu[0],sigma**0.5)
b = stats.norm.pdf(x,mu[1],sigma**0.5)


def findIntersection(a,b,x):
    frac = a>b
    intersection = np.argmax(frac==True)

    return x[intersection]

intersect = findIntersection(a,b,x)    


plt.style.use('ggplot')
plt.plot(x,a,'r1',alpha=0.5)
plt.plot(x,b,'b1',alpha=0.5)
plt.axvline(intersect)

print("intersection is +" + str(intersect) + "cm")




   
              
    