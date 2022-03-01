from scipy import stats 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0B2436","#C9A3A4","#D1B899","#BBBBBE","#AEB5AC","#858686","#ADCDEC","#B6BDCA"])

#joint distribution
x = np.linspace(-6,6,10000)
x1 = stats.norm.pdf(x,0,2)
x2 = stats.norm.pdf(x,1,3)

def jointFunc(x1,x2):
    return (1/2*np.pi)*(1/6)*np.exp((-1/2)*((x1**2)*1/4+(1/9)*(x2-1)**2))

jointFunc = jointFunc(x1,x2)
plt.plot(x,jointFunc)
plt.title("skrt")
plt.show()


#joint and correlated distribution 
def jointCorFunc(x1,x2,rho):
    return (1/2*np.pi)*(1/6)*1/(np.sqrt(1-rho**2))*np.exp((-1/2)*(1/(1-rho**2))*((x1**2)*1/4+(1/9)*(x2-1)**2)-2*rho*(1/6)*x1*(x2-1))

jointCorFunc = jointCorFunc(x1,x2,(2/3))
plt.plot(x,jointCorFunc)
plt.show()






