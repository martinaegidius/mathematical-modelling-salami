
dirIn = '/home/max/Desktop/s194119/Salami/Exercise 2 - Salami/Data/data/'
import helpFunctions as hf 
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from scipy import stats

multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn)

# multiIm is a multi spectral image - the dimensions can be seen by
multiIm.shape
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);

def statistics(meatPix,fatPix,multiIm):
    meanArr = np.empty([multiIm.shape[2],2])
    devArr  = np.empty([multiIm.shape[2],2])
    meanArr[:,0] = np.mean(meatPix,0)
    meanArr[:,1] = np.mean(fatPix,0)
    devArr[:,0] = np.std(meatPix,0)
    devArr[:,1] = np.std(fatPix,0)
    return meanArr, devArr #format meat, fat 
   
meanArr, devArr = statistics(meatPix,fatPix,multiIm) 

def simpleIntersection(meanArr): ##find channelwise treshold vals assuming that there is equal variance in fat/meat measurements
    #tresholds = np.empty(meanArr.shape[0])
    tresholds = (meanArr[:,1]+meanArr[:,0])/2
    return tresholds

simpleTresholds = simpleIntersection(meanArr)
    

#more complicated tresholding using normal distributions with variable variances between meat and fat 
def normDist(meanArr,devArr,x): #creates normal-distribution array from mean and standard deviation
    normArr = np.empty([2,meanArr.shape[0],len(x)])
    for i in range(meanArr.shape[0]):
        normArr[0,i,:] =  stats.norm.pdf(x,meanArr[i,0],devArr[i,0]) #meat entry   #whoops should devArr be square?
        normArr[1,i,:] =  stats.norm.pdf(x,meanArr[i,1],devArr[i,1]) #fat entry
    
    return normArr


x = np.linspace(0,120,10000) #define samplerange and rate
normArr = normDist(meanArr,devArr,x) 

def findIntersection(normArr,COI,x,plot): #find intersection between the normal-distributions
    frac = normArr[0,COI,:]>normArr[1,COI,:] #find at which intensity p(meat) is less than p(fat)
    intersections = np.where(frac == frac.max())
    
    multSects = [intersections[0][0],intersections[0][-1]]
    intersection = np.argmax(frac==True)
    print("Intersections at pixel intensities " + str(x[multSects[0]]) + " and "+str(x[multSects[1]]))
    
    if(plot==1):
        plt.style.use('ggplot')
        plt.plot(x,normArr[0,COI,:],'r1',alpha=0.5)
        plt.plot(x,normArr[1,COI,:],'b1',alpha=0.5)
        plt.axvline(x[multSects[0]])
        plt.axvline(x[multSects[1]])
    else: 
        pass
    
    return intersection, x[multSects]

minIntersect, pixelVals = findIntersection(normArr,18,x,1)

def IntersectionSaver(normArr,x): #save intersections in a numpy array
    tresholds = np.empty([normArr.shape[1]])
    for i in range(len(tresholds)):
        minIntersect, pixelVals = findIntersection(normArr,i,x,1)
        tresholds[i] = pixelVals[1]
    return tresholds


complicatedTresholds = IntersectionSaver(normArr,x)

