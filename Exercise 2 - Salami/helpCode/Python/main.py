dirIn = '/home/max/Desktop/s194119/Salami/Exercise 2 - Salami/Data/data/'
import helpFunctions as hf 
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from scipy import io
from scipy import stats
import matplotlib.image as mpimg
import os
import logging
import datetime

a = datetime.datetime.now()
a = str(a)
logging.basicConfig(filename=f"{a[0:16]} LOG.txt",format='- %(message)s', level=logging.INFO)

# plt.imshow(fatImg,alpha=0.5)
# plt.imshow(meatImg,alpha=0.5)
# plt.show()

def statistics(meatPix,fatPix,multiIm):
    meanArr = np.empty([multiIm.shape[2],2])
    devArr  = np.empty([multiIm.shape[2],2])
    meanArr[:,0] = np.mean(meatPix,0)
    meanArr[:,1] = np.mean(fatPix,0)
    devArr[:,0] = np.std(meatPix,0)
    devArr[:,1] = np.std(fatPix,0)
    return meanArr, devArr #format meat, fat 
   

def simpleIntersection(meanArr): ##find channelwise treshold vals assuming that there is equal variance in fat/meat measurements

    tresholds = (meanArr[:,1]+meanArr[:,0])/2
    return tresholds


    
#more complicated tresholding using normal distributions with variable variances between meat and fat 
def normDist(meanArr,devArr,x): #creates normal-distribution array from mean and standard deviation
    normArr = np.empty([2,meanArr.shape[0],len(x)])
    for i in range(meanArr.shape[0]):
        normArr[0,i,:] =  stats.norm.pdf(x,meanArr[i,0],devArr[i,0]) #meat entry   #whoops should devArr be square?
        normArr[1,i,:] =  stats.norm.pdf(x,meanArr[i,1],devArr[i,1]) #fat entry
    
    return normArr

def findIntersection(normArr,COI,x,plot): #find intersection between the normal-distributions
    frac = normArr[0,COI,:]>normArr[1,COI,:] #find at which intensity p(fat) is less than p(meat)
    intersections = np.where(frac == frac.max())
    
    multSects = [intersections[0][0],intersections[0][-1]]
    intersection = np.argmax(frac==True)
    # print("Intersections at pixel intensities " + str(x[multSects[0]]) + " and "+str(x[multSects[1]]))
    
    
    if(plot==1):
        plt.style.use('ggplot')
        plt.plot(x,normArr[0,COI,:],'r1',alpha=0.5)
        plt.plot(x,normArr[1,COI,:],'b1',alpha=0.5)
        plt.axvline(x[multSects[0]])
        plt.axvline(x[multSects[1]])
        plt.show()
    else: 
        pass
    
    return intersection, x[multSects]

def IntersectionSaver(normArr,x): #save intersections in a numpy array
    tresholds = np.empty([normArr.shape[1]])
    for i in range(len(tresholds)):
        minIntersect, pixelVals = findIntersection(normArr,i,x,0)
        tresholds[i] = pixelVals[1]
    return tresholds

def inference(image,tresholds):
    predictions = np.empty([514,514,2,image.shape[2]])
    for j in range(len(tresholds)): #
        predictions[:,:,0,j] = (image[:,:,j]<tresholds[j])                        
        predictions[:,:,1,j] = (image[:,:,j]>tresholds[j])  
    
    return predictions

def inferencePlot(predictions,image):
    fmap = matplotlib.colors.ListedColormap(['green','blue'])    
    plt.figure()
    fig, axs = plt.subplots(4,5,figsize=(21,17))
    for i in range(4):
        for j in range(5):
            axs[i,j].imshow(image[:,:,i+j],cmap="gray")
            axs[i,j].imshow(predictions[:,:,0,i+j],cmap = fmap, alpha=0.5)
            axs[i,j].axis('off')
            axs[i,j].set_title("Ch"+str(5*i+j),fontsize=18,pad=-2)
    
    plt.savefig(f"{day}_simpleplot.png")
            
            


def calcCov(images):
    X = np.zeros([19,514*514])
    for i in range(images.shape[2]):
        X[i] = np.reshape(images[:,:,i],[1,-1])
    return X, np.cov(X,bias=True)


def calcDiscriminant(cov,X,mu,annotationIm,multiIm):
    S = np.zeros([514,514,2])
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if(annotationIm[i,j,0] or annotationIm[i,j,1] or annotationIm[i,j,2]):
                
                x = multiIm[i,j,:].T #all 19 channel vals
                #print(x.shape)
                for k in range(2):
                    firstTerm = np.dot(x.T,np.dot(np.linalg.inv(cov),mu[:,k]))#
                    secondTerm = np.dot((1/2)*mu[:,k].T,np.dot(np.linalg.inv(cov),mu[:,k]))
                    S[i,j,k] = firstTerm-secondTerm
                    
            else: 
                S[i,j,:] = np.nan
               
    return S


def multivariateInference(S):
    inferenceArr = np.zeros([S.shape[0],S.shape[1]]) #channel 0 = meat, channel 1 = fat
    totalPixels=0
    fatPixels=0
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j,0]>=S[i,j,1]:
                inferenceArr[i,j] = 255 # kød
                totalPixels+=1
                #inferenceArr[i,j,1] = np.nan
                
            elif S[i,j,0]<S[i,j,1]:
                inferenceArr[i,j] = -255 #fedt
                totalPixels+=1
                fatPixels+=1
                #inferenceArr[i,j,0] = np.nan
            #else: 
              #  inferenceArr[i,j,:] = np.nan
    fatPercentage = (fatPixels/totalPixels)*100
    logging.info(f'Total fatpercentage found was {fatPercentage:.2f}%! \n')
    return inferenceArr                

def needyBoi(preds,simpleTresholds, meatR,meatC,fatR,fatC, S=False):
    meatDict = {}
    fatDict = {}
    
    for k in range(0,len(simpleTresholds)):
        allOverAllTot = 0
        counterTot=0
        correctCounter=0
        wrongCounter=0
        allWrongsTot = 0
        for mR in meatR:
            for mC in meatC:
                counterTot+=1
                allOverAllTot+=1
                if S:
                    if preds[mR,mC]>0:
                        correctCounter+=1
                    elif preds[mR,mC]<0:
                        wrongCounter+=1
                        allWrongsTot+=1
                else:
                    if preds[mR,mC,0,k]:
                        correctCounter+=1
                    elif preds[mR,mC,1,k]:
                        wrongCounter+=1
                        allWrongsTot+=1
       
        correctnessPercent = (correctCounter/counterTot)*100
        if not S:
            # print(f'Channel {k+1} was {correctnessPercent:.2f}% right i finding meat!')
            # print(f'Channel {k+1} was {(wrongCounter/counterTot)*100:.2f}% wrong in finding meat!')
            meatDict[k]=correctnessPercent
        else:

            logging.info(f'Multivariate was {correctnessPercent:.2f}% right i finding meat!')
            logging.info(f'Multivariate was {(wrongCounter/counterTot)*100:.2f}% wrong in finding meat!'+'\n')
        counterTot=0
        correctCounter=0
        wrongCounter=0
        for fR in fatR:
            for fC in fatC:
                counterTot+=1
                allOverAllTot+=1
                if S:
                    if preds[fR,fC]>0:
                        correctCounter+=1
                    elif preds[fR,fC]<0:
                        wrongCounter+=1
                        allWrongsTot+=1
                else:
                    if preds[fR,fC,1,k]:
                        correctCounter+=1
                    elif preds[fR,fC,0,k]:
                        wrongCounter+=1
                        allWrongsTot+=1
        correctnessPercent = (correctCounter/counterTot)*100
        if not S:
            # print(f'Channel {k+1} was {correctnessPercent:.2f}% right i finding fat!')
            # print(f'Channel {k+1} was {(wrongCounter/counterTot)*100:.2f}% wrong in finding fat!')
            # print('\n')
            fatDict[k]=correctnessPercent
            # print(f'Error rate for channel {k+1} is {(allWrongsTot/allOverAllTot)*100:.2f} % !')
        else:
            logging.info(f'Multivariate was {correctnessPercent:.2f}% right i finding fat!')
            logging.info(f'Multivariate was {(wrongCounter/counterTot)*100:.2f}% wrong in finding fat!')
            logging.info(f'Error rate for multivariate is {(allWrongsTot/allOverAllTot)*100:.2f} % !'+'\n')
            break
    
    if not S:
        meat_max = max(meatDict, key=meatDict.get)
        fat_max = max(fatDict, key=fatDict.get)

        maxAvg = [0,0]
        for key in meatDict.keys():
            tempAvg = (meatDict[key]+fatDict[key])/2
            if tempAvg>maxAvg[0]:
                maxAvg[0] = tempAvg
                maxAvg[1] = key

    if not S:

        logging.info(f'Best channel for meat is channel {meat_max+1} with correctness of {meatDict[meat_max]:.2f}%!')
        logging.info(f'Best channel for fat is channel {fat_max+1} with correctness of {fatDict[fat_max]:.2f}%!')
        logging.info(f'Best average channel is channel {maxAvg[1]+1} with an average of {maxAvg[0]:.2f}%!' + '\n')
        # print('\n')
    # else:
    #     print(f'')



def runAll():
    dayList = ['01','06','13','20','28']

    for day in dayList:
        logging.info('-'*20+f' NOW RUNNING ON DATA FROM DAY {day} '+'-'*20+'\n')
        print(f'Now running analysis for day {day}...')
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{day}.mat' , f'annotation_day{day}.png', dirIn)

        multiIm.shape
        [fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
        [meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);


        fatImg = np.zeros([514,514])
        fatImg[fatR,fatC] = True
        meatImg = np.zeros([514,514])
        meatImg[meatR,meatC] = True

        meanArr, devArr = statistics(meatPix,fatPix,multiIm) 
        simpleTresholds = simpleIntersection(meanArr)

        x = np.linspace(0,120,10000) #define samplerange and rate
        normArr = normDist(meanArr,devArr,x) 
        minIntersect, pixelVals = findIntersection(normArr,1,x,0)
        complicatedTresholds = IntersectionSaver(normArr,x)
        preds = inference(multiIm,simpleTresholds)    
        # inferencePlot(preds,multiIm)
        X, cov = calcCov(multiIm)
        S = calcDiscriminant(cov, X, meanArr,annotationIm,multiIm)
        outputImages = multivariateInference(S)

        fig = plt.figure()

        fmap = matplotlib.colors.ListedColormap(['green','#FFFFFF00','blue'])
        im1 = plt.imshow(outputImages[:,:],cmap=fmap,interpolation="None",alpha=1)
        plt.title(day) #something like this (we need enumeration)
        plt.show()

        needyBoi(preds,simpleTresholds,meatR,meatC,fatR,fatC)
        needyBoi(outputImages,simpleTresholds,meatR,meatC,fatR,fatC,S=True)
    print(f'JOB IS DONE! Please look in " {a[0:16]} LOG.txt " for results.')

#runAll()

def train(day):
    #dayList = ['01','06','13','20','28']
    dayList = [day]

    for day in dayList:
        logging.info('-'*20+f' NOW TRAINING ON DATA FROM DAY {day} '+'-'*20+'\n')
        print(f'Now training on day {day}...')
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{day}.mat' , f'annotation_day{day}.png', dirIn)

        multiIm.shape
        [fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
        [meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);


        fatImg = np.zeros([514,514])
        fatImg[fatR,fatC] = True
        meatImg = np.zeros([514,514])
        meatImg[meatR,meatC] = True

        meanArr, devArr = statistics(meatPix,fatPix,multiIm) 
        simpleTresholds = simpleIntersection(meanArr)
        
    return meanArr,simpleTresholds



def modelEval(dayModel,meanArr,simpleTresholds):
    dayList = ['01','06','13','20','28']
    for day in dayList:
        logging.info('-'*20+f' NOW EVALUATING ON MODEL TRAINED FROM DAY {dayModel} on {day}'+'-'*20+'\n')
        print(f'Now running evaluation of model trained on {dayModel} on day {day}...')
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{day}.mat' , f'annotation_day{day}.png', dirIn)

        multiIm.shape
        [fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
        [meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);
        
        
        preds = inference(multiIm,simpleTresholds)    
        inferencePlot(preds,multiIm)
        
        X, cov = calcCov(multiIm)
        S = calcDiscriminant(cov, X, meanArr,annotationIm,multiIm)
        
        outputImages = multivariateInference(S)
    
        fig = plt.figure()
    
        fmap = matplotlib.colors.ListedColormap(['green','#FFFFFF00','blue'])
        im1 = plt.imshow(outputImages[:,:],cmap=fmap,interpolation="None",alpha=1)
        plt.title(day) #something like this (we need enumeration)
        #plt.show()
        plt.savefig(f'{dayModel}_on_{day}.png')
        needyBoi(preds,simpleTresholds,meatR,meatC,fatR,fatC)
        needyBoi(outputImages,simpleTresholds,meatR,meatC,fatR,fatC,S=True)
    print(f'JOB IS DONE! Please look in " {a[0:16]} LOG.txt " for results.')


dayList = ['01','06','13','20','28']
for day in dayList: 
    muT,tauT = train(day)
    modelEval(day,muT,tauT)

#modelEval('01',muT,tauT)
          
    
