
import matplotlib.pyplot as plt
import os 

path = os.path.dirname(os.path.abspath(__file__))
imdir = os.path.join(path,'Data/data')
image = os.path.join(imdir,'color_day01.png')

image = plt.imread(image)
plt.imshow(image)