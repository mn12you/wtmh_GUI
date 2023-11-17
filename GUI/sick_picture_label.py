from CNN import CNN
from segment import segment
import matplotlib.pyplot as plt

path="6108.txt"
predict= CNN(path)
error_beats=segment(path)
sickpeak_list=[]
sickbeat_list=[]
for i in range(len(predict)):
    if predict[i] == "N":
        pass
    else: 
        plt.ylim(-1000,750)
        plt.text(88,850,str(predict[i+1]))
        plt.plot(error_beats[i+1])
        plt.show()
