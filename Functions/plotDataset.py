import matplotlib.pyplot as plt
import numpy as np
import random

def plotDataset(X, y, dataset, path, fileName):
     
     #Split up the instances of benign data vs malicious data
     benign_instances = np.where(np.array(y) == 0)[0]
     malicious_instances = np.where(np.array(y) == 1)[0]

     #Select a random instance to plot
     numB = np.random.randint(0, len(benign_instances),1)[0]
     numM = np.random.randint(0, len(malicious_instances),1)[0]

     idx_B = benign_instances[numB]
     idx_M = malicious_instances[numM]

     fig, (benign, malicious) = plt.subplots(2,1)

     benign.plot(X[idx_B][0:90])
     malicious.plot(X[idx_M][0:90])

     #Set titles
     benign.title.set_text(f"Benign Energy Consumption Data - {dataset}")
     malicious.title.set_text('Malicious Energy Consumption Data')

     #Set x axis
     malicious.set_xlabel('t(h)')
     
     #Set y axis
     benign.set_ylabel('kW')
     malicious.set_ylabel('kW')

     plt.show()

     fig.savefig(path + fileName)
