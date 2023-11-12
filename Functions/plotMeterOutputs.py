import matplotlib.pyplot as plt
import numpy as np
import itertools

def plotMeterOutputs(profile, names_meters, path):
    
    fig, meters = plt.subplots(1,1)

    marker = itertools.cycle((',', '+', '.', 'o', '*','d','x','|','1')) 

    for i in range(len(names_meters)):
        meters.plot(np.gradient(profile[:,i],axis=0), marker = next(marker))
    meters.grid()
    meters.title.set_text('Meter Data')
    meters.set_xlabel('t(h)')
    meters.set_ylabel('kW')

    #Reduce the size of the plot by 20% so the legend can be placed on the outside
    box = meters.get_position()
    meters.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #Create a legend so reader can discern which curve belong to which meters
    meters.legend(names_meters, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    fig.savefig(path + "MeterReadings.png")
