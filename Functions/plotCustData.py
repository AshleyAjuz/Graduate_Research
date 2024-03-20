import matplotlib.pyplot as plt
import numpy as np
import itertools

def plotCustData(randCust,cust_data, attack_data, stpIdx, path):
    
    fig, (reg, attack) = plt.subplots(1,2)

    marker = itertools.cycle((',', '+', '.', 'o', '*','d','x','|','1')) 
    attack_data_labels = ['f1','f2','f3','f4','f5']
    x_axis = [i for i in range(stpIdx)]

    reg.plot(x_axis, cust_data[0:stpIdx])
    reg.title.set_text(f'Honest Customer Readings: Customer {randCust}')
    reg.set_xlabel('t(h)')
    reg.set_ylabel('kW')

    attackIdx = randCust
    for i in range(1,6):
        attack.plot(x_axis, attack_data[attackIdx][0:stpIdx], marker = next(marker))
        attackIdx += 761 # Need to increment by the total number of customers in order to get to the next attack function

    attack.title.set_text('Attack Data Readings')
    attack.set_xlabel('t(h)')
    attack.set_ylabel('kW')

    #Reduce the size of the plot by 20% so the legend can be placed on the outside
    box = attack.get_position()
    attack.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #Create a legend so reader can discern which curve belong to which meters
    attack.legend(attack_data_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.show()

    fig.savefig(path + "HonestvsAttack.png")
