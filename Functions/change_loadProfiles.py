import py_dss_interface

#Given a dss circuit and data (a 2D matrix containing label names and values)
#will instantiate loadshapes and apply them to the loads specified by 
#loadNames.

def change_loadProfiles(dss, data, loadNames):
    
    #Variable for storing the number of Loadshapes
    num_loadShapes = len(data[0])

    #Variable for storing the number of Loads
    num_loads = len(loadNames)
    
    #Create the loadshapes
    for i in range(num_loadShapes):
        dss.text(f"New LoadShape.{data[0][i]} npts=96 interval=0.5 mult={data[1][i]}")

    #Assign the loadshapes to the user specified loads
    for i in range(num_loads):
        dss.loads_write_name(f"{loadNames[i]}")
        dss.loads_write_daily(f"{data[0][i]}")
    
    return(dss)
