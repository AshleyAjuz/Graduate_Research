import py_dss_interface
import pathlib

#Function will setup dss interface and install energy meters are user specified locations
def placeMeters(path, Lines):
    
    dss = py_dss_interface.DSSDLL()
    dss_file = pathlib.Path(path).joinpath("ieee37.dss")
    dss.text(f"compile [{dss_file}]")

    #Get the number of meters
    num_meters = len(Lines)

    #Create the meters
    for i in range(num_meters):
        dss.text(f'New EnergyMeter.meter{i}')
        dss.text(f"edit Energymeter.meter{i} element=Line.{Lines[i]} enabled=True")

    dss.solution_write_mode(1)
    dss.solution_write_number(1)

    #return the dss interface
    return dss

