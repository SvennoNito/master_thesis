def dictToListParams(parameter):
    "dictToListParams{'ENa': 65.[mV], ...}"
    from collections import OrderedDict as OD

    # Set default parameters
    #default = {
    #    "VT": -63., "d": 96., "area": 28947.456, "Cm": 1,
#
    #    "ENa": 65., "EK": -90., "EL": -70., "ECa": 120.,
#
    #    "gNa": 50., "gK": 5., "gL": 0.1, "gCa": 0.01, "gM": 0.08,
#
    #    "M1": -0.32, "M2": -13., "M3": 4., "M4": 0.28, "M5": -40., "M6": 5.,
    #    "H1": 0.128, "H2": 17., "H3": 18., "H4": 4., "H5": 40., "H6": 5.,
    #    "N1": -0.032,"N2": 15., "N3": 5., "N4": .5, "N5": 10., "N6": 40.,
    #    "P1": 98., "P2": 10., "P3": 3.3, "P4": 20., "tauMax": 4000.
    #}
    #default = OD(sorted(default.items(), key=lambda t: t[0]))
    default = OD()

    default["VT"] = -63.
    default["d"] = 96.
    default["area"] = 3216.990#28947.456

    default["Cm"] = 1.

    default["ENa"] = 65.
    default["EK"] = -90.
    default["EL"] = -70.
    default["ECa"] = 120.

    default["gNa"] = 50.
    default["gK"] = 5.
    default["gL"] = 0.1
    default["gCa"] = 0.01
    default["gM"] = 0.08

    default["M1"] = -0.32
    default["M2"] = -13.
    default["M3"] = 4.
    default["M4"] = 0.28
    default["M5"] = -40.
    default["M6"] = 5.

    default["H1"] = 0.128
    default["H2"] = 17.
    default["H3"] = 18.
    default["H4"] = 4.
    default["H5"] = 40.
    default["H6"] = 5.

    default["N1"] = -0.032
    default["N2"] = 15.
    default["N3"] = 5.
    default["N4"] = .5
    default["N5"] = 10.
    default["N6"] = 40.

    default["P1"] = 98.
    default["P2"] = 10.
    default["P3"] = 3.3
    default["P4"] = 20.
    default["tauMax"] = 4000.

    # Replace given paramters
    for key, val in parameter.items():
        default[key] = val

    # Store in list
    out = list(default.values())
    return out

def buildStimVec(nA, stim_start, stim_duration, dt):
    "create_stimulus_vector(2.5[nA], 1000[ms], 500[ms], 0.01[step/ms])"
    import numpy as np
    start           = int(stim_start*1/dt)                       #  5.000
    length          = int(stim_duration*1/dt+2*start)            # 20.000
    stop            = length-start                               # 15.000
    stimulus_vector = np.zeros(length)                           # array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
    stimulus_vector[start:stop] = nA                             # array([ 0.,  0.,  1., ...,  1.,  0.,  0.])
    return stimulus_vector

#print(dictToListParams({"ENa": 5})) 
#print(create_stimulus_vector(2.5, 500, 500, 0.01))
