# DEAP: https://github.com/DEAP/deap/tree/54b83e2cc7a73be7657cb64b01033a31584b891d
from HH_cython import hhModel
from HH_helper import *
from HH_track import *

# IMPORT
import random, numpy, os, efel, scipy, math, time, array, json, timeit, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from collections import OrderedDict as OD
from deap import algorithms, base, creator, tools, benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
#from plot_gen import *
from scoop import futures

# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--stim',                 type=int, nargs='+', default=[1])
parser.add_argument('-n', '--numParents',           type=int,            default=1000)
parser.add_argument('-g', '--numGens',              type=int,            default=100)
parser.add_argument('-t', '--toSave',               type=int, nargs='+', default=[0])
parser.add_argument('-c', '--channelsToInclude',    type=int, nargs='+', default=[1,1,1,1],  help='INa + IK + IL + IM')
parser.add_argument('-f', '--features',                       nargs='+', default=['voltage_base', 'peak_voltage', 'peak_time'])
parser.add_argument('-p', '--modelParams',                    nargs='+', default=['EL', 'gL', 'area', 'Cm', 'ENa', 'gNa', 'EK', 'gK', 'gM', 'VT', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'P1', 'P2', 'P3', 'P4', 'tauMax'])
args = parser.parse_args()





### SETTINGS
###############################################################################

#PATH
mainpath   = r''
pathout    = r'results'
pathout    = os.path.abspath(pathout)
f0         = str(len(args.channelsToInclude)) + 'ch_'
f1         = ''.join(str(x) for x in args.stim) + 'stim_'
f2         = str(len(args.modelParams)) + 'par_'
f3         = str(args.numGens) + 'gen_'
f4         = str(args.numParents) + 'n_'
f5         = str(len(args.features)) + 'feat'
foldername = f0+f1+f2+f3+f4+f5

# PARALLEL PROCESSING yes/no
parallel_processing = "yes"
plotflag            = "no"
saveflag            = "yes"
sortbykey           = lambda t: t[0]

# STARTING VALUES
channels = {"EL": -70, "gL": 0.1, "area": 3216.990, "Cm": 1,
            "ENa": 65., "gNa": 50,
            "EK": -90., "gK": 5.,
            "gM": 0.08,
            "VT": -63,
            "M1": -0.32, "M2": -13., "M3": 4., "M4": 0.28, "M5": -40., "M6": 5.,
            "H1": 0.128, "H2": 17., "H3": 18., "H4": 4., "H5": 40., "H6": 5.,
            "N1": -0.032, "N2": 15., "N3": 5., "N4": .5, "N5": 10., "N6": 40.,
            "P1": 98., "P2": 10., "P3": 3.3, "P4": 20., "tauMax": 4000.

            } 
bounds   = {"EL": [-300, -0.001], "gL": [0.001, 1], "area": [0.001, 36000], "Cm": [0.001, 1],
            "ENa": [-300, 300], "gNa": [-300, 300],
            "EK":  [-300, 300], "gK": [-300, 300],
            "gM":  [-300, 300],
            "VT":  [-300, 300],
            "M1": [-300, 300], "M2": [-300, 300], "M3": [-300, 300], "M4": [-300, 300], "M5": [-300, 300], "M6": [-300, 300],
            "H1": [-300, 300], "H2": [-300, 300], "H3": [-300, 300], "H4": [-300, 300], "H5": [-300, 300], "H6": [-300, 300],
            "N1": [-300, 300], "N2": [-300, 300], "N3": [-300, 300], "N4": [-300, 300], "N5": [-300, 300], "N6": [-300, 300],
            "P1": [-300, 300], "P2": [-300, 300], "P3": [-300, 300], "P4": [-300, 300], "tauMax": [-300, 300]
            }

# SET PARAMETERS
num_rep        = 1                                               # how often to repeat EA
sigma          = 1                                               # individual is taken from gaussian distribution with mean=x and std=abs(x)/sigma
prob_crossover = 0.6                                             # probability that crossover takes place
eta            = 25                                              # mutation power = 1/eta+1 (high eta -> offspring and parents are similar
bigerror       = 100.                                            # error value if channel value is out of bounds or brian model exploded
stim_start, stim_duration = 500, 500                             # ms
dt             = 0.01                                            # timestep

stim_currents  = args.stim                                       # external current stimulation [nA], must be in a list
num_parents    = args.numParents                                 # number of parents
num_gen        = args.numGens                                    # number of generations
model_params   = args.modelParams                                # model Parameters
fI             = np.array(args.channelsToInclude, dtype=np.int32)# assign a factor to INA, IK, IL and IM
gens_to_save   = list(set(args.toSave + [0] + [num_gen]))        # generations to save
num_params     = len(model_params)                               # number of parameters to optimize
prob_mutation  = 1/num_params                                    # probability that mutation takes place

channels       = OD((k, channels[k]) for k in model_params)                        # ordered dictionary
bounds         = OD((k, bounds[k]) for k in model_params)                          # ordered dictionary
channels       = OD(sorted(channels.items(), key=sortbykey))
bounds         = OD(sorted(bounds.items(), key=sortbykey))

low, up        = [x[0] for x in bounds.values()], [x[1] for x in bounds.values()]  # lower and upper bounds
channelnames   = list(channels.keys())                                             # channel names

# EFEL FEATURES
feature_list    = args.features  
num_features    = len(feature_list)
num_finalpop    = (2**(num_features-1))*10
#['Spikecount', 'peak_voltage', 'min_AHP_values', 'AP_begin_voltage', 'spike_half_width', 'voltage_base', 'steady_state_voltage_stimend', 'AP_begin_time', 'peak_time']
# min_AHP_values: Absolute voltage values at the first after-hyperpolarization

# Efel SETTINGS
efel.reset()                     # reset settings
efel.setThreshold(20)            # Def.: -20 --- The voltage has to cross the threshold, and go below the threshold again
efel.setDerivativeThreshold(4)   # Def.: +10 --- Used to detect the beginning of a spike (after the Threshold was used to detect the spike as a whole). Once the dV/dt crosses this threshold for 3 time point (in the interval starting from previous AHP and tip of spike), the beginning of the spike is set there.

# PLOT SETTINGS
#pylab.rcParams['figure.figsize']   = 13, 6.5
#sns.set_style()
#gens_to_plot = 2  # plot every xth generation

# LOAD FEATURES OF EXPERIMENTAL DATA
xl_mean = pd.read_json("median")
xl_var  = pd.read_json("distance")

# hold all important information in dictionary
global allinfo
allinfo = OD()
allinfo['INa+IK+IL+IM'] = fI
allinfo['stimulus_currents'] = stim_currents
allinfo['stimulus_start'] = stim_start
allinfo['stimulus_duration'] = stim_duration
allinfo['initial_jitter'] = sigma
allinfo['prob_crossover'] = prob_crossover
allinfo['prob_mutation'] = prob_mutation
allinfo['eta'] = eta
allinfo['biggest_error'] = bigerror
allinfo['number_parents'] = num_parents
allinfo['number_gens'] = num_gen
allinfo['number_parameter'] = num_params
allinfo['size_pareto'] = num_finalpop
allinfo['features'] = feature_list
allinfo['parameters'] = channelnames
allinfo['starting_values'] = channels
allinfo['constraints'] = bounds










### ERROR FUNCTION
###############################################################################
# spikeDistance = 1/T * (sum(abs(spikeTimesTrace(1:numSpikesToCompare) - (spikeTimesModel(1:numSpikesToCompare)))) +  abs(numel(spikeTimesTrace)- numel(spikeTimesModel));

def spikeDistance(model_spikeTime, stim_current, bigerror=bigerror, xl_mean=xl_mean, xl_var=xl_var):
    model_spikeTime    = model_spikeTime
    model_spikeCount   = len(model_spikeTime)
    bio_spikeTime      = numpy.array(xl_mean.loc[xl_mean['stimulus'] == stim_current, "peak_time"].values[0])
    bio_spikeCount     = numpy.array(xl_mean.loc[xl_mean['stimulus'] == stim_current, "Spikecount"].values[0])
    numSpikesToCompare = min(bio_spikeCount, model_spikeCount)
    spikeTimeDif       = abs(bio_spikeTime[:numSpikesToCompare] - model_spikeTime[:numSpikesToCompare])
    spikeCountDif      = abs(bio_spikeCount - model_spikeCount)
    
    # normalize model_spikeTime
    spikeTimes = []
    if not list(model_spikeTime):
        spikeTimes = bigerror
    for i in range(numSpikesToCompare):
        if model_spikeTime[i] == bio_spikeTime[i]:
            spikeTimes.append(0)
        elif model_spikeTime[i] < bio_spikeTime[i]:
            spikeTimes.append(spikeTimeDif[i] / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, "peak_time"].values[0][i][0])))
        elif model_spikeTime[i] > bio_spikeTime[i]:
            spikeTimes.append(spikeTimeDif[i] / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, "peak_time"].values[0][i][1])))
    
    # normalize model_spikeCount
    if model_spikeCount == bio_spikeCount:
        spikeCount = 0
    elif model_spikeCount < bio_spikeCount:
        spikeCount = spikeCountDif / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, "Spikecount"].values[0][0]))
    elif model_spikeCount > bio_spikeCount:
        spikeCount = spikeCountDif / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, "Spikecount"].values[0][1]))
    
    # error function
    error = numpy.mean(spikeTimes) + spikeCount
    return error
    
# EFEL: GET MEDIAN
def efel_median(features):
    for key, val in features.items():
        if key == "peak_time":
            pass
        elif val is None or not list(val) or numpy.isnan(val).any():
            features[key] = 0
        else:
            features[key] = scipy.nanmedian(val)
    return features

# CHECK IF BRIAN2 EXPLODED -> contains NaN -> set error to 9999
def check_voltage(voltage, bigerror, num_features):
    if numpy.isnan(voltage).any():
        return [bigerror] * num_features
    return []

def check_voltage2(voltage, bigerror, num_features):
    if len(voltage) != (stim_duration + 2*stim_start)/dt:
        return [bigerror] * num_features
    return []

    
# [model - median_data] / quartiledistance_data
def get_error(model_value, bio_value, bigerror, stim_current, symptom, xl_var=xl_var):
    if symptom == "peak_time":
        error = spikeDistance(model_value, stim_current)
        return error
    model_minus_experiment = float(abs(model_value - bio_value))
    if model_value == bio_value:
        error = 0.
    elif model_value == 0:
        error = bigerror
    elif model_value < bio_value:
        error = model_minus_experiment / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, symptom].values[0][0]))
    elif model_value > bio_value:
        error = model_minus_experiment / (1 + float(xl_var.loc[xl_var['stimulus'] == stim_current, symptom].values[0][1]))
    return error

# CONSTRAINT FUNCION
def feasible(individual, low=low, up=up):
    fsbl = [1 if val >= l and val <= u else 0 for val, l, u in zip(individual, low, up)]
    return fsbl.count(1) == len(fsbl)

def deathpenalty(individual, num_features=num_features):
    inlimits = feasible(individual)
    if not inlimits:
        return [bigerror] * num_features
    return []

# efficiency
def starttime(key):
    if not key in allinfo:
        return time.time()

def endtime(key, starttime):
    if not key in allinfo:
        allinfo[key] = [round(time.time() - starttime, 2)]
        return allinfo
    elif key == "time_hypervolume":
        allinfo[key].append(round(time.time() - starttime, 2))

# ERROR FUNCTION
def error_function(stim_currents, indi, xl_mean=xl_mean, xl_var=xl_var):
    start_time_error_function = starttime("time_error_function")
    
    allerrors = [[] for _ in stim_currents]
    # For all stimuli [nA]
    for i, stim_current in enumerate(stim_currents):

        # Check feasability of channel values
        allerrors[i] = deathpenalty(indi)
        if allerrors[i]:
            #return tuple(allerrors)
            continue

        # BRIAN: run model
        channel_params    = dict(zip(channels.keys(), indi))
        start_time_model  = starttime("time_model")
        Iext              = buildStimVec(stim_current, stim_start, stim_duration, dt)
        voltage, timevec  = hhModel(dictToListParams(channel_params), Iext, dt, fI)
        endtime("time_model", start_time_model)

        # Check if brian model exploded, if so, break error function
        allerrors[i] = check_voltage(voltage, bigerror, num_features)
        if allerrors[i]:
            #return tuple(allerrors)
            continue
    
        # EFEL: extract features and get median
        trace           = {'T': timevec, 'V': voltage, 'stim_start': [stim_start], 'stim_end': [stim_start + stim_duration]}
        start_time_efel = starttime("time_efel")
        features_raw    = efel.getFeatureValues([trace], feature_list)[0]
        endtime("time_efel", start_time_efel)
        features        = efel_median(features_raw)
    
        # ERROR FUNCTION: GET ERROR VALUE
        for symptom, model_value in sorted(features.items()):
            # get median of biological data
            bio_value = xl_mean.loc[xl_mean['stimulus'] == stim_current, symptom].values[0]
            # check model against biological data
            error     = get_error(model_value, bio_value, bigerror, stim_current, symptom)
            # append error value of this feature
            allerrors[i].append(error)
           
    allerrors = np.max(allerrors, axis=0)
    endtime("time_error_function", start_time_error_function)
    #if not "time_error_function" in allinfo:
    #    allinfo["time_error_function"] = time.time() - start_time_error_function
    return tuple(allerrors)

# GET FEATURE ORDER
def get_featureorder(stim_current, indi):
    # BRIAN: run model
    channel_params = dict(zip(channels.keys(), indi))
    Iext = buildStimVec(stim_current, stim_start, stim_duration, dt)
    voltage, timevec = hhModel(dictToListParams(channel_params), Iext, dt, fI)
    # EFEL: extract features and get median
    trace = {'T': timevec, 'V': voltage, 'stim_start': [stim_start], 'stim_end': [stim_start + stim_duration]}
    features_raw = efel.getFeatureValues([trace], feature_list)[0]
    features = efel_median(features_raw)
    return list(sorted(features.keys()))


        









### SAVE DATA
###############################################################################
    
# CHECK IF FILE EXISTS
def version_control(name, type):
    file = name + "." + type
    version = 1
    while os.path.isfile(file):
        version += 1
        if version < 10:
            file = "%s_0%d.%s" %(name, version, type)
        else:
            file = "%s_%d.%s" %(name, version, type)
    return file

# CREATE NEW FOLDER
def make_path(gen, pathout, gens_to_save=gens_to_save, allinfo=allinfo):
    
    # create new folder v01
    def change_folder(name):
        folder    = name + "_01"
        directory = os.path.join(pathout, folder)
        # create folder v01 if v01 doesn't exists
        if not os.path.exists(directory):
            os.makedirs(directory)
            return directory
        # v01 -> v02 if v01 exists
        version = 1
        while os.path.isdir(directory):
            version += 1
            if version < 10:
                folder = "%s_0%d" %(name, version)
            else:
                folder = "%s_%d" %(name, version)
            directory = os.path.join(pathout, folder)
        # create v02
        os.makedirs(directory)
        return directory
            
    # create new folder v01, v02, v03, whatever you need
    if gen == 0:
        pathout = change_folder(foldername)
    # create new folder gen01, gen10, gen20, whatever you need
    if gen in gens_to_save:
        if gen < 10:
            directory = os.path.join(pathout, "gen0%s" %gen)
        else:
            directory = os.path.join(pathout, "gen%s" %gen)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
    return pathout

# SAVE FINAL POPULATION (EXCEL)
#def save_pop(hof, sortfeatures, channelnames):
#    import pandas as pd
#    import os.path
#    body    = [ind + list(ind.fitness.values) for ind in hof]
#    header  = channelnames + sortfeatures
#    summary = pd.DataFrame(body, columns=header)
#    file = version_control("finalpop", "xlsx")
#    summary.to_excel(file)

# SAVE FITNESS OF FINAL POPULATION (JSON)
def save_fitness(pop, num_gen, allfit, gen=0, gens_to_save=gens_to_save):
    fit = [list(ind.fitness.values) for ind in pop]
    fit = [list(x) for x in zip(*fit)]
    allfit = allfit + fit
    if gen in gens_to_save:
        file = version_control("fit", "json")
        with open(file, "w") as f:
            f.write(json.dumps(allfit))
    return allfit

# SAVE FINAL POPULATION (JSON)
def save_values(pop, num_gen, children, gen=0, gens_to_save=gens_to_save):
    child = [list(ind) for ind in pop]
    child = [list(x) for x in zip(*child)]
    children = children + child
    if gen in gens_to_save:
        file = version_control("values", "json")
        with open(file, "w") as f:
            f.write(json.dumps(children))
    return children

# SAVE PARETO FRONT (JSON)
def save_pareto(pop, k, gen, sortfeatures, gens_to_save=gens_to_save, num_gen=num_gen, channels=channels, stim_start=stim_start, stim_duration=stim_duration, stim_currents=stim_currents):
    if gen in gens_to_save:
        # Paretofront
        pareto = tools.sortNondominated(pop, k, first_front_only=True)[0]
        print("Paretolength before Hypervolume: %s" %len(pareto), " ... reducing to %s" %num_finalpop, "...")
        # Hypervolume
        #start_time_hypervolume = starttime("time_hypervolume")
        start_time_hypervolume = time.time()                  # measure efficiency
        while len(pareto)>num_finalpop:                       # select by hypervolume
            i = tools.indicator.hypervolume(pareto)
            del pareto[i]
        endtime("time_hypervolume", start_time_hypervolume)   # measure efficiency
            #print(len(pareto))
        # Get Fitness, Pop and Model
        #print("Paretolength after Hypervolume: %s" %len(pareto))
        parfit  = [list(ind.fitness.values) for ind in pareto]
        parpop  = [ind for ind in pareto]
        #par    = {"pop": parpop, "fitness": parfit, "voltage": v, "time": timevec.tolist(), "nA": stim_currents}
        par    = {"channels": list(channels.keys()), "objectives": sortfeatures,"pop": parpop, "fitness": parfit, "nA": stim_currents}
        # Save to pareto.json
        file = version_control("pareto", "json")
        with open(file, "w") as f:
                f.write(json.dumps(par))
                
# save all information
def save_info(allinfo):
    file = version_control("allinfo", "txt")
    file = os.path.join(pathout, file)
    with open(file, "w") as f:
        for key, val in allinfo.items():
            f.write(key + ": " + str(val) + "\n")







### PREPARE EA
###############################################################################
# CREATOR:          http://deap.readthedocs.io/en/master/tutorials/basic/part1.html
# TOOLBOX:          http://deap.readthedocs.io/en/master/examples/ga_onemax.html
# TOOLBOX:          http://deap.readthedocs.io/en/master/api/tools.html#module-deap.tools
# TOOLBOX.REGISTER: http://deap.readthedocs.io/en/master/api/tools.html#module-deap.tools
            
# HOW TO CREATE INDIVIDUALS
def initIndividual(container):
    return container(random.gauss(x, abs(x)/sigma) for x in channels.values())
            
# DEFINE FITNESS PROBLEM: The create() function takes at least two arguments, a name for the newly created class and a base class. Any subsequent argument becomes an attribute of the class. Neg. weights relate to minizing, pos. weight to maximizing problems. Define fitness problem (which params are min./max. problems with which weight)
creator.create("FitnessMin", base.Fitness, weights=tuple(numpy.ones(num_features) * -1))

# ASSOCIATE FITNESS PROBLEM TO INDIVIDUALS: Next we will create the class Individual, which will inherit the class list and contain our previously defined. FitnessMulti class in its fitness attribute. Note that upon creation all our defined classes will be part of the creator container and can be called directly.
creator.create("Individual", list, fitness=creator.FitnessMin)

# TOOLBOX: All the objects we will use on our way, an individual, the population, as well as all functions, operators, and arguments will be stored in a DEAP container called Toolbox. It contains two methods for adding and removing content, register() and unregister().
toolbox = base.Toolbox()

# REGISTER HOW INDIVUALS ARE CREATED
toolbox.register("individual", initIndividual, creator.Individual)

# REGISTER HOW POPULATION IS CREATED
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# CROSSOVER
toolbox.register("mate", tools.cxSimulatedBinary, eta=eta)

# MUTATION
toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta, low=low, up=up, indpb=prob_mutation)

# SELECTION
toolbox.register("select", tools.selNSGA2)

# EVALUATION
toolbox.register("evaluate", error_function, stim_currents)

# CONSTRAINT
#toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, bigerror))

# PARALLEL PROCESSING
if parallel_processing=="yes":
    toolbox.register("map", futures.map)







### Evolutionary algorithm
###############################################################################
# NSGA2: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py    
def main():
    global pathout, sortfeatures
    
    # GENERATON ZERO
    # register statistics to the toolbox to maintain stats of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg"
    # create random parent population pop
    pop = toolbox.population(n=num_parents)
    # features change in position
    sortfeatures = get_featureorder(stim_currents[0], pop[0])
    #print important EA settings
    #print("FEATURES", sortfeatures)
    #print("CHANNELS", channelnames)
    # evaluate parent population
    #invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # assign crowding distance to the individuals, no actual selection is done
    pop = toolbox.select(pop, len(pop))
    # statistics of fitness values
    record = stats.compile(pop)
    # print logbook
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)
    # plot and save genereation 0
    #if plotflag == "yes":
    #    pbm_allerrors('all', 0, pop, sortfeatures, channels, stim_current, num_gen, gens_to_plot=gens_to_plot)
    if saveflag == "yes":
        pathout  = make_path(0, pathout)
        allfit   = save_fitness(pop, num_gen, [sortfeatures])
        children = save_values(pop, num_gen, [channelnames])
        save_pareto(pop, num_finalpop, 0, sortfeatures)   
                  

    # GENERATION 1-100
    for gen in range(1, num_gen+1):

        # increase the variance in my population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # so that offspring is not just the reference to pop but a proper variable
        offspring = [toolbox.clone(ind) for ind in offspring]

        # CROSSOVER
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            #clone ind1 and ind2
            clone1, clone2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
            # take always pairs of two children
            if random.random() <= prob_crossover:
                toolbox.mate(ind1, ind2)
            # MUTATION
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            # if mutation or crossover took place, delete fitness values
            if clone1 != ind1: del ind1.fitness.values
            if clone2 != ind2: del ind2.fitness.values

        # fitness: shall this individual be re-evaluated?
        not_yet_evaluated = [ind for ind in offspring if not ind.fitness.valid]
        # EVALUATION
        fitnesses = toolbox.map(toolbox.evaluate, not_yet_evaluated)
        #fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(not_yet_evaluated, fitnesses):
            ind.fitness.values = fit
        # SELECTION
        pop = toolbox.select(pop + offspring, num_parents)
        # print
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
        print(logbook.stream)
        # plot genereation 1 onwards
        #if plotflag == "yes":
        #    pbm_allerrors('all', gen, pop, sortfeatures, channels, stim_current, num_gen, gens_to_plot=gens_to_plot)
        if saveflag == "yes":
            make_path(gen, pathout)
            allfit   = save_fitness(pop, num_gen, allfit, gen=gen)   # fitness
            children = save_values(pop, num_gen, children, gen=gen)  # ion channels
            save_pareto(pop, num_finalpop, gen, sortfeatures)        # pareto
            
        trackOneObjective('steady_state_voltage_stimend', gen, pop, sortfeatures, channels, stim_currents[0], num_gen, dt, fI)
    return pop, logbook







### RUN
###############################################################################

if __name__ == "__main__":
    for i in range(num_rep):
        start_time_EA = time.time()
        pop, logbook = main()
        print(logbook)
        
        # efficiency
        allinfo['time_EA']            = time.time() - start_time_EA
        #allinfo['time_model']         = timeit.timeit(brian_hh_model(stim_currents[0], 0, stim_start, stim_duration, dict(zip(channels.keys(), pop[0]))), number=1)
        #voltage, timevec = brian_hh_model(stim_currents[0], 0, stim_start, stim_duration, dict(zip(channels.keys(), pop[0])))
        #trace = {'T': timevec, 'V': voltage, 'stim_start': [stim_start], 'stim_end': [stim_start + stim_duration]}
        #allinfo['time_efel']          = timeit.timeit(efel.getFeatureValues([trace], feature_list), number=1)
        #allinfo['time_hypervolume']   = start_time_hypervolume - end_time_hypervolume
        #allinfo['time_error_function] = 
        save_info(allinfo)
        
        # PARETOFRONT
        #paretofront = tools.sortNondominated(pop, num_finalpop, first_front_only=True)[0]
            
        # Plot Final Population
        #pbm_final(paretofront, channels, stim_current, sortfeatures, rows=5)
        #Plot Paretofront
        #pbm_paretofront(paretofront, pareto_features, sortfeatures)
        # Plot channel distribution
        #pbm_channels_sub(paretofront, channels)
        # Save plots
        #file = version_control("allplots", "pdf"); multipage(file)
        
        # SAVE FINAL POP AND PARETO        
        #if saveflag == "yes":
            #save_pop(paretofront, sortfeatures, channelnames)
            #st1 = time.time()
            #print("Paretolength before Hypervolume: %s" %len(paretofront))
            #save_pareto(paretofront)
            #print("save_pareto time: ", time.time()-st1)

        # close all figures
        #plt.close("all")
        
        if parallel_processing=="no":
            import winsound
            winsound.Beep(2000, 500); winsound.Beep(1500, 500)
#
#
    