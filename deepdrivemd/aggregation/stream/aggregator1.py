import adios2
import numpy as np
import subprocess
#import logging
from myutils import *
import time
import random
import sys
import glob
import os
from utils import format_cm
from OutlierDB import *
import pickle
import mytimer
import threading
from lockfile import LockFile
import queue
from myconvert import *
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start_unit", dest="start_unit", type=int, help="start of unit range")
parser.add_argument("-e", "--end_unit", dest='end_unit', type=int, help="end of unit range")
parser.add_argument("-o", "--output_postfix" , dest='output_postfix', help="postfix of the aggregated bp file")
parser.add_argument("-c", "--current_dir", dest='current_dir', help="current directory")
parser.add_argument("-r", "--run_dir", dest='run_dir', help="running directory")
parser.add_argument("-m", "--max_iterations", dest='max_iterations', type=int, help="maximum number of iterations")
args = parser.parse_args() 


kill_prob1 = 0.85
kill_prob2 = 0.995

def q_kill_simulation(db, sim_md5, lastN):
    return True
    '''
    if(db == None):
        return random.random() > kill_prob2
    if(len(sim_md5) == 0):
        return random.random() > kill_prob2
    '''
    #print("In q_kill_simulation")
    #print(f"type(sim_md5)={type(sim_md5)}")
    #print(f"sim_md5={sim_md5}")
    #print(f"lastN={lastN}"); sys.stdout.flush()
    sim = set(sim_md5[-lastN:])
    #print(f'sim = {sim}')
    #print(f'len(sim)={len(sim)}'); sys.stdout.flush()
    outliers = set(db.dictionary.keys())
    #print(f'outliers = {outliers}')
    #print(f'len(outliers)={len(outliers)}'); sys.stdout.flush()
    intersection = sim & outliers
    #print(f'intersection = {intersection}')
    size = len(intersection)
    print(f'len(intersection) = {size}'); sys.stdout.flush()

    if(size) == 0:
        return random.random() > kill_prob1
    else:
        return False

def get_outliers(dir):
    #print('In get_outliers'); sys.stdout.flush()
    fn = f'{dir}/OutlierDB.pickle'
    if(not os.path.exists(fn)):
        print('In get_outliers:return 1'); sys.stdout.flush()
        return None
    mylock1 = LockFile(fn)
    # my_lock(fn)
    mylock1.acquire()
    try:
        with open(fn, 'rb') as f:
            db = pickle.load(f)
    except:
        db = None
    # my_unlock(fn)
    mylock1.release()
    #print('Getting out of get_outliers'); sys.stdout.flush()
    return db



def outliers_changed(dir):
    #print('In outliers_changed'); sys.stdout.flush()
    global previous_db_md5
    dbfn = f'{dir}/OutlierDB.pickle'
    if(not os.path.exists(dbfn)):
        #print('In outliers_changed:return 1'); sys.stdout.flush()        
        return False
    #print('In outliers_changed:return 1a'); sys.stdout.flush()        

    md5 = os.stat(dbfn).st_mtime
    # md5 = subprocess.getstatusoutput(f'md5sum {dbfn}')[1].split(" ")[0]
    #print('In outliers_changed:return 1b'); sys.stdout.flush() 
    if(md5 == previous_db_md5):
        #print('In outliers_changed:return 2'); sys.stdout.flush()        
        return False
    previous_db_md5 = md5
    #print('In outliers_changed:return 3'); sys.stdout.flush()        
    return True

def mytop():
    print("="*10 + " top " + "="*10)
    user = os.getenv('USER')
    print(subprocess.getstatusoutput(f"top -U {user} -b -n 1")[1])
#    print("="*5)
#    print(subprocess.getstatusoutput("nvidia-smi")[1])
    print("="*25); sys.stdout.flush()


def get_simulations(stype):
    simulations = glob.glob(f"{dir_simulations}/{stype}/*")
    final_simulations = []
    srange = list(range(args.start_unit, args.end_unit + 1))
    for s in simulations:
        d = int(s.split("_")[-1])
        if d in srange:
            final_simulations.append(s)
    return final_simulations

if(__name__ == '__main__'):
    current_dir = args.current_dir
    run_dir = args.run_dir
    MAX_ITERATIONS = args.max_iterations
    dir_aggregator = f'{run_dir}/aggregate'
    dir_simulations = f'{current_dir}/MD_exps/fs-pep'
    dir_outliers = f'{current_dir}/Outlier_search'
    ADIOS_XML = f'{current_dir}/adios.xml'
    ADIOS_XML_AGGREGATOR = f'{dir_aggregator}/adios.xml'
    bpaggregator = f'{dir_aggregator}/aggregator{args.output_postfix}.bp'
    bpfile = 'cms_positions.bp'
    dotsst = '.sst'
    lastN = 1000
    previous_db_md5 = -1
    db = None

    print("hostname = ", subprocess.getstatusoutput("hostname")[1])


    #logging.basicConfig(filename=f'{dir_aggregator}/aggregator.log', filemode='w', level=logging.INFO)
    print("Start")
    print(get_now())
    print(f"current_dir = {current_dir}")
    print(f"dir_simulations={dir_simulations}")
    print(f"dir_aggregator={dir_aggregator}")

    sim_streams = {}
    sim_cm = {}
    sim_positions = {}
    sim_md5 = {}

    max_iterations = MAX_ITERATIONS

    aggregator_stream = adios2.open(name=bpaggregator,
                                    mode="w", config_file=ADIOS_XML_AGGREGATOR,
                                    io_in_config_file="AggregatorOutput")

    #print(subprocess.getstatusoutput(f"rm -f {current_dir}/clean.done"))

    while(max_iterations > 0):
        #mytop()
        mytimer.mytime_label("aggregator_iteration", 1)

        print("="*30)
        print(f"max_iterations = {max_iterations}")
        new_simulations = get_simulations("new")
        #print("Before potential lock 1")
        while(len(new_simulations)==0 and max_iterations == MAX_ITERATIONS):
            time.sleep(2)
            print("Sleeping for new simulations")
            new_simulations = get_simulations("new")

        #print("After potential lock 1")

        #print("="*30)
        #print("new_simulations = ")
        #print(new_simulations)
        print(f"There are {len(new_simulations)} new_simulations")
        #print("="*30)

        running_simulations = get_simulations("running")

        #print("="*30)
        #print("running_simulations = ")
        #print(running_simulations)
        print(f"There are {len(running_simulations)} running_simulations")
        #print("="*30)

        #print("Before while")


        mytimer.mytime_label("aggregator_wait1", 1)

        while(len(new_simulations + running_simulations) == 0):
            print("No new or running simulations left. Waiting for new simulations...")
            sys.stdout.flush()
            time.sleep(20)
            new_simulations = get_simulations("new")
            running_simulations = get_simulations("running")

            '''
            print("No new or running simulation left. Exiting ...")
            break
            '''

        mytimer.mytime_label("aggregator_wait1", -1)


        #print("After while")

        mytimer.mytime_label("aggregator_wait2", 1)

        for n in new_simulations:
            r = n.replace("new","running")
            a = n.replace("new","all")
            #print("Before potential lock 2");sys.stdout.flush()
            while(not os.path.exists(f"{a}/{bpfile}{dotsst}")):
                print(f"Waiting for {a}/{bpfile}{dotsst}"); sys.stdout.flush()
                time.sleep(2);sys.stdout.flush()
            #print("After potential lock 2")
            subprocess.getstatusoutput(f"mv {n} {r}")
            #print("Before potential lock 5")
            while(True):
                try:
                    #adios = adios2.ADIOS(ADIOS_XML, True)
                    #io = adios.DeclareIO("SimulationOutput")

                    # adios.RemoveAllIOs()
                    #print(os.path.basename(a)); sys.stdout.flush()
                    while(not os.path.exists(f"{a}/{bpfile}{dotsst}")):
                        time.sleep(2)
                    ADIOS_XML =  f"{a}/adios.xml"
                    #print(f"ADIOS_XML={ADIOS_XML}")
                    adios = adios2.ADIOS(ADIOS_XML, True)
                    io = adios.DeclareIO(os.path.basename(a))
                    io.SetParameters({"ControlModule":"epoll"})
                    stream = io.Open(f"{a}/{bpfile}", adios2.Mode.Read)

                    '''

                    simulation_stream = adios2.open(name=f"{a}/{bpfile}",
                                                    mode="r", config_file=ADIOS_XML,
                                                    io_in_config_file="SimulationOutput")
                    '''
                    #print(f"Opened {a}/{bpfile}")
                    break
                except Exception as e:
                    print(e)
                    print(f"Cannot open {a}/{bpfile}{dotsst}; trying again")
                    qexists = os.path.exists(f"{a}/{bpfile}{dotsst}")
                    print(f"File exists: {qexists}") 
                    continue
            #print("After potential lock 5")
            rb = os.path.basename(r)
            # sim_streams[rb] = simulation_stream
            # sim_streams[rb] = (adios, io, stream)
            sim_streams[rb] = (adios, io, stream)
            
        mytimer.mytime_label("aggregator_wait2", -1)


        remove = []
        mytimer.mytime_label("aggregator_internal_loop", 1)

        q = queue.Queue()
        for sim_dir in sim_streams.keys():
            q.put(sim_dir)

        qs = q.qsize()
        qq = 0

        while(not q.empty()):
            qs = q.qsize()
            sim_dir = q.get()
            qq += 1
            if(len(sim_streams) > 200 or qs > 200):
                print(f"qq={qq}, qs={qs}, len(sim_streams)={len(sim_streams)}, {sim_dir}"); sys.stdout.flush()


            # mytop()

            # adios, io, stream = sim_streams[sim_dir]
            adios,io,stream = sim_streams[sim_dir]

            #print(f"after triple"); sys.stdout.flush()


            status = stream.BeginStep(adios2.StepMode.Read, 0.0)

            if(status == adios2.StepStatus.NotReady):
                q.put(sim_dir)
                continue
            if(status == adios2.StepStatus.EndOfStream):
                print(f"{sim_dir} reached end of stream")
                try:
                    stream.Close()
                    io.RemoveAllVariables()
                    adios.RemoveAllIOs()
                except Exception as e:
                    print("Exception: EndOfStream")
                    print(e); sys.stdout.flush()
                remove.append(sim_dir)
                subprocess.Popen(["touch",f"{dir_simulations}/all/{sim_dir}/stop.simulation"])
                try:
                    subprocess.getstatusoutput(f"mv {dir_simulations}/running/{sim_dir} {dir_simulations}/stopped/{sim_dir}") 
                except:
                    pass
                continue
            if(status == adios2.StepStatus.OtherError):
                print(f"{sim_dir} encountered an error")
                try:
                    stream.Close()
                    io.RemoveAllVariables()
                    adios.RemoveAllIOs()
                except Exception as e:
                    print("Exception: OtherError")
                    print(e); sys.stdout.flush()
                remove.append(sim_dir)
                subprocess.Popen(["touch",f"{dir_simulations}/all/{sim_dir}/stop.simulation"])
                try:
                    subprocess.getstatusoutput(f"mv {dir_simulations}/running/{sim_dir} {dir_simulations}/stopped/{sim_dir}") 
                except:
                    pass
                continue                
            
            mytimer.mytime_label("aggregator_read", 1)
            
            stepA = np.zeros(1, dtype=np.int32)
            varStep = io.InquireVariable("step")
            stream.Get(varStep, stepA)

            varCM = io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0]*ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.int32)
            stream.Get(varCM, cm)

            varPositions = io.InquireVariable("positions")
            shapePositions = varPositions.Shape()
            ndimPositions = len(shapePositions)
            start = [0]*ndimPositions
            count = shapePositions
            varPositions.SetSelection([start, count])
            positions = np.zeros(shapePositions, dtype=np.float64)
            stream.Get(varPositions, positions)

            varVelocities = io.InquireVariable("velocities")
            shapeVelocities = varVelocities.Shape()
            ndimVelocities = len(shapeVelocities)
            start = [0]*ndimVelocities
            count = shapeVelocities
            varVelocities.SetSelection([start, count])
            velocities = np.zeros(shapeVelocities, dtype=np.float64)
            stream.Get(varVelocities, velocities)

            varMD5 = io.InquireVariable("md5")
            shapeMD5 = varMD5.Shape()
            ndimMD5 = len(shapeMD5)
            start = [0]*ndimMD5
            count = shapeMD5
            varMD5.SetSelection([start, count])
            md5 = np.zeros(shapeMD5, dtype=np.int64)
            stream.Get(varMD5, md5)

            stream.EndStep()

            
            #print(f"shapeCM = {shapeCM}") 
            #print(f"shapePositions = {shapePositions}")
            #print(f"shapeMD5 = {shapeMD5}")
            #sys.stdout.flush()

            """
            print(f"Before format: type(cm)={type(cm)}, cm.shape={cm.shape}")

            skip10 = 1
            count10 = 27
            i10 = 0
            while(i10 < cm.shape[0]):
                for k10 in range(skip10):
                    print(" ", end = ' ')
                skip10 += 1
                for k10 in range(count10):
                    print(cm[i10], end = ' ')
                    i10 += 1
                count10 -= 1
                print("")
            """

            step = stepA[0]
            md5 = intarray2hash(md5)

            # cm1 = cm.copy()

            cm = format_cm(cm)

            """
            print(f"After format: type(cm)={type(cm)}, cm.shape={cm.shape}")

            for k10 in range(cm.shape[0]):
                for k11 in range(cm.shape[1]):
                    print(f"{int(cm[k10][k11][0])}", end=' ')
                print("")
            """

            #print(f"step={step}")
            #print(f"type(md5) = {type(md5)}"); sys.stdout.flush()
            #print(f"md5 = {md5}"); sys.stdout.flush()

            mytimer.mytime_label("aggregator_read", -1)

            mytimer.mytime_label("aggregator_write", 1)
            aggregator_stream.write("md5", md5)
            aggregator_stream.write("step", np.array([step]))
            aggregator_stream.write("dir", sim_dir)
            aggregator_stream.write("positions", positions, list(positions.shape), [0]*len(positions.shape), list(positions.shape))
            aggregator_stream.write("velocities", velocities, list(velocities.shape), [0]*len(velocities.shape), list(velocities.shape))
            aggregator_stream.write("contact_map", cm, list(cm.shape), [0]*len(cm.shape), list(cm.shape), end_step=True)
            mytimer.mytime_label("aggregator_write", -1)
            #print("here:7"); sys.stdout.flush()

            '''
            try:
                sim_cm[sim_dir].append(cm)
            except:
                sim_cm[sim_dir] = [cm]
            try:
                sim_positions[sim_dir].append(positions)
            except:
                sim_positions[sim_dir] = [positions]
            '''
            try:
                sim_md5[sim_dir].append(md5)
            except:
                sim_md5[sim_dir] = [md5]

            '''
            sim_cm[sim_dir] = sim_cm[sim_dir][:lastN]
            sim_positions[sim_dir] = sim_positions[sim_dir][:lastN]
            '''
            sim_md5[sim_dir] = sim_md5[sim_dir][-lastN:]

            # print("here:8"); sys.stdout.flush()

            '''
            if(outliers_changed(dir_outliers)):
                print("Before get_outliers")
                db = get_outliers(dir_outliers)
            db = None
            '''

            # print("here:8a"); sys.stdout.flush()

            mytimer.mytime_label("aggregator_kill", 1)

            if(q_kill_simulation(db, sim_md5[sim_dir], lastN)):
                print(f"Killing the simulation in {sim_dir}")
                subprocess.Popen(["touch",f"{dir_simulations}/all/{sim_dir}/stop.simulation"])
                adios,io,stream = sim_streams[sim_dir]
                '''
                try:
                    stream.Close()
                    io.RemoveAllVariables()
                    adios.RemoveAllIOs()
                except Exception as e:
                    print("Exception: kill")
                    print(e); sys.stdout.flush()
                remove.append(sim_dir)
                subprocess.getstatusoutput(f"mv {dir_simulations}/running/{sim_dir} {dir_simulations}/stopped/{sim_dir}") 
                '''
                # print("here:9"); sys.stdout.flush()
            #print("here:10"); sys.stdout.flush()
            mytimer.mytime_label("aggregator_kill", -1)
        mytimer.mytime_label("aggregator_internal_loop", -1)
        for r in remove:
            del sim_streams[r]
        mytimer.mytime_label("aggregator_iteration", -1)
        max_iterations -= 1

    subprocess.getstatusoutput(f"touch {dir_aggregator}/stop.aggregator")
        
    for sim_dir in sim_streams.keys():
        subprocess.Popen(["touch",f"{dir_simulations}/all/{sim_dir}/stop.simulation"])
        print(f"Stopping simulation in {sim_dir}")
        subprocess.getstatusoutput(f"mv {dir_simulations}/running/{sim_dir} {dir_simulations}/stopped/{sim_dir}")
        adios,io,stream = sim_streams[sim_dir]
        stream.Close()
        io.RemoveAllVariables()
        adios.RemoveAllIOs()

    aggregator_stream.close()


    print(get_now())
    print("Finished")


