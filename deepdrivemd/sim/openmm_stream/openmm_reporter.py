import simtk.openmm.app as app
import simtk.openmm as omm
import simtk.unit as u 
import os

import numpy as np 
import h5py 
import sys

import adios2
import hashlib

from MDAnalysis.analysis import distances
from hashconvert import *

class ContactMapReporter(object):
    def __init__(self, reportInterval, cfg):
        self._reportInterval = reportInterval
        self._adios_stream = adios2.open(name=cfg.bp_file, mode="w", 
                                         config_file=cfg.adios_cfg, 
                                         io_in_config_file=os.path.basename(os.getcwd()))
        self.step = 0
    def __del__(self):
        self._adios_stream.close()
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        step = self.step
        stateA = simulation.context.getState(getPositions=True, getVelocities=True)
        ca_indices = []
        for atom in simulation.topology.atoms():
            if atom.name == 'CA':
                ca_indices.append(atom.index)
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        velocities = stateA.getVelocities(asNumpy=True)

        velocities = np.array([ [ x[0]._value, x[1]._value, x[2]._value ] for x in velocities ])

        m = hashlib.md5()
        m.update(positions.tostring())
        md5 = m.hexdigest()
        md5 = hash2intarray(md5)

        time = int(np.round(state.getTime().value_in_unit(u.picosecond)))
        positions_ca = positions[ca_indices].astype(np.float32)
        distance_matrix = distances.self_distance_array(positions_ca)
        contact_map = np.asarray((distance_matrix < 8.0), dtype=np.int32)
        stepA = np.array([step], dtype=np.int32)

        self._adios_stream.write("md5", md5, list(md5.shape), 
                                 [0]*len(md5.shape), list(md5.shape))
        self._adios_stream.write("step", stepA, list(stepA.shape), 
                                 [0]*len(stepA.shape), list(stepA.shape))
        self._adios_stream.write("positions", positions, list(positions.shape), 
                                 [0]*len(positions.shape), list(positions.shape))
        self._adios_stream.write("velocities", velocities, list(velocities.shape), 
                                 [0]*len(velocities.shape), list(velocities.shape))
        self._adios_stream.write("contact_map", contact_map, list(contact_map.shape), 
                                 [0]*len(contact_map.shape), list(contact_map.shape), end_step=True)
        self.step += 1
