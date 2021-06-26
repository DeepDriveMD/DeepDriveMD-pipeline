import simtk.openmm.app as app
import simtk.openmm as omm
import simtk.unit as u 
import os
from MDAnalysis.analysis import distances, rms
import MDAnalysis
import numpy as np 
import h5py 
import sys

import adios2
import hashlib

from hashconvert import *

class ContactMapReporter(object):
    def __init__(self, reportInterval, cfg):
        self._reportInterval = reportInterval
        print(cfg)
        print(f"report interval = {reportInterval}")
        sys.stdout.flush()
        stream_name = os.path.basename(cfg.output_path)
        self._adios_stream = adios2.open(name=str(cfg.bp_file), mode="w", config_file=str(cfg.adios_cfg), io_in_config_file = stream_name)
        self.step = 0
        self.cfg = cfg
    def __del__(self):
        self._adios_stream.close()
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        step = self.step
        stateA = simulation.context.getState(getPositions=True, getVelocities=True)
        ca_indices = []
        pca_indices = []
        for atom in simulation.topology.atoms():
            if atom.name == 'CA':
                ca_indices.append(atom.index)

        positions = np.array(state.getPositions().value_in_unit(u.angstrom)).astype(np.float32)
        velocities = stateA.getVelocities(asNumpy=True)

        velocities = np.array([ [ x[0]._value, x[1]._value, x[2]._value ] for x in velocities ]).astype(np.float32)

        m = hashlib.md5()
        m.update(positions.tostring())
        md5 = m.hexdigest()
        md5 = hash2intarray(md5)

        time = int(np.round(state.getTime().value_in_unit(u.picosecond)))
        positions_ca = positions[ca_indices].astype(np.float32)
        # distance_matrix = distances.self_distance_array(positions_ca)
        # contact_map = np.asarray((distance_matrix < self.cfg.threshold), dtype=np.int32)

        contact_map = distances.contact_matrix(positions_ca, cutoff=self.cfg.threshold, returntype='numpy', box=None).astype('int8')
        #print(f'contact_map.dtype = {contact_map.dtype}')
        #print(contact_map)

        
        mda_u = MDAnalysis.Universe(str(self.cfg.reference_pdb_file))
        reference_positions = mda_u.select_atoms(self.cfg.mda_selection).positions.copy()
        rmsd = rms.rmsd(positions_ca, reference_positions, superposition=True)

        # print("rmsd = ", rmsd)

        # print(f"step = {step}, x0={positions[0,0]}, vx0={velocities[0,0]}")

        stepA = np.array([step], dtype=np.int32)

        rmsdA = np.array([rmsd], dtype=np.float32)


        self._adios_stream.write("md5", md5, list(md5.shape), 
                                 [0]*len(md5.shape), list(md5.shape))
        self._adios_stream.write("step", stepA, list(stepA.shape), 
                                 [0]*len(stepA.shape), list(stepA.shape))
        self._adios_stream.write("rmsd", rmsdA, list(rmsdA.shape), 
                                 [0]*len(rmsdA.shape), list(rmsdA.shape))
        self._adios_stream.write("positions", positions, list(positions.shape), 
                                 [0]*len(positions.shape), list(positions.shape))
        self._adios_stream.write("velocities", velocities, list(velocities.shape), 
                                 [0]*len(velocities.shape), list(velocities.shape))
        self._adios_stream.write("contact_map", contact_map, list(contact_map.shape), 
                                 [0]*len(contact_map.shape), list(contact_map.shape), end_step=True)
        self.step += 1
