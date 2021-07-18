import simtk.openmm.app as app
import simtk.unit as u 
import os
from MDAnalysis.analysis import distances, rms
import MDAnalysis
import numpy as np 
from deepdrivemd.utils import t1Dto2D, t2Dto1D, hash2intarray

import adios2
import hashlib

class ContactMapReporter(object):
    def __init__(self, reportInterval, cfg):
        self._reportInterval = reportInterval
        print(cfg)
        print(f"report interval = {reportInterval}")
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
        """
        Computes contact maps, md5 sum of positions, rmsd to the reference state and records them into self._adios_stream
        """
        step = self.step
        stateA = simulation.context.getState(getPositions=True, getVelocities=True)
        ca_indices = []
        for atom in simulation.topology.atoms():
            if atom.name == self.cfg.openmm_selection[0]:
                ca_indices.append(atom.index)

        positions = np.array(state.getPositions().value_in_unit(u.angstrom)).astype(np.float32)
        velocities = stateA.getVelocities(asNumpy=True)

        velocities = np.array([ [ x[0]._value, x[1]._value, x[2]._value ] for x in velocities ]).astype(np.float32)

        m = hashlib.md5()
        m.update(positions.tostring())
        md5 = m.hexdigest()
        md5 = hash2intarray(md5)

        positions_ca = positions[ca_indices].astype(np.float32)
        contact_map = distances.contact_matrix(positions_ca, cutoff=self.cfg.threshold, returntype='numpy', box=None).astype('uint8')
        contact_map = t2Dto1D(contact_map)
        contact_map = np.packbits(contact_map)
        
        mda_u = MDAnalysis.Universe(str(self.cfg.reference_pdb_file))
        reference_positions = mda_u.select_atoms(self.cfg.mda_selection).positions.copy()
        rmsd = rms.rmsd(positions_ca, reference_positions, superposition=True)
        step = np.array([step], dtype=np.int32)
        rmsd = np.array([rmsd], dtype=np.float32)

        output = {"md5" : md5, "step" : step, "rmsd" : rmsd, "positions" : positions, "velocities": velocities, "contact_map" : contact_map}
        self.write_adios_step(output)
        self.step += 1

    def write_adios_step(self, output):
        """
        output is a dictionary: adios column name, variable name
        The function writes a row into self._adios_stream.
        """
        for k in output:
            v = output[k]
            self._adios_stream.write(k, v, list(v.shape), [0]*len(v.shape), list(v.shape))
        self._adios_stream.end_step()
