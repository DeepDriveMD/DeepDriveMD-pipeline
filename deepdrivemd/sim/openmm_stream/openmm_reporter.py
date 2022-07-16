import datetime
import hashlib
import sys
from typing import Dict

import adios2
import MDAnalysis
import numpy as np
from MDAnalysis.analysis import distances, rms

from deepdrivemd.utils import hash2intarray, timer


class ContactMapReporter(object):
    """Periodically reports the results of the openmm simulation"""

    def __init__(self, reportInterval, cfg):
        self._reportInterval = reportInterval
        print(cfg)
        print(f"report interval = {reportInterval}")
        print("ContactMapRepoter constructor")
        self._adios_stream = cfg._adios_stream

        self.step = 0
        self.cfg = cfg

        if cfg.compute_zcentroid or cfg.compute_rmsd:
            self.universe_init = MDAnalysis.Universe(self.cfg.init_pdb_file)
        if cfg.compute_zcentroid:
            self.heavy_atoms = self.universe_init.select_atoms(self.cfg.zcentroid_atoms)
            self.heavy_atoms_indices = self.heavy_atoms.indices
            self.heavy_atoms_masses = self.heavy_atoms.masses
        if cfg.compute_rmsd:
            self.rmsd_positions = self.universe_init.select_atoms(
                self.cfg.mda_selection
            ).positions.copy()

        self._adios_file = adios2.open(
            name=str(self.cfg.current_dir) + "/trajectory.bp",
            mode="w",
            config_file=str(cfg.adios_xml_file),
            io_in_config_file="Trajectory",
        )

    def __del__(self):
        print("ContactMapRepoter destructor")
        self._adios_file.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)

    def zcentroid(self, positions):
        return np.average(
            positions[self.heavy_atoms_indices, 2], weights=self.heavy_atoms_masses
        )

    def report(self, simulation, state):
        """Computes contact maps, md5 sum of positions, rmsd to the reference state and records them into `_adios_stream`"""
        timer("reporting", 1)
        step = self.step
        stateA = simulation.context.getState(getPositions=True, getVelocities=True)
        ca_indices = []
        natoms = 0
        for atom in simulation.topology.atoms():
            natoms += 1
            if atom.name == self.cfg.openmm_selection[0]:
                ca_indices.append(atom.index)

        positions = state.getPositions(asNumpy=True).astype(np.float32)


        if self.cfg.compute_zcentroid:
            centroid = np.array(self.zcentroid(positions), dtype=np.float32)
            print(f"centroid = {centroid}")
            sys.stdout.flush()

        velocities = stateA.getVelocities(asNumpy=True)

        velocities = np.array(
            [[x[0]._value, x[1]._value, x[2]._value] for x in velocities]
        ).astype(np.float32)

        m = hashlib.sha512()
        m.update(positions.tostring())
        md5 = m.hexdigest()
        md5 = hash2intarray(md5)

        positions_ca = positions[ca_indices].astype(np.float32)
        point_cloud = positions_ca.copy()

        d = positions_ca.shape[0]
        if not (positions_ca.shape[0] % self.cfg.divisibleby == 0):
            d = positions_ca.shape[0] // self.cfg.divisibleby * self.cfg.divisibleby
            positions_ca = positions_ca[:d]

        print(f"len(ca_indices) = {len(ca_indices)}, d = {d}, natoms = {natoms}")
        sys.stdout.flush()

        if self.cfg.model == "cvae":
            contact_map = distances.contact_matrix(
                positions_ca, cutoff=self.cfg.threshold, returntype="numpy", box=None
            ).astype("uint8")

        step = np.array([step], dtype=np.int32)
        gpstime = np.array([int(datetime.datetime.now().timestamp())], dtype=np.int32)

        output = {
            "md5": md5,
            "step": step,
            "positions": positions,
            "velocities": velocities,
            "gpstime": gpstime,
        }

        if self.cfg.model == "cvae":
            output["contact_map"] = contact_map
        elif self.cfg.model == "aae":
            output["point_cloud"] = point_cloud

        if self.cfg.compute_zcentroid:
            output["zcentroid"] = centroid

        if self.cfg.compute_rmsd:
            reference_positions = self.rmsd_positions[:d].copy()
            rmsd = rms.rmsd(positions_ca, reference_positions, superposition=True)
            rmsd = np.array([rmsd], dtype=np.float32)
            output["rmsd"] = rmsd

        if (
            hasattr(self.cfg, "multi_ligand_table")
            and self.cfg.multi_ligand_table.is_file()
        ):
            output["ligand"] = np.array([self.cfg.ligand], dtype=np.int32)
            output["natoms"] = np.array([natoms], dtype=np.int32)

        self.write_adios_step(output)
        timer("reporting", -1)
        self.step += 1

    def write_adios_step(self, output: Dict[str, np.ndarray]):
        """Write a step into `_adios_stream`

        Parameters
        ----------
        output : Dict[str, np.ndarray]
             key - adios column name to which to write a value of the dictionary
             representing one step

        """
        for k, v in output.items():
            if k == "gpstime":
                continue
            self._adios_stream.write(
                k, v, list(v.shape), [0] * len(v.shape), list(v.shape)
            )
        self._adios_stream.end_step()

        for k, v in output.items():
            self._adios_file.write(
                k, v, list(v.shape), [0] * len(v.shape), list(v.shape)
            )
        self._adios_file.end_step()
