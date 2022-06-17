from datetime import datetime
import hashlib
import sys
from typing import Dict

import adios2
import MDAnalysis
import numpy as np
from MDAnalysis.analysis import distances, rms
from deepdrivemd.config import BaseSettings

from deepdrivemd.utils import hash2intarray, timer

class ADIOSBPConfig(BaseSettings):
    name: str
    mode: str
    config_file: str
    io_in_config_file: str

class StreamReporter:
    def __init__(self, sst_stream: "adios2.adios2.File", bp_file_config: ADIOSBPConfig) -> None:
        self._sst_stream = sst_stream

        # TODO: BP file is used to write trajectory and other data to disk
        #       for long term storage. We should consider writing to standard
        #       formats such as DCD files instead.
        self._bp_file = adios2.open(**bp_file_config.dict())
        self._step = 0

    def __del__(self):
        self._bp_file.close()

    # TODO: This function could go in a ADIOS class
    def _write_adios_step(self, stream: "adios2.adios2.File", output: Dict[str, np.ndarray]) -> None:
        for k, v in output.items():
            stream.write(
                k, v, list(v.shape), [0] * len(v.shape), list(v.shape)
            )
        stream.end_step()

    def format_scalar(self, x: int, dtype: np.dtype) -> np.ndarray:
        return  np.array([x], dtype=dtype)

    def write_adios_step(self, output: Dict[str, np.ndarray]) -> None:
        """Write a step to the ADIOS SST stream and the BP file.

        Parameters
        ----------
        output : Dict[str, np.ndarray]
             key - adios column name to which to write a value of the dictionary
             representing one step

        """
        output["step"] = self.format_scalar(self._step, np.int32)
        self._write_adios_step(self._sst_stream, output)

        # Write gpstime to the BP file
        gpstime = int(datetime.now().timestamp())
        output["gpstime"] = self.format_scalar(gpstime, np.int32)
        self._write_adios_step(self._bp_file, output)
        self._step += 1


# TODO: It may make sense to expose a more generic reporter interface to handle the
#       various data fields that users may want e.g. rmsd, contact map, etc.
class ContactMapReporter:
    """Periodically reports the results of the openmm simulation"""

    def __init__(self, reportInterval, cfg):
        self._reportInterval = reportInterval
        print(cfg)
        print(f"report interval = {reportInterval}")
        print("ContactMapRepoter constructor")
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

        adios_file_config = ADIOSBPConfig(
            name=str(self.cfg.current_dir) + "/trajectory.bp",
            mode="w",
            config_file=str(cfg.adios_xml_file),
            io_in_config_file="Trajectory"
        )
        self.stream = StreamReporter(cfg._adios_stream, adios_file_config)

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)

    def zcentroid(self, positions):
        return np.average(
            positions[self.heavy_atoms_indices, 2], weights=self.heavy_atoms_masses
        )

    def report(self, simulation, state):
        """Computes contact maps, md5 sum of positions, rmsd to the reference state and records them into an ADIOS stream."""
        timer("reporting", 1)
        # TODO: Probably don't need to get the positions again
        stateA = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True).astype(np.float32)
        velocities = stateA.getVelocities(asNumpy=True)
        velocities = np.array(
            [[x[0]._value, x[1]._value, x[2]._value] for x in velocities]
        ).astype(np.float32)

        ca_indices = []
        natoms = 0
        for atom in simulation.topology.atoms():
            natoms += 1
            if atom.name == self.cfg.openmm_selection[0]:
                ca_indices.append(atom.index)

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

        output = {
            "md5": md5,
            "positions": positions,
            "velocities": velocities,
        }

        if self.cfg.model == "cvae":
            output["contact_map"] = contact_map
        elif self.cfg.model == "aae":
            output["point_cloud"] = point_cloud

        if self.cfg.compute_zcentroid:
            zcentroid = self.zcentroid(positions)
            output["zcentroid"] = self.stream.format_scalar(zcentroid, np.float32)

        if self.cfg.compute_rmsd:
            reference_positions = self.rmsd_positions[:d].copy()
            rmsd = rms.rmsd(positions_ca, reference_positions, superposition=True)
            output["rmsd"] = self.stream.format_scalar(rmsd, np.float32)

        if (
            hasattr(self.cfg, "multi_ligand_table")
            and self.cfg.multi_ligand_table.is_file()
        ):
            output["ligand"] = self.stream.format_scalar(self.cfg.ligand, np.int32)
            output["natoms"] = self.stream.format_scalar(natoms, np.int32)

        self.stream.write_adios_step(output)
        timer("reporting", -1)

