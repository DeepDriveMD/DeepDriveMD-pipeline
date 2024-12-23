import os
# Written by Nothando Khumalo, August 16, 2024

""" This program takes the trainig data generated from nwchem and formats it as input data for
    n2p2 calculations. """

def create_file(filename):
    """
    Creates a new file with the given filename.
    """
    with open(filename, 'w') as file:
        file.write("")  # Create an empty file

def write_to_file(filename, molecule_name, coord_file, type_map_file, type_file, force_file, energy_file, mol_identifier):
    """
    Writes the necessary data to the file according to the provided algorithm.
    """
    with open(filename, 'a') as file:

        # Read the data from input files
        geometries = read_geometries(coord_file)
        coords = geometries[0]
        num_geom = len(geometries)
        num_atoms = len(coords)
        elements = read_elements(type_map_file, type_file, num_atoms)
        elements_str = read_unique_elements(type_map_file)
        forces = read_forces(force_file)
        energies = read_energy(energy_file)

        for j in range(num_geom):
            # Write the header
            file.write("begin\n")
            file.write(f"comment {molecule_name} ({elements_str})\n")
            coords = geometries[j]
            fcoords = forces[j]
            # Write the data to the file
            for i in range(num_atoms):
                file.write("atom ")
                x1, y1, z1 = coords[i]
                e1 = elements[i]
                fx1, fy1, fz1 = fcoords[i]
                c1, n1 = 0.0, 0.0  # These values are not used

                # Write the atom line
                file.write(f"{x1} {y1} {z1} {e1} {c1} {n1} {fx1} {fy1} {fz1}\n")

            # Write the energy value from the beginning of the list
            file.write(f"energy {energies[j]}\n")  # Use the first energy value

            # Write footer
            file.write("charge 0.0\n")
            file.write("end\n")
        print("wrote to file")

def read_geometries(coord_file):
    """
    Reads the geometries from coord.raw and the coordinates of each geometry.
    The coordinates of each geometry are converted into a list of tuples,
    and the geometries are returned as a list of coordinates.
    In the coordinates each tuple corresponds to the (x, y, z) coordinates of an atom.
    """
    geometries = []
    with open(coord_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = []
            values = list(map(float, line.split()))
            for i in range(0, len(values), 3):
                x, y, z = values[i], values[i+1], values[i+2]
                coords.append((x, y, z))
            geometries.append(coords)
    print("coords taken")
    return geometries

def read_elements(type_map_file, type_file, num_atoms):
    """
    Reads the element symbols from the type_map.raw file. The element indeces come
    from type.raw. Then look up the elements names and compile a list of chemical
    symbols. The function returns a list of elements, where each element corresponds to an atom.
    """
    elements = []
    with open(type_map_file, 'r') as file:
        # Read all element symbols from the file (assuming they are space-separated on a single line)
        element_symbols = file.read().split()
        for ii in range(len(element_symbols)):
            element_symbols[ii] = element_symbols[ii].capitalize()
    with open(type_file, 'r') as file:
        # Read all element indices from the file
        element_indeces = file.read().split()
    if len(element_indeces) != num_atoms:
        raise RuntimeError(f"Inconsistent number of atoms!\nnum_atoms={str(num_atoms)}\nelement_indeces={str(element_indeces)}")
    
    # Repeat or slice the element symbols to match the number of atoms
    for i in range(num_atoms):
        elements.append(element_symbols[int(element_indeces[i])])
    print("element read")
    #print(elements)
    return elements

def read_unique_elements(type_map_file):
    """
    Reads the element symbols from the type_map.raw file and return them as a string. 
    """
    with open(type_map_file, 'r') as file:
        # Read all element symbols from the file (assuming they are space-separated on a single line)
        element_symbols = file.read().split()
    element_str = ""
    for element in element_symbols:
        if len(element_str) > 0:
            element_str += " "
        element_str += element.capitalize()
    print("unique element read")
    #print(element_symbols)
    return element_str

def read_forces(force_file):
    """
    Reads the force values from the force.raw file and returns them as a list of tuples.
    Each tuple corresponds to the (fx, fy, fz) forces acting on an atom.
    """
    forces = []
    with open(force_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            fcoords = []
            values = list(map(float, line.split()))
            for i in range(0, len(values), 3):
                fx, fy, fz = values[i], values[i+1], values[i+2]
                fcoords.append((fx, fy, fz))
            forces.append(fcoords)
    print("forces acquired")
    return forces

def read_energy(energy_file):
    """
    Reads the energy values from the energy.raw file and returns them as a list of floats.
    """
    energies = []
    with open(energy_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            energy = float(line.strip())
            energies.append(energy)
    print("energy")
    return energies

def find_molecule_folders(directory='.'):
    """
    Finds folders with names starting with 'training_mol_' in the specified directory.
    Returns a list of tuples (folder_path, molecule_identifier).
    """
    folders = []
    for entry in os.listdir(directory):
        if entry.startswith('training_mol_') and os.path.isdir(os.path.join(directory, entry)):
            mol_identifier = entry[len('training_mol_'):]  # Extract the part after 'training_mol_'
            folder_path = os.path.join(directory, entry)
            folders.append((folder_path, mol_identifier))
        if entry.startswith('validate_mol_') and os.path.isdir(os.path.join(directory, entry)):
            mol_identifier = entry[len('validate_mol_'):]  # Extract the part after 'training_mol_'
            folder_path = os.path.join(directory, entry)
            folders.append((folder_path, mol_identifier))
    print(folders)
    print("training_mol folders found")
    return folders

def generate_n2p2_test_files_for_all_folders():
    """
    Finds all relevant folders and generates n2p2 test files for each.
    """
    print("going through files")
    folders = find_molecule_folders()
    folder_path, mol_identifier = folders[0]
    output_filename = os.path.join(folder_path, "..", "input.data")
    create_file(output_filename)
    for folder_path, mol_identifier in folders:
        molecule_name = mol_identifier
        # Turns out N2P2 reads only 1 file for the whole training set
        #output_filename = os.path.join(folder_path, f"{molecule_name}_input.data")
        coord_file = os.path.join(folder_path, "coord.raw")
        type_map_file = os.path.join(folder_path, "type_map.raw")
        type_file = os.path.join(folder_path, "type.raw")
        force_file = os.path.join(folder_path, "force.raw")
        energy_file = os.path.join(folder_path, "energy.raw")

        generate_n2p2_test_file(output_filename, molecule_name, coord_file, type_map_file, type_file, force_file, energy_file, mol_identifier)

def generate_n2p2_test_file(output_filename, molecule_name, coord_file, type_map_file, type_file, force_file, energy_file, mol_identifier):
    """
    Generates the n2p2 test file by calling the necessary functions.
    """
    write_to_file(output_filename, molecule_name, coord_file, type_map_file, type_file, force_file, energy_file, mol_identifier)

# Run the script for all folders
if __name__ == "__main__":
    sample = generate_n2p2_test_files_for_all_folders()
