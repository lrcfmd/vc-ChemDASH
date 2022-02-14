"""
|=============================================================================|
|                           N E I G H B O U R H O O D                         |
|=============================================================================|
|                                                                             |
| This module contains a routine that implements the local combinatorial      |
| neighbourhood algorithm. This algorithm checks through each atom to see if  |
| it can be moved into a lower energy position.                               |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     local_combinatorial_neighbourhood                                       |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 31/01/2020                                                       |
|=============================================================================|
"""

from builtins import range

try:
    import chemdash.gulp_calc
except ModuleNotFoundError:
    import gulp_calc

import ase
import ase.io
import os
import time


# =============================================================================
# =============================================================================
def local_combinatorial_neighbourhood(
    structure, atom_distance_limit, num_points, gulp_lib
):
    """
    Move all atoms in the structure along x,y, and z axes in order to see if
    any can be moved into lower energy positions.

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure object containing the ase atoms object for the initial structure.
    atom_distance_limit : float
        The minimum allowable distance between atoms. This avoids atoms
        being unphysically placed on top on one another.
    num_points : int
        The number of points each atom is placed at along the a, b, and c axes.
    gulp_lib : str
        The filename containing the gulp library.

    Returns
    -------
    updated_atoms : ase atoms
        The atoms object after moving atoms into lower energy positions.

    ---------------------------------------------------------------------------
    Paul Sharp 09/12/2019
    """
    # false needed for while loop
    complete = False
    # gets cell parameters
    cell_params = structure.atoms.get_cell_lengths_and_angles()
    # local combinatorial neighbourhood search time started
    lcn_start_time = time.time()

    # set the energy calculator to gulp
    gulp_command = os.environ["ASE_GULP_COMMAND"]
    # set environment variable for ase to interface running with gulp
    os.environ["ASE_GULP_COMMAND"] = "gulp-4.4 < PREFIX.gin > PREFIX.got"

    # runs initial gulp calculation to get intital structure
    structure, _, _, _ = gulp_calc.multi_stage_gulp_calc(
        structure, 1, ["lcn"], "sing", [""], [""], [[""]], [""], [], gulp_lib
    )
    # saves the structure energy to the current energy
    current_energy = structure.energy

    # I assume this means the number of energy calculations to do in gulp?
    sp_calcs = 1
    # no idea what the atom loops are for
    atom_loops = 0

    # while the loop in incomplete (as defined to False earlier)
    while not complete:

        # We will finish unless we find a better position for one of the atoms.
        # this seems an odd way to implement this
        complete = True

        # atom loops now equal to 1
        atom_loops += 1

        # for each atom in atoms collection
        for iatom in range(0, len(structure.atoms)):

            # Loop over a, b, c axes
            for i in range(0, 3):

                # gets axis length (indexing based on whether a b or c axis)
                axis_length = cell_params[i]

                # To construct grid points
                for j in range(1, num_points):

                    # Set the new grid point for the atom
                    new_coord = float(j) * axis_length / float(num_points)

                    # set old coordinates to old_coord
                    old_coord = structure.atoms[iatom].position[i]

                    # set new coordinates
                    structure.atoms[iatom].position[i] = new_coord

                    # remove the current atom from other atoms
                    other_atoms = list(range(0, len(structure.atoms)))
                    other_atoms.remove(iatom)

                    # If the new atom is not too close to another atom, see if the new position lowers the energy
                    if (
                        min(structure.atoms.get_distances(iatom, other_atoms, mic=True))
                        > atom_distance_limit
                    ):

                        # runs gulp with current structure
                        structure, _, _, _ = gulp_calc.multi_stage_gulp_calc(
                            structure,
                            1,
                            ["lcn"],
                            "sing",
                            [""],
                            [""],
                            [[""]],
                            [""],
                            [],
                            gulp_lib,
                        )

                        # sp calcs must be a counter for number of calculations attempted
                        sp_calcs += 1

                        # Set new position if lower energy, and go through the full set of atoms once more
                        if structure.energy < current_energy:

                            current_energy = structure.energy
                            complete = False

                        else:

                            structure.atoms[iatom].position[i] = old_coord

                    else:

                        structure.atoms[iatom].position[i] = old_coord

    os.environ["ASE_GULP_COMMAND"] = gulp_command

    calc_time = time.time() - lcn_start_time

    return structure, current_energy, calc_time, atom_loops, sp_calcs


if __name__ == "__main__":
    os.environ["GULP_LIB"] = ""
    os.environ["ASE_GULP_COMMAND"] = "gulp-4.4 < PREFIX.gin > PREFIX.got"
    # os.environ["ASE_GULP_COMMAND"] = "timeout " + str(time_limit) + " " + executable + " < PREFIX.gin > PREFIX.got"

    structure = ase.io.read("structure_15.cif")
    new_str = local_combinatorial_neighbourhood(structure, 1.0)

    # structure = ase.io.read('current.cif')
    # new_str = local_combinatorial_neighbourhood(structure, 1.0)

    new_str.write("new_str.cif", format="cif")
