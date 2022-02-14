"""
|=============================================================================|
|                               C h e m D A S H                               |
|=============================================================================|
|                                                                             |
| This module contains the main code for performing structure prediction with |
| ChemDASH.                                                                   |
|                                                                             |
| This code deals with initialisation of the structures, swapping atoms in    |
| the structures, and evaluating the relaxed structures with the basin        |
| hopping method.                                                             |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     ChemDASH                                                                |
|     Structure                                                               |
|     Counts                                                                  |
|     write_output_file_header                                                |
|     optimise_structure                                                      |
|     update_potentials                                                       |
|     generate_new_structure                                                  |
|     strip_vacancies                                                         |
|     output_list                                                             |
|     report_rejected_structure                                               |
|     report_statistics                                                       |
|     read_restart_file                                                       |
|     write_restart_file                                                      |
|     search_local_neighbourhood                                              |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 25/03/2020                                                       |
|=============================================================================|
"""
import math
from builtins import range

import collections
import copy
import os
import sys
import time

import ase.io
from ase import Atom

import numpy as np

try:
    import chemdash.bonding as bonding
except ModuleNotFoundError:
    import bonding
    import dope
    import gulp_calc
    import initialise
    import inputs
    import neighbourhood
    import rngs
    import swap
    import symmetry
    import vasp_calc
    from magnetic_write_cif import write_custom_cif
else:
    import chemdash.dope as dope
    import chemdash.gulp_calc as gulp_calc
    import chemdash.initialise as initialise
    import chemdash.inputs as inputs
    import chemdash.neighbourhood as neighbourhood
    import chemdash.rngs as rngs
    import chemdash.swap as swap
    import chemdash.symmetry as symmetry
    import chemdash.vasp_calc as vasp_calc
    from chemdash.magnetic_write_cif import write_custom_cif


# TODO: multiply_out_moments doesnt work for floats
# =============================================================================
# =============================================================================


class ChemDASH(object):
    """
    ChemDASH predicts the structure of new materials by placing atomic species
    at random positions on a grid and exploring the potential energy surface
    using a basin hopping approach.

    The structure is manipulated by swapping the positions of atoms.
    The swaps involve either: cations, anions, cations and anions, or cations,
    anions and vacancies, along with any custom group of atoms. We can swap any
    number of atoms per structure manipulation.

    The code makes extensive use of the Atomic Simulation Environment (ase),
    allowing us to perform structure relaxation using GULP and VASP.

    Parameters
    ----------
    calc_name : string
        The seed name for this calculation, which forms the name of the
        .input and .atoms files.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 25/03/2020



    Magnetism implementation, doping methods and class packaging
    ---------------------------------------------------------------------------
    Robert Dickson 29/07/2021
    """

    def __init__(self, calc_name):

        # calculation time and names
        self.num_structures_considered = 0
        self.start_time = time.time()
        self.calc_name = calc_name
        self.input_file = calc_name + ".input"
        self.open_files = []

        # initialise input parameters
        self.default_params = inputs.initialise_default_params(self.calc_name)

        # check for errors in input parameters
        self.error_log = self.calc_name + ".error"
        self.params, self.errors = self.check_input()

        # output files
        self.output = open(self.params["output_file"]["value"], mode="a")
        self.energy_step_file = open(self.params["energy_step_file"]["value"], mode="a")
        self.energy_file = open(self.params["energy_file"]["value"], mode="a")
        self.bvs_file = open(self.params["bvs_file"]["value"], mode="a")
        self.potentials_file = open(self.params["potentials_file"]["value"], mode="a")
        self.derivs_file = open(self.params["potential_derivs_file"]["value"], mode="a")
        self.open_files = [
            self.output,
            self.energy_step_file,
            self.energy_file,
            self.bvs_file,
            self.potentials_file,
            self.derivs_file,
        ]

        # initialise inputs
        self.additional_inputs, self.rng = self.initialise_calculation()
        self.calc_bvs = False
        self.calc_pot = False

        # initialise atoms
        self.initial_atoms = self.initialise_atoms()

        # initialise structure and data
        self.best_structure = Structure()
        self.new_structure = Structure()
        self.current_structure = Structure(
            self.initial_atoms,
            0,
            dopant_atoms=self.params["bvs_dopant_atoms"]["value"],
            dopant_atom_charges=self.params["bvs_dopant_atoms_charges"]["value"],
        )

        self.atomic_numbers_list = []
        self.positions_list = []

        self.basins = {}
        self.outcomes = {}

        self.structure_count = Counts()
        self.structure_index = 0

        self.moments = None
        self.current_moments = None
        self.total_structures = self.params["max_structures"]["value"]

        self.initialise_structure()

        # initialise doping parameters
        self.doping_pool = self.params["random_dopant_atoms"]["value"]

        # Set up ASE trajectory files - we track all structures, as well as only accepted structures (relaxed/unrelaxed)
        if self.params["output_trajectory"]["value"]:
            self.all_traj = ase.io.trajectory.Trajectory(filename="all.traj", mode="w")
            self.all_relaxed_traj = ase.io.Trajectory(
                filename="all_relaxed.traj", mode="w"
            )
            self.accepted_traj = ase.io.Trajectory(filename="accepted.traj", mode="w")
            self.accepted_relaxed_traj = ase.io.Trajectory(
                filename="accepted_relaxed.traj", mode="w"
            )

        # initialise swap groups
        self.no_valid_swap_groups = False
        self.swap_groups = self.initialise_swap_groups()

        # initialise swapping method
        self.swapping_method = self.check_bvs_or_site_potential()

        # set counter for nth step
        if self.params["nth_doping_step"]["value"] is not None:
            self.nth_step = 0
        else:
            self.nth_step = None

    def check_input(self):
        """

        Checks through all input parameters in the .input file to

        Returns
        -------

        """
        # Set parameters from input file
        params, errors = inputs.parse_input(self.input_file, self.calc_name)

        # Convert, and ensure all input file parameters are of the correct type
        params, errors = inputs.convert_values(params, errors)
        params, errors = inputs.handle_dependencies(params, errors)

        # if any input file errors are found, the program terminates and writes to error log
        if len(errors) > 0:
            inputs.report_input_file_errors(errors, self.input_file, self.error_log)
            sys.exit(
                "Terminating execution - see error file {0}".format(self.error_log)
            )

        return params, errors

    def initialise_calculation(self):
        # output by default set to calc_name + .chemdash

        write_output_file_header(self.output, self.params)

        # Override restart option if necessary
        if self.params["restart"]["value"] and (
            not os.path.isfile(self.params["restart_file"]["value"])
        ):
            self.output.write(
                f'OVERRIDING OPTION -- "restart" is specified as True, but the restart file '
                f'"{self.params["restart_file"]["value"]}" file does not exist. Proceeding with restart = False.'
                f"\n\n"
            )

            self.params["restart"]["value"] = False

        # Set GULP command or VASP script
        if self.params["calculator"]["value"] == "gulp":
            gulp_calc.set_gulp_command(
                self.params["gulp_executable"]["value"],
                self.params["calculator_cores"]["value"],
                self.params["calculator_time_limit"]["value"],
                "",
            )

        # allows for setting of vasp executable, number of cores and pseudopotential directory directly from file
        elif self.params["calculator"]["value"] == "vasp":
            vasp_script = "run_vasp.py"
            vasp_calc.set_vasp_script(
                vasp_script,
                self.params["vasp_executable"]["value"],
                self.params["calculator_cores"]["value"],
                self.params["vasp_pp_dir"]["value"],
                method=self.params["run_method"]["value"],
            )

        else:
            sys.exit(
                'ERROR - The calculator "{0}" is not currently supported in ChemDASH.'.format(
                    self.params["calculator"]["value"]
                )
            )

        # Set keywords and options for each stage of GULP/VASP calculations
        additional_inputs = {
            "gulp_keywords": [],
            "gulp_options": [],
            "gulp_max_gnorms": [],
            "vasp_settings": [],
        }

        for i in range(1, self.params["num_calc_stages"]["value"] + 1):
            # GULP inputs
            additional_inputs["gulp_keywords"].append(
                self.params["gulp_calc_" + str(i) + "_keywords"]["value"]
            )
            additional_inputs["gulp_options"].append(
                self.params["gulp_calc_" + str(i) + "_options"]["value"]
            )
            additional_inputs["gulp_max_gnorms"].append(
                self.params["gulp_calc_" + str(i) + "_max_gnorm"]["value"]
            )

            # VASP inputs
            additional_inputs["vasp_settings"].append(
                self.params["vasp_calc_" + str(i) + "_settings"]["value"]
            )

        # Initialise and seed random number generator
        if not self.params["random_seed"]["specified"]:
            self.params["random_seed"]["value"] = rngs.generate_random_seed(
                self.params["seed_bits"]["value"]
            )

        self.output.write(
            "The seed for the random number generator is {0:d}\n".format(
                self.params["random_seed"]["value"]
            )
        )

        rng = rngs.NR_Ran(self.params["random_seed"]["value"])
        rng.warm_up(self.params["rng_warm_up"]["value"])

        return additional_inputs, rng

    def initialise_atoms(self):

        # TODO: a function that directly reads a cif file would be useful but would still need to specify
        #  oxidation states. If a spinel is specified with Zn8Fe16O32 the ChemDASH routine (or something in it)
        #  changes the order of atoms to Zn2Fe4O8Zn2Fe4O8Zn2Fe4O8Zn2Fe4O8 which breaks the magnetic ordering

        # check that a ".atoms" file exists, stop if not
        if not os.path.isfile(self.params["atoms_file"]["value"]):
            sys.exit(
                'ERROR - the ".atoms" file "{0}" does not exist.'.format(
                    self.params["atoms_file"]["value"]
                )
            )

        atoms_data = initialise.read_atoms_file(self.params["atoms_file"]["value"])

        # check that the structure is charge balanced, stop if not
        charge_balance = initialise.check_charge_balance(atoms_data)

        if charge_balance != 0:
            sys.exit(
                "ERROR - the structure is not charge balanced. The overall charge is: {0}.".format(
                    charge_balance
                )
            )

        # check if initial structure is to be taken from file
        if self.params["initial_structure_file"]["specified"]:

            self.output.write(
                'The initial structure will be read from the file: "{0}".\n'.format(
                    self.params["initial_structure_file"]["value"]
                )
            )

            initial_atoms = initialise.initialise_from_cif(
                self.params["initial_structure_file"]["value"],
                atoms_data,
                mag_moms=self.params["initial_mag_moments"]["value"],
            )

        # or initialise on grid
        else:

            self.output.write(
                "The atoms will be initialised on a {0} grid.\n".format(
                    (self.params["grid_type"]["value"].replace("_", " "))
                )
            )

            if self.params["grid_type"]["value"] == "close_packed":

                # Generate a sequence of close packed anion layers if one has not been specified
                if not self.params["cp_stacking_sequence"]["specified"]:
                    self.params["cp_stacking_sequence"][
                        "value"
                    ] = initialise.generate_random_stacking_sequence(
                        self.params["grid_points"]["value"][2], self.rng
                    )

                # This allows us to define the anion sub-lattice of the close-packed
                # grid in the input file, whilst fitting in with existing routines.
                self.params["cell_spacing"]["value"][2] *= 0.5
                self.output.write(
                    "The close packed stacking sequence is: {0}.\n".format(
                        "".join(self.params["cp_stacking_sequence"]["value"])
                    )
                )

            elif any(
                grid_type in self.params["grid_type"]["value"]
                for grid_type in ["orthorhombic", "rocksalt"]
            ):

                # Adjust the cell spacing to define according to full grid rather than anion grid
                self.params["cell_spacing"]["value"][:] = [
                    0.5 * x for x in self.params["cell_spacing"]["value"]
                ]

            # Set up atoms object and grid, and list of unoccupied points (all possible points)
            initial_cell, anion_grid, cation_grid = initialise.set_up_grids(
                self.params["grid_type"]["value"],
                self.params["grid_points"]["value"],
                self.params["cp_stacking_sequence"]["value"],
                self.params["cp_2d_lattice"]["value"],
            )
            scaled_cell = initialise.scale_cell(
                initial_cell, self.params["cell_spacing"]["value"]
            )
            (
                initial_atoms,
                anion_grid,
                cation_grid,
            ) = initialise.populate_grids_with_atoms(
                scaled_cell, anion_grid, cation_grid, atoms_data, self.rng
            )

            # Use unused points as vacancies if we are not using vacancy grids
            if not self.params["vacancy_grid"]["value"]:
                initial_atoms = initialise.populate_points_with_vacancies(
                    initial_atoms.copy(), anion_grid + cation_grid
                )

        initial_atoms = ase.Atoms(
            sorted(initial_atoms, key=lambda x: x.symbol),
            cell=initial_atoms.cell,
            pbc=initial_atoms.pbc,
        )
        return initial_atoms

    def initialise_structure(self):
        # Set up "Structure" object for the structure currently under consideration
        if self.params["restart"]["value"]:
            (
                self.best_structure,
                self.current_structure,
                self.atomic_numbers_list,
                self.positions_list,
                self.basins,
                self.outcomes,
                self.structure_count,
                self.structure_index,
                self.moments,
            ) = read_restart_file(self.params["restart_file"]["value"])

            # added to restart the magnetic structure from the previous run
            self.current_moments = copy.deepcopy(self.moments)

            if not self.params["vacancy_grid"]["value"]:
                self.current_structure.atoms = strip_vacancies(
                    self.current_structure.atoms
                )

        else:
            self.moments = copy.deepcopy(self.params["initial_mag_moments"]["value"])

        atom_set = strip_vacancies(
            self.current_structure.atoms.copy()
        ).get_chemical_formula()
        self.output.write(f"The set of atoms used in this simulation is {atom_set}.\n")
        self.output.write("\n")
        self.output.write("The cell parameters are:\n")
        self.output.write("\n")
        self.output.write(
            " a = \t{c[0]:.8f}\talpha = \t{c[3]:.8f}\n b = \t{c[1]:.8f}\tbeta  = \t{c[4]:.8f}\n c = \t{c["
            "2]:.8f}\tgamma = \t{c[5]:.8f}\n".format(
                c=self.current_structure.atoms.get_cell_lengths_and_angles()
            )
        )
        self.output.write("\n")

        # custom function written to allow for a magnetic cif file to be written
        write_custom_cif("current.cif", images=self.current_structure.atoms)

    def initialise_swap_groups(self):

        # Set default swap groups and weightings if none were given in the input
        if not self.params["swap_groups"]["specified"]:
            self.params["swap_groups"]["value"] = swap.initialise_default_swap_groups(
                self.current_structure, self.params["swap_groups"]["value"]
            )

        swap_lengths = [len(group) for group in self.params["swap_groups"]["value"]]
        if all(length == 1 for length in swap_lengths):
            self.params["swap_groups"][
                "value"
            ] = swap.initialise_default_swap_weightings(
                self.params["swap_groups"]["value"]
            )

        # Check swap groups against the initial structure
        unique_elements = list(
            set(
                self.current_structure.atoms.get_chemical_symbols()
                + [x.symbol for x in self.current_structure.dopant_atoms]
            )
        )
        test_atoms = self.current_structure

        # Include vacancies in check if we are using vacancy grids
        if self.params["vacancy_grid"]["value"]:
            unique_elements += "X"
            test_atoms.atoms.extend(
                ase.Atoms(
                    "X",
                    cell=test_atoms.atoms.get_cell(),
                    charges=[0],
                    pbc=[True, True, True],
                )
            )

        # checks custom swap groups for valid input
        custom_swap_group_errors = swap.check_elements_in_custom_swap_groups(
            self.params["swap_groups"]["value"], unique_elements
        )

        # check validity of swap groups
        valid_swap_groups, verifying_swap_group_errors = swap.verify_swap_groups(
            test_atoms, self.params["swap_groups"]["value"]
        )

        swap_group_errors = custom_swap_group_errors + verifying_swap_group_errors

        if len(swap_group_errors) > 0:
            inputs.report_input_file_errors(
                swap_group_errors, self.input_file, self.error_log
            )
            sys.exit(
                "Terminating execution - see error file {0}".format(self.error_log)
            )

        swap_group_names = [group[0] for group in self.params["swap_groups"]["value"]]

        swap_groups_string = ", ".join([str(x) for x in swap_group_names])
        self.output.write(
            f"We will consider the swap groups: {swap_groups_string}.\n\n"
        )
        self.output.write("-" * 80)
        self.output.write("\n\n")

        return swap_group_names

    def check_bvs_or_site_potential(self):
        # =====================================================================
        # Determine whether or not the Bond Valence Sum and site potential need
        # to be, and can be, calculated.

        if (
            self.params["atom_rankings"]["value"] == "bvs"
            or self.params["atom_rankings"]["value"] == "bvs+"
        ):

            self.calc_bvs = True
            missing_bonds = bonding.check_R0_values(self.current_structure.atoms.copy())

            if len(missing_bonds) > 0:
                self.output.write(
                    "The Bond Valence Sum cannot be calculated for this structure because we do not have R0 values "
                    "for the following bonds: {0}.\n".format("".join(missing_bonds))
                )

                self.calc_bvs = False

                self.output.write(
                    "OVERRIDING OPTION -- The atom rankings were specified to be determined by the Bond Valence Sum, "
                    "but the Bond Valence Sum cannot be correctly calculated. Proceeding with atom rankings "
                    "determined at random."
                )

                self.params["atom_rankings"]["value"] = "random"
                return "random"

        if (
            self.params["atom_rankings"]["value"] == "site_pot"
            or self.params["atom_rankings"]["value"] == "bvs+"
        ):
            self.calc_pot = True

        return self.params["atom_rankings"]["value"]

    def optimise_first_structure(self):
        # =====================================================================
        # Optimise first structure

        self.output.write("Initial structure\n")

        (
            visited,
            self.positions_list,
            self.atomic_numbers_list,
        ) = swap.check_previous_structures(
            self.current_structure.atoms, self.positions_list, self.atomic_numbers_list
        )

        if self.params["search_local_neighbourhood"]["value"]:
            self.current_structure = search_local_neighbourhood(
                self.current_structure, self.output, self.params
            )

        if not self.params["testing"]["value"]:
            # Relax initial structure
            # current_structure.atoms = (current_structure.atoms.copy()).rattle(params["rattle_stdev"]["value"])

            self.current_structure, result, outcomes, relax_time = optimise_structure(
                self.current_structure,
                self.params,
                self.additional_inputs,
                0,
                self.outcomes,
            )
            self.output.write(
                f"This calculation is: {result}, {self.params['calculator']['value'].upper()} "
                f"calculation time: {relax_time:f}s\n"
            )
        else:
            result = "converged"

        # Abort if first structure is not converged, or if GULP/VASP failed or has timed out.
        if (
            self.params["converge_first_structure"]["value"]
            and result != "converged"
            and not self.params["testing"]["value"]
        ):

            if result == self.params["calculator"]["value"] + " failure":
                self.output.write(
                    "ERROR -- {0} has failed to perform an optimisation of the initial structure, aborting "
                    "calculation.\n".format(self.params["calculator"]["value"].upper())
                )
            if result == "timed out":
                self.output.write(
                    "ERROR -- Optimisation of initial structure has timed out, aborting calculation.\n"
                )
            if result == "unconverged":
                self.output.write(
                    "ERROR -- Optimisation of initial structure not achieved, aborting calculation.\n"
                )

            self.output.write(
                "Time taken: {0:f}s\n".format(time.time() - self.start_time)
            )
            sys.exit("Terminating Execution")

        # if magnetic moments, write to output
        if (
            self.params["initial_mag_moments"]["specified"]
            and not self.params["testing"]["value"]
        ):
            write_moments = self.current_structure.atoms.get_magnetic_moments()
        else:
            write_moments = self.params["initial_mag_moments"]["value"]

        # Find symmetries in final structure
        try:
            self.current_structure.atoms = symmetry.symmetrise_atoms(
                self.current_structure.atoms
            )
        except TypeError:
            self.output.write("Error symmetrising atoms...")

        if self.params["output_trajectory"]["value"]:
            self.all_traj.write(atoms=strip_vacancies(self.initial_atoms.copy()))
            self.all_relaxed_traj.write(
                atoms=strip_vacancies(self.current_structure.atoms.copy())
            )

        # accept or reject structure
        self.current_structure, basins = accept_structure(
            self.current_structure,
            self.params,
            self.energy_step_file,
            result,
            self.basins,
            self.calc_bvs,
            self.calc_pot,
            self.energy_file,
            self.bvs_file,
            self.potentials_file,
            self.derivs_file,
            self.initial_atoms,
        )

        # Set the current structure as the best structure found so far
        self.best_structure = Structure(
            self.current_structure.atoms.copy(),
            0,
            self.current_structure.energy,
            self.current_structure.volume,
            self.current_structure.ranked_atoms,
            self.current_structure.bvs_atoms,
            self.current_structure.bvs_sites,
            self.current_structure.potentials,
            self.current_structure.derivs,
            self.current_structure.dopant_atoms,
            self.current_structure.dopant_atom_charges,
        )

        # Accept structure if it converges, set dummy values if not
        if result == "converged":

            self.structure_count.zero_conv = 1
            self.structure_count.converged += 1

            write_custom_cif(
                "structure_0.cif",
                images=strip_vacancies(self.current_structure.atoms.copy()),
                moments=write_moments,
            )

            self.output.write(
                "The energy of structure {0:d} is {1:.8f} eV/atom\n".format(
                    self.current_structure.index, self.current_structure.energy
                )
            )

            if self.params["output_trajectory"]["value"]:
                self.accepted_traj.write(
                    atoms=strip_vacancies(self.initial_atoms.copy())
                )
                self.accepted_relaxed_traj.write(
                    atoms=strip_vacancies(self.current_structure.atoms.copy())
                )

        else:

            self.structure_count = report_rejected_structure(
                self.output,
                result,
                self.params["calculator"]["value"],
                self.structure_count,
            )

        # Output current and best structure
        write_custom_cif(
            "current.cif", images=self.current_structure.atoms, moments=write_moments
        )
        write_custom_cif(
            "best.cif", images=self.best_structure.atoms, moments=write_moments
        )

        self.output.write("\n")
        self.structure_index = 1

        return self.current_structure

    def optimise_new_structure(self, i, change):

        # =============================================================
        # Optimise the structure
        ##################

        if self.params["search_local_neighbourhood"]["value"]:
            self.new_structure = search_local_neighbourhood(
                self.new_structure, self.output, self.params
            )

        ##################
        if not self.params["testing"]["value"]:
            self.new_structure, result, outcomes, relax_time = optimise_structure(
                self.new_structure,
                self.params,
                self.additional_inputs,
                i,
                self.outcomes,
            )

            self.output.write(
                "This calculation is: {0}, {1} calculation time: {2:f}s\n".format(
                    result, self.params["calculator"]["value"].upper(), relax_time
                )
            )
        else:
            result = "converged"

        if result != "converged":
            self.structure_count = report_rejected_structure(
                self.output,
                result,
                self.params["calculator"]["value"],
                self.structure_count,
            )
            os.rename("CONTCAR", f"rejected_structure_{i}.vasp")
            return None, None, None

        self.structure_count.converged += 1

        # testing does not run the optimisation but instead accepts ALL structures
        if not self.params["testing"]["value"]:

            # get magnetic moments for writing
            try:
                write_moments = self.new_structure.atoms.get_magnetic_moments()
                self.new_structure.final_magnetic_moments = (
                    self.new_structure.atoms.get_magnetic_moments()
                )
            except AttributeError:
                print("Cannot get current_magnetic_moments")
                write_moments = copy.deepcopy(self.current_moments)
                self.new_structure.final_magnetic_moments = copy.deepcopy(
                    self.current_moments
                )

            moments_string = ""
            for moment in write_moments:
                moments_string += f"{round(moment, 3)}, "
            moments_string.rstrip(",")

            self.output.write(f"The final magnetic moments are {moments_string}\n")

            # if self.params["convex_hull_file"]["value"]:
            #     energy_diff = "convex hull energy should be obtained here!!"

            if change == "swap":
                if self.params["bvs_dopant_atoms"]["value"]:
                    end_members = self.params["solid_solution_end_members"]["value"]
                    energy_diff = dope.get_solid_solution_energy_difference(
                        self.new_structure, end_members
                    )
                else:
                    energy_diff = (
                        self.new_structure.energy - self.current_structure.energy
                    )
            elif change == "dope":
                end_members = self.params["solid_solution_end_members"]["value"]
                energy_diff = dope.get_solid_solution_energy_difference(
                    self.new_structure, end_members
                )
            else:
                raise ValueError("change must be either dope or swap")

            self.output.write(
                "The energy of structure {0:d} is {1:.8f} eV/atom. "
                "The difference in energy is {2:f} eV/atom.\n".format(
                    i, self.new_structure.energy, energy_diff
                )
            )
            self.output.write("\n")

        else:
            energy_diff = 0
            write_moments = copy.deepcopy(self.current_moments)
            self.new_structure.final_magnetic_moments = copy.deepcopy(
                self.current_moments
            )

        # There is a bug here in spglib where something has a NoneType
        # The bug has not been diagnosed, hence the try-except statement
        try:
            self.new_structure.atoms = symmetry.symmetrise_atoms(
                self.new_structure.atoms.copy()
            )
        except TypeError:
            pass

        if self.params["output_trajectory"]["value"]:
            self.all_traj.write(atoms=strip_vacancies(self.new_structure.atoms.copy()))
            self.all_relaxed_traj.write(
                atoms=strip_vacancies(self.new_structure.atoms.copy())
            )

        return energy_diff, result, write_moments

    def generate_new_chemdash_structure(self, i):

        # Generate the next structure by swapping/doping atoms and vacancies
        self.new_structure = copy.deepcopy(self.current_structure)

        # add dopant atoms to structure if there are any
        if self.new_structure.dopant_atoms:
            for atom in self.new_structure.dopant_atoms:
                self.new_structure.atoms.append(atom)
            self.new_structure.dopant_atoms = []

        # Check which swap groups remain valid for this structure
        valid_swap_groups, verifying_swap_group_errors = swap.verify_swap_groups(
            self.new_structure, self.params["swap_groups"]["value"]
        )

        # End the simulation if there are no valid swap groups
        if len(valid_swap_groups) == 0:
            self.no_valid_swap_groups = True
            return "break"

        # if constant magnetic structure, get current magnetic moments
        # This was breaking the magnetism as the changing of current and previous magnetic moments
        # is handled by current_moments and moments
        if self.params["keep_mag_structure_constant"]["value"]:
            self.current_moments = copy.deepcopy(self.moments)
            print(f"current_moments = {self.current_moments}")

        (
            self.new_structure,
            change,
            dope_in,
            dope_out,
            self.doping_pool,
            nth_step,
            self.current_moments,
        ) = generate_new_structure(
            self.new_structure,
            self.params,
            self.output,
            valid_swap_groups,
            self.new_structure.ranked_atoms,
            self.rng,
            self.current_moments,
            nth_step=self.nth_step,
            dopant_atoms=self.new_structure.dopant_atoms,
        )

        self.new_structure = Structure(
            self.new_structure.atoms.copy(),
            i,
            self.new_structure.energy,
            self.new_structure.volume,
            self.new_structure.ranked_atoms,
            self.new_structure.bvs_atoms,
            self.new_structure.bvs_sites,
            self.new_structure.potentials,
            self.new_structure.derivs,
            self.new_structure.dopant_atoms,
            self.new_structure.dopant_atom_charges,
        )

        return self.new_structure, change

    def finish_chemdash_run(self):
        # =====================================================================
        # Finish up

        write_restart_file(
            self.best_structure,
            self.current_structure,
            self.atomic_numbers_list,
            self.positions_list,
            self.basins,
            self.outcomes,
            self.structure_count,
            self.total_structures,
            self.moments,
        )

        structures_considered = self.total_structures

        if self.no_valid_swap_groups:
            self.output.write(
                "The supplied swap groups are no longer valid for this structure.\n"
            )
            structures_considered = self.num_structures_considered
        elif self.total_structures == self.params["max_structures"]["value"]:
            self.output.write("Swapping complete.\n")
        else:
            self.output.write("Requested number of structures considered.\n")

        report_statistics(
            self.output,
            self.basins,
            self.outcomes,
            self.structure_count,
            structures_considered,
            self.params["calculator"]["value"],
        )

        self.output.write(
            "The best structure is structure {0:d}, with energy {1:.8f} eV/atom and volume {2:.8f} A0^3, and has been "
            "written to best.cif\n".format(
                self.best_structure.index,
                self.best_structure.energy,
                self.best_structure.volume,
            )
        )

        self.output.write("\n")
        self.output.write("Time taken: {0:f}s\n".format(time.time() - self.start_time))

    def run_chemdash(self):

        # optimise first structure
        if not self.params["restart"]["value"]:
            self.current_structure = self.optimise_first_structure()

        # Basin Hopping loop
        self.output.write("Starting Basin Hopping Loop...\n\n")
        self.output.write("-" * 80)
        self.output.write("\n")

        default_directed_num_atoms = self.params["directed_num_atoms"]["value"]

        if self.params["num_structures"]["specified"]:
            self.total_structures = min(
                self.structure_index + self.params["num_structures"]["value"],
                self.params["max_structures"]["value"],
            )

        # Loop starts here
        for i in range(self.structure_index, self.total_structures):
            self.output.write("\n")
            self.output.write("Structure {0:d}\n".format(i))

            # Write restart file
            write_restart_file(
                self.best_structure,
                self.current_structure,
                self.atomic_numbers_list,
                self.positions_list,
                self.basins,
                self.outcomes,
                self.structure_count,
                i,
                self.moments,
            )

            # Update all output files
            for f in self.open_files:
                f.flush()

            # generate new structure
            self.new_structure, change = self.generate_new_chemdash_structure(i)

            # run optimisation
            energy_diff, result, write_moments = self.optimise_new_structure(i, change)

            if result == "unconverged":
                self.output.write(
                    "This structure has not converged, and will therefore be rejected.\n\n"
                )
                self.output.write("-" * 80)
                self.output.write("\n")

                self.structure_count.repeated += 1
                self.params["pair_weighting"]["value"] /= self.params[
                    "pair_weighting_scale_factor"
                ]["value"]
                self.params["directed_num_atoms"]["value"] += self.params[
                    "directed_num_atoms_increment"
                ]["value"]
                continue

            # Check whether this proposed structure has been previously considered, if so try a new swap
            # for magnetic structures the tolerance needs to be lower as different final magnetic moments
            # artificially render the same structures non-equivalent

            (
                visited,
                positions_list,
                atomic_numbers_list,
            ) = swap.check_previous_structures(
                self.new_structure.atoms,
                self.positions_list,
                self.atomic_numbers_list,
                pos_tol=self.params["position_tolerance"]["value"],
            )

            # Minima hopping - larger moves for repeated basins, smaller moves for new basins
            if visited:
                self.output.write(
                    "This structure has been considered previously, and will therefore be rejected.\n\n"
                )
                self.output.write("-" * 80)
                self.output.write("\n")

                self.structure_count.repeated += 1
                self.params["pair_weighting"]["value"] /= self.params[
                    "pair_weighting_scale_factor"
                ]["value"]
                self.params["directed_num_atoms"]["value"] += self.params[
                    "directed_num_atoms_increment"
                ]["value"]
                continue

            if result == "converged":
                write_custom_cif(
                    f"structure_{i}.cif",
                    images=self.new_structure.atoms,
                    moments=write_moments,
                )

            self.params["pair_weighting"]["value"] *= self.params[
                "pair_weighting_scale_factor"
            ]["value"]

            # =============================================================
            # Determine whether or not the swap should be accepted

            if (
                swap.accept_swap(energy_diff, self.params["temp"]["value"], self.rng)
                and not self.params["testing"]["value"]
            ):
                if change == "swap":

                    self.output.write("The swap is accepted.\n\n")
                    self.output.write("-" * 80)
                    self.output.write("\n")

                elif change == "dope":
                    self.output.write("The doping is accepted.\n")
                    self.params["random_dopant_atoms"]["value"] = self.doping_pool

                self.params["directed_num_atoms"]["value"] = default_directed_num_atoms
                if not self.params["testing"]["value"]:
                    os.rename("POSCAR", f"final_structure_{i}.vasp")
                if self.structure_count.converged > 0:
                    self.energy_step_file.write(
                        "{0:d} {1:.8f} {2:.8f}\n".format(
                            i,
                            self.current_structure.energy,
                            self.current_structure.volume,
                        )
                    )

                self.structure_count.accepted += 1
                self.params["temp"]["value"] /= self.params["temp_scale_factor"][
                    "value"
                ]

                self.moments = copy.deepcopy(self.current_moments)
                print(
                    f"Moments to start new calculation after acceptance are: {self.moments}"
                )

                self.current_structure = copy.deepcopy(
                    Structure(
                        self.new_structure.atoms.copy(),
                        i,
                        self.new_structure.energy,
                        self.new_structure.volume,
                        self.new_structure.ranked_atoms,
                        self.new_structure.bvs_atoms,
                        self.new_structure.bvs_sites,
                        self.new_structure.potentials,
                        self.new_structure.derivs,
                        self.new_structure.dopant_atoms,
                        self.new_structure.dopant_atom_charges,
                    )
                )

                self.current_structure, basins = accept_structure(
                    self.current_structure,
                    self.params,
                    self.energy_step_file,
                    result,
                    self.basins,
                    self.calc_bvs,
                    self.calc_pot,
                    self.energy_file,
                    self.bvs_file,
                    self.potentials_file,
                    self.derivs_file,
                    self.current_structure.atoms,
                )

                write_custom_cif(
                    "current.cif",
                    images=self.current_structure.atoms,
                    moments=write_moments,
                )

                if self.params["output_trajectory"]["value"]:
                    self.accepted_traj.write(
                        atoms=strip_vacancies(self.new_structure.atoms.copy())
                    )
                    self.accepted_relaxed_traj.write(
                        atoms=strip_vacancies(self.new_structure.atoms.copy())
                    )

                # Keep track of best structure
                if self.current_structure.energy < self.best_structure.energy:
                    self.best_structure = Structure(
                        self.new_structure.atoms.copy(),
                        i,
                        self.current_structure.energy,
                        self.current_structure.volume,
                        self.current_structure.ranked_atoms,
                        self.current_structure.bvs_atoms,
                        self.current_structure.bvs_sites,
                        self.current_structure.potentials,
                        self.current_structure.derivs,
                    )

                    write_custom_cif(
                        "best.cif",
                        images=self.best_structure.atoms,
                        moments=write_moments,
                    )

            elif self.params["testing"]["value"]:
                self.output.write(
                    "The swap is accepted (automatically for testing).\n\n"
                )
                self.output.write("-" * 80)
                self.output.write("\n")

                # set current moments to the last used set of moments if structure is accepted
                moments_string = ""
                for moment in write_moments:
                    moments_string += f"{round(moment, 3)}, "
                moments_string.rstrip(",")
                self.moments = copy.deepcopy(self.current_moments)
                self.output.write(
                    f"The new order of magnetic moments for structure {i} are: {moments_string}\n"
                )

                self.params["directed_num_atoms"]["value"] = default_directed_num_atoms

                self.structure_count.accepted += 1

                self.current_structure = Structure(
                    self.new_structure.atoms.copy(),
                    i,
                    self.new_structure.energy,
                    self.new_structure.volume,
                    self.new_structure.ranked_atoms,
                    self.new_structure.bvs_atoms,
                    self.new_structure.bvs_sites,
                    self.new_structure.potentials,
                    self.new_structure.derivs,
                    self.new_structure.dopant_atoms,
                    self.new_structure.dopant_atom_charges,
                )

                write_custom_cif(
                    "current.cif",
                    images=self.current_structure.atoms,
                    moments=write_moments,
                )

                if self.params["output_trajectory"]["value"]:
                    self.accepted_traj.write(
                        atoms=strip_vacancies(self.new_structure.atoms.copy())
                    )
                    self.accepted_relaxed_traj.write(
                        atoms=strip_vacancies(self.new_structure.atoms.copy())
                    )

                    write_custom_cif(
                        "best.cif",
                        images=self.best_structure.atoms,
                        moments=write_moments,
                    )
            else:
                self.output.write("The swap is rejected.\n\n")
                self.output.write("-" * 80)
                self.output.write("\n")
                print(
                    f"Moments to start new calculation after rejection are: {self.moments}"
                )
                os.rename("POSCAR", f"rejected_structure_{i}.vasp")
                self.params["temp"]["value"] *= self.params["temp_scale_factor"][
                    "value"
                ]

            self.num_structures_considered += 1

        self.finish_chemdash_run()

        # When loop is finished, write final part of outputs
        output_list(
            self.potentials_file,
            "{0:d}".format(self.total_structures - 1),
            self.current_structure.potentials,
        )
        output_list(
            self.derivs_file,
            "{0:d}".format(self.total_structures - 1),
            self.current_structure.derivs,
        )
        self.output.write("\n")

        for f in self.open_files:
            # f.flush()
            f.close()


# =============================================================================
# =============================================================================
class Structure(object):
    """
    Stores data referring to particular structures.

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    # =========================================================================
    def __init__(
        self,
        atoms=None,
        index=0,
        energy=0.0,
        volume=0.0,
        ranked_atoms=None,
        bvs_atoms=None,
        bvs_sites=None,
        potentials=None,
        derivs=None,
        dopant_atoms=[],
        dopant_atom_charges=[],
    ):
        """
        Initialise the Structure data.

        Parameters
        ----------
        atoms : ase atoms
            The atoms object containing the structure.
        index : int
            The number of basin hopping moves to get to this structure.
        energy : float
            The energy of the structure.
        volume : float
            The volume of the structure.
        ranked_atoms : dict
            Lists of integers ranking the atoms according to BVS/site potential
            for each atomic species in the structure.
        bvs_atoms : list
            The value of the Bond Valence Sum for each atom in the structure.
        bvs_sites : list
            The value of the Bond Valence Sum for each sites in the structure, with every type of
            atom present in each site.
        potentials : list
            Site potential for each atom in the current structure.
        derivs : list
            Resolved derivatives of the site potentials for each atom in the current structure.
        dopant_atoms : list
            List of atomic species that can be substituted into the structure.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 10/04/2019
        """

        if dopant_atoms is None:
            dopant_atoms = []
        if derivs is None:
            derivs = []
        if potentials is None:
            potentials = []
        if bvs_sites is None:
            bvs_sites = []
        if bvs_atoms is None:
            bvs_atoms = []
        if ranked_atoms is None:
            ranked_atoms = {}
        self.atoms = atoms

        # the tags are for doping purposes; tag = 1 atoms are in the structure; tag = 0 atoms are in doping pool
        if self.atoms:
            for atom in self.atoms:
                atom.tag = 1

        self.index = index
        self.energy = energy
        self.volume = volume
        self.ranked_atoms = ranked_atoms
        self.bvs_atoms = bvs_atoms
        self.bvs_sites = bvs_sites
        self.potentials = potentials
        self.derivs = derivs

        self.dopant_atoms = dopant_atoms
        self.dopant_atom_charges = dopant_atom_charges

        if self.dopant_atoms:
            self.dopant_atom_labels = dopant_atoms.copy()

            for i, (atomic_symbol, atomic_charge) in enumerate(
                zip(dopant_atoms, dopant_atom_charges)
            ):
                if type(atomic_symbol) == str:
                    dopant_atoms[i] = Atom(
                        atomic_symbol,
                        position=[math.nan, math.nan, math.nan],
                        charge=float(atomic_charge),
                        magmom=0,
                        index=None,
                    )
                elif type(atomic_symbol) == Atom:
                    dopant_atoms[i] = Atom(
                        atomic_symbol.symbol,
                        position=[math.nan, math.nan, math.nan],
                        charge=float(atomic_charge),
                        magmom=0,
                        index=None,
                    )
                dopant_atoms[i].tag = 0

    def write_cif(self, filename):
        """
        Write the structure to a cif file.

        Parameters
        ----------
        filename : str
            The name of the cif file.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 03/08/2017
        """

        self.atoms.write(filename, format="cif")


# =============================================================================
# =============================================================================
class Counts(object):
    """
    Keeps track of the number of structures that achieve certain outcomes.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2017
    """

    # =========================================================================
    def __init__(
        self,
        accepted=0,
        converged=0,
        unconverged=0,
        repeated=0,
        timed_out=0,
        zero_conv=0,
    ):
        """
        Initialise the counts.

        Parameters
        ----------
        accepted : int
            Number of accepted basin hopping moves.
        accepted : int
            Number of converged structures.
        repeated : int
            Number of moves resulting in repeated structures.
        timed_out : int
            Number of structures that time out in GULP.
        zero_conv : int
            1 if the first structure converged, 0 otherwise.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 09/08/2017
        """

        self.accepted = accepted
        self.converged = converged
        self.unconverged = unconverged
        self.repeated = repeated
        self.timed_out = timed_out
        self.zero_conv = zero_conv


# =============================================================================
# =============================================================================
def write_output_file_header(output, params):
    """
    Write the header for the start of the run in the ChemDASH output file

    Parameters
    ----------
    output : file
        The open file object for the ChemDASH output file.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    from datetime import datetime

    line_chars = 80
    title_chars = int(0.5 * (line_chars - 10))

    output.write("\n")
    output.write("#" * line_chars + "\n")
    output.write("#" + " " * title_chars + "ChemDASH" + " " * title_chars + "#\n")
    output.write("#" * line_chars + "\n")
    output.write("\n")

    output.write(f"Calculation started {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

    output.write("Summary of Inputs\n")
    output.write("-" * line_chars + "\n")

    sorted_keywords = sorted(params.keys())
    for keyword in sorted_keywords:

        if params[keyword]["specified"]:
            value = params[keyword]["value"]

            # Check for lists, output them as comma separated values
            if not isinstance(value, str) and isinstance(value, collections.Sequence):

                # Check for lists of lists, output them as comma separated values
                if value:  # Checks list is non-empty

                    if not isinstance(value[0], str) and isinstance(
                        value[0], collections.Sequence
                    ):
                        output.write(
                            "{0:30} = {1}\n".format(
                                keyword,
                                ", ".join(
                                    sorted(
                                        [
                                            str(", ".join([str(y) for y in x]))
                                            for x in value
                                        ]
                                    )
                                ),
                            )
                        )
                    else:
                        output.write(
                            "{0:30} = {1}\n".format(
                                keyword, ", ".join(sorted([str(x) for x in value]))
                            )
                        )

            # Write dictionaries as comma separated "key: value" pairs
            elif isinstance(value, dict):
                output.write(
                    "{0:30} = {1}\n".format(
                        keyword,
                        ", \n                                 ".join(
                            sorted(
                                [
                                    "{0}: {1}".format(key, val)
                                    for (key, val) in value.items()
                                ]
                            )
                        ),
                    )
                )

            else:
                output.write("{0:30} = {1}\n".format(keyword, value))

    output.write("-" * line_chars + "\n")
    output.write("\n")

    return None


# =============================================================================
def optimise_structure(structure, params, additional_inputs, structure_index, outcomes):
    """
    This routine optimises the input structure using the chosen calculator.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    additional_inputs : dict
        Dictionary containing calculator inputs for each individual stage of the optimisation.
    structure_index : int
        The index for the structure being considered - used to label the calculator output files.
    outcomes : dict
        Dictionary of the different GULP outcomes and the number of times they occurred.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure with atomic positions and unit cell parameters optimised.
    result : string
        The result of the calculation,
        either "converged", "unconverged", "[calculator] failure", or "timed out"
    outcomes : dict
        Updated dictionary of the different GULP outcomes and the number of times they occurred.
    time : float
        Time taken for the optimisation.

    ---------------------------------------------------------------------------
    Paul Sharp 15/06/2020
    """

    if params["calculator"]["value"] == "gulp":

        gulp_files = [
            "structure_" + str(structure_index) + "_" + suffix
            for suffix in params["gulp_files"]["value"]
        ]
        (
            new_structure,
            result,
            outcome,
            calculation_time,
        ) = gulp_calc.multi_stage_gulp_calc(
            structure,
            params["num_calc_stages"]["value"],
            gulp_files,
            params["gulp_keywords"]["value"],
            additional_inputs["gulp_keywords"],
            params["gulp_options"]["value"],
            additional_inputs["gulp_options"],
            additional_inputs["gulp_max_gnorms"],
            params["gulp_shells"]["value"],
            params["gulp_library"]["value"],
        )

        outcomes = gulp_calc.update_outcomes(outcome, outcomes)

    elif params["calculator"]["value"] == "vasp":

        vasp_file = "structure_" + str(structure_index) + ".vasp"

        # check input parameters
        # check_input_params = check_vasp_input_parameters()

        new_structure, result, calculation_time = vasp_calc.multi_stage_vasp_calc(
            structure,
            params["num_calc_stages"]["value"],
            vasp_file,
            params["vasp_settings"]["value"],
            additional_inputs["vasp_settings"],
            params["vasp_max_convergence_calcs"]["value"],
            params["save_outcar"]["value"],
            params["ionic_convergence_steps"]["value"],
        )
    else:
        raise ValueError(
            "Calculator is not specified or supported. Valid calculators are 'gulp' or 'vasp'"
        )
    return new_structure, result, outcomes, calculation_time


# =============================================================================
def accept_structure(
    structure,
    params,
    energy_step_file,
    result,
    basins,
    calc_bvs,
    calc_pot,
    energy_file,
    bvs_file,
    potentials_file,
    derivs_file,
    unrelaxed_atoms,
):
    """
    Update ChemDASH records when a new structure is accepted.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    energy_step_file : file
        The open file object for the ChemDASH energy step file.
    result : string
        The result of the GULP/VASP calculation,
        either "converged", "unconverged", "[calculator] failure", or "timed out"
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    calc_bvs, calc_pot : boolean
        True if BVS/site potential values need to be calculated.
    energy_file : file
        The open file object for the ChemDASH energy file.
    bvs_file, potentials_file, derivs_file : TextIO
        Files recording BVS/site potential values.
    unrelaxed_atoms : ASE atoms
        ASE atoms object for the structure before relaxation.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure with atomic positions and unit cell parameters optimised.
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    # Set up vacancy grids if we are using optimised geometries.
    if params["update_atoms"]["value"] and params["vacancy_grid"]["value"]:
        # Set up vacancy grid and update atoms in current structure if we are using this optimised geometry
        vacancy_grid = initialise.create_vacancy_grid(
            strip_vacancies(structure.atoms.copy()),
            params["vacancy_separation"]["value"],
            params["vacancy_exclusion_radius"]["value"],
        )
        structure.atoms = initialise.populate_points_with_vacancies(
            strip_vacancies(structure.atoms.copy()), vacancy_grid
        )

    # Structures will usually have converged, this covers the case where
    # the initial structure does not converge but the run continues.
    if result == "converged":

        energy_file.write(
            "{0:d} {1:.8f} {2:.8f}\n".format(
                structure.index, structure.energy, structure.volume
            )
        )
        energy_step_file.write(
            "{0:d} {1:.8f} {2:.8f}\n".format(
                structure.index, structure.energy, structure.volume
            )
        )

        basins = swap.update_basins(basins, structure.energy)

        if calc_bvs:
            structure.bvs_atoms = bonding.bond_valence_sum_for_atoms(
                structure.atoms.copy()
            )
            structure.bvs_sites = bonding.bond_valence_sum_for_sites(
                structure.atoms.copy(),
                params["bvs_dopant_atoms"]["value"],
                params["bvs_dopant_atoms_charges"]["value"],
            )

            # TODO: The bvs.atoms are never actually written to file. Is this needed?
            output_list(bvs_file, structure.index, structure.bvs_sites)

            # These next two lines are not actually used anywhere but have left as comments
            # atom_indices = [atom.index for atom in structure.atoms if atom.symbol != "X"]
            # desired_atoms = swap.find_desired_atoms(structure.bvs_sites, atom_indices)

        # Obtain values for the site potential of all atoms and vacancies in the relaxed structure if necessary.
        if calc_pot:
            pot_structure = Structure(structure.atoms, structure.index)
            structure.potentials, structure.derivs, _, _, _ = update_potentials(
                pot_structure, params["gulp_library"]["value"]
            )

        output_list(potentials_file, structure.index, structure.potentials)
        output_list(derivs_file, structure.index, structure.derivs)

        # Set swap rankings for each atom
        structure = swap.update_atom_rankings(
            structure, params["atom_rankings"]["value"]
        )

    # If we use not using optimised geometries, reset the geometry to that of the unrelaxed structure and set up
    # vacancy grid.
    if not params["update_atoms"]["value"]:

        structure.atoms = unrelaxed_atoms.copy()

        if params["vacancy_grid"]["value"]:
            # Set up vacancy grid and in current structure if we are using the unoptimised geometry
            vacancy_grid = initialise.create_vacancy_grid(
                strip_vacancies(structure.atoms.copy()),
                params["vacancy_separation"]["value"],
                params["vacancy_exclusion_radius"]["value"],
            )
            structure.atoms = initialise.populate_points_with_vacancies(
                strip_vacancies(structure.atoms.copy()), vacancy_grid
            )

    return structure, basins


# =============================================================================
def update_potentials(structure, gulp_lib):
    """
    Re-calculate the potentials for the ChemDASH structure.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    gulp_lib : string
        Name of the file containing the GULP force field.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure class with updated potentials.
    result : string
        The result of the calculation,
        either "converged", "unconverged", "GULP failure", or "timed out"
    outcomes : dict
        Updated dictionary of the different GULP outcomes and the number of times they occurred.
    calculation_time : float
        Time taken for the site potential calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 09/04/2019
    """

    structure, result, outcome, calculation_time = gulp_calc.multi_stage_gulp_calc(
        structure,
        1,
        ["pot"],
        ["sing pot"],
        [""],
        [""],
        [[""]],
        [""],
        [],
        gulp_lib,
        remove_vacancies=False,
    )

    return structure.potentials, structure.derivs, result, outcome, calculation_time


# =============================================================================
def generate_new_structure(
    structure,
    params,
    output,
    valid_swap_groups,
    sorted_atomic_indices,
    rng,
    moments,
    nth_step=None,
    dopant_atoms=None,
):
    """
    Generate a new structure by swapping atoms.

    Parameters
    ----------
    structure : ChemDASH Structure
        The ChemDASH Structure object for the current structure.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    output : file
        The open file object for the ChemDASH output file.
    valid_swap_groups : list
        List of all valid swap groups with their weightings.
    sorted_atomic_indices : dict
        Dictionary of indices for the each atomic species, sorted by the values
        in the chosen ranking list.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.
    moments: list
        List of magnetic moments for the current structure


    Returns
    -------
    new_atoms : ASE atoms
        The ASE atoms object for the new structure.
    change : str
        A string indicating whether a dope or swap has occurred

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    change = None
    doping_threshold = params["doping_threshold"]["value"]

    # this if statement is a quick hack to dope atoms doping_threshold * 100 % of the time
    if params["random_dopant_atoms"]["value"]:

        # if no dopable atoms are specified, this defaults to all current atoms being allowed
        if not params["dopable_atoms"]["value"]:
            params["dopable_atoms"]["value"] = structure.atoms.get_chemical_symbols()

        # if doping is done every nth step
        if params["nth_doping_step"]["value"] != 0:
            if nth_step == params["nth_doping_step"]["value"] - 1:
                change = "dope"
                nth_step = 0
            else:
                nth_step += 1

        # if doping is done when doping_threshold is exceeded
        elif rng.real() < doping_threshold and params["nth_doping_step"]["value"] == 0:
            change = "dope"

        # if doping is to happen this step
        if change == "dope":
            output.write(
                "We will attempt to perform a doping from {0} atoms into the group of dopable atoms:"
                " {1}\n".format(
                    params["random_dopant_atoms"]["value"],
                    params["dopable_atoms"]["value"],
                )
            )

            new_atoms, new_dopant_atoms, dope_in, dope_out = dope.random_atom_dope(
                structure.atoms.copy(),
                params["random_dopant_atoms"]["value"],
                params["dopable_atoms"]["value"],
                rng,
                moments=moments,
            )
            # store the new pool of possible dopants

            # params["random_dopant_atoms"]["value"] = random_dopant_atoms

            return new_atoms, change, dope_in, dope_out, new_dopant_atoms, nth_step

    change = "swap"
    swap_weightings = [group[1] for group in valid_swap_groups]
    atom_group = valid_swap_groups[rng.weighted_choice(swap_weightings)][0]
    elements_list, max_swaps = swap.determine_maximum_swaps(structure, atom_group)
    if dopant_atoms:
        elements_list += dopant_atoms

    num_swaps = swap.choose_number_of_atoms_to_swap(
        max_swaps,
        params["number_weightings"]["value"],
        rng,
        params["pair_weighting"]["value"],
    )

    output.write(
        "We will attempt to perform a swap of {0:d} atoms from the group: {1}\n".format(
            num_swaps, atom_group
        )
    )

    selection_pool, num_swaps = swap.generate_selection_pool(elements_list, num_swaps)
    output.write(
        "A valid selection pool has been generated to swap {0:d} atoms.\n".format(
            num_swaps
        )
    )

    if params["atom_rankings"]["value"] != "random":
        output.write(
            "We will consider an extra {0:d} atoms at the top of the ranking list for each species.\n".format(
                params["directed_num_atoms"]["value"]
            )
        )

    output.write("The value of kT is {0:f} eV/atom".format(params["temp"]["value"]))

    swap_list = swap.generate_swap_list(selection_pool, num_swaps, rng)

    # swap atoms
    new_structure, swap_text, moments = swap.swap_atoms(
        structure,
        swap_list,
        copy.deepcopy(sorted_atomic_indices),
        params["directed_num_atoms"]["value"],
        params["initial_structure_file"]["specified"],
        params["vacancy_exclusion_radius"]["value"],
        rng,
        params["force_vacancy_swaps"]["value"],
        mag_moms=moments,
        magnetic_structure_only=params["swap_magnetic_moments"]["value"],
    )

    if params["verbosity"]["value"] == "verbose":
        output.write("\n")
        """for entry in swap_text:
            output.write(
                "The {e[0]} atom at {e[1]} (atom {e[2]:d}) has been replaced by a {e[3]} atom\n".format(e=entry))"""
    output.write("\n")

    return new_structure, change, None, None, None, nth_step, moments


# =============================================================================
def strip_vacancies(structure):
    """
    This code removes vacancies, represented by "X" atoms, from an ase structure.

    Parameters
    ----------
    structure : ase atoms
        A structure that includes vacancies represented as an "X" atom.

    Returns
    -------
    structure : ase atoms
        The structure with vacancies removed.

    ---------------------------------------------------------------------------
    Paul Sharp 02/05/2017
    """

    del structure[[atom.index for atom in structure if atom.symbol == "X"]]

    return structure


# =============================================================================
def output_list(file_object, header, list_of_floats):
    """
    This routine outputs a list of floats to a file, keeping the values in the
    floating point format if possible.

    Parameters
    ----------
    file_object : file
        An open file object
    header : string
        A string that precedes the list of floating point values.
    list_of_floats : list
        The list of floats we wish to output.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 05/06/2020
    """

    file_object.write(str(header))

    for element in list_of_floats:
        try:
            file_object.write(" {0:.8f}".format(element))
        except (TypeError, ValueError):
            file_object.write(" {0}".format(element))

    file_object.write("\n")

    return None


# =============================================================================
def report_rejected_structure(output, result, calculator, structure_count):
    """
    Write the reason for a rejection of a structure to the output file.

    Parameters
    ----------
    output : file
        The open file object for the ChemDASH output file.
    result : string
        The result of the structural optimisation.
    calculator : string
        The name of the materials modelling code used for the optimisation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out calculations considered so far.

    Returns
    -------
    structure_count : Counts
        Updated number of accepted, converged, unconverged/failed, and
        repeated structures and timed out calculations considered so far.

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    if result == calculator + " failure":
        output.write(
            "{0} has failed to perform the calculation for this structure, so it will be rejected.\n".format(
                calculator.upper()
            )
        )

    if result == "timed out":
        output.write(
            "The optimisation of this structure has timed out, so it will be rejected.\n"
        )
        structure_count.timed_out += 1

    if result == "unconverged":
        output.write(
            "The optimisation of this structure has not converged, so it will be rejected.\n"
        )
        structure_count.unconverged += 1

    return structure_count


# =============================================================================
def report_statistics(
    output, basins, outcomes, structure_count, total_structures, calculator
):
    """
    This routine outputs the visited basins, GULP outcomes and output file.

    Parameters
    ----------
    output : file
        An open file object
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    total_structures : int
        Number of structures considered so far.
    calculator : str
        The materials modelling code used for structure relaxation.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    line_chars = 80
    basin_chars = 20

    # Report record of visited basins
    output.write("\n")
    output.write("In total, {0:d} basins were visited\n".format(len(basins)))
    output.write("\n")

    output.write("Basins:\n")
    output.write("-" * basin_chars + "\n")
    for energy in basins:
        output.write("| {0:.5f} | {1:4d} |\n".format(energy, basins[energy]))

    output.write("-" * basin_chars + "\n")
    output.write("\n")

    # Report convergence rate
    unique_structures = total_structures - structure_count.repeated
    try:
        convergence_rate = (
            100.0 * float(structure_count.converged) / float(unique_structures)
        )
    except ZeroDivisionError:
        convergence_rate = 0.0

    output.write(
        "{0:d} structures out of {1:d} converged, so the convergence rate is {2:f}%\n".format(
            structure_count.converged, unique_structures, convergence_rate
        )
    )
    output.write("\n")

    # Report results of optimisations
    output.write("Optimisation Results:\n")
    output.write("-" * line_chars + "\n")
    output.write("| {0:4d} | Converged\n".format(structure_count.converged))
    output.write("| {0:4d} | Unconverged\n".format(structure_count.unconverged))
    output.write("| {0:4d} | Timed Out\n".format(structure_count.timed_out))
    output.write("-" * line_chars + "\n")
    output.write("\n")

    # Report GULP outcomes
    if calculator == "gulp":

        output.write("GULP Outcomes:\n")
        output.write("-" * line_chars + "\n")

        for gulp_outcome in outcomes:
            output.write(
                "| {0:4d} | {1}\n".format(outcomes[gulp_outcome], gulp_outcome)
            )

        output.write("-" * line_chars + "\n")
        output.write("\n")

    # Report swap acceptance rate
    try:
        swap_acceptance = (
            100.0
            * float(structure_count.accepted)
            / float(structure_count.converged - structure_count.zero_conv)
        )
    except ZeroDivisionError:
        swap_acceptance = 0.0

    output.write(
        "Of the converged structures, {0:d} out of {1:d} swaps were accepted,"
        " so the swap acceptance rate was {2:f}%\n".format(
            structure_count.accepted,
            structure_count.converged - structure_count.zero_conv,
            swap_acceptance,
        )
    )
    output.write("\n")

    return None


# =============================================================================
def read_restart_file(restart_file):
    """
    This routine reads a ".npz" file with all information needed to restart a
    calculation.

    Parameters
    ----------
    restart_file : str
       Name of the ".npz" archive.

    Returns
    -------
    best_structure, current_structure : Structure
        ASE atoms object and associated data for the best and current structures.
    atomic_numbers_list : int
        Atomic numbers of each atom for all unique structures considered so far.
    positions_list : float
        Positions of each atom for all unique structures considered so far.
    basins : dict
        Value of energy for each basin visited, and the number of times each basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    structure_index : int
        Number of structures considered so far.

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    with np.load(restart_file, allow_pickle=True) as restart_data:
        best_structure = restart_data["best_structure"][()]
        current_structure = restart_data["current_structure"][()]

        atomic_numbers_list = restart_data["atomic_numbers_list"].tolist()
        positions_list = restart_data["positions_list"].tolist()

        # Final index needed to return a dictionary as opposed to an array
        basins = restart_data["basins"][()]
        outcomes = restart_data["outcomes"][()]

        structure_count = restart_data["structure_count"][()]
        structure_index = int(restart_data["structure_index"])

        magnetic_moments = list(restart_data["magnetic_moments"][()])

    return (
        best_structure,
        current_structure,
        atomic_numbers_list,
        positions_list,
        basins,
        outcomes,
        structure_count,
        structure_index,
        magnetic_moments,
    )


# =============================================================================
def search_local_neighbourhood(structure, output, params):
    """
    LCN algorithm.

    Parameters
    ----------
    structure : ChemDASH Structure
    output : file
    params : dict


    Returns
    -------
    structure : ChemDASH Structure


    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    lcn_structure = Structure(structure.atoms, structure.index)

    lcn_structure, _, _, _ = gulp_calc.multi_stage_gulp_calc(
        lcn_structure,
        1,
        ["lcn"],
        "sing",
        [""],
        [""],
        [[""]],
        [""],
        [],
        params["gulp_library"]["value"],
    )
    lcn_initial_energy = lcn_structure.energy
    output.write("\n")
    output.write(
        "LCN {0:d}: Initial energy = {1:.8f} eV/atom\n".format(
            structure.index, lcn_structure.energy
        )
    )
    (
        lcn_structure,
        lcn_final_energy,
        lcn_time,
        atom_loops,
        sp_calcs,
    ) = neighbourhood.local_combinatorial_neighbourhood(
        lcn_structure,
        params["neighbourhood_atom_distance_limit"]["value"],
        params["num_neighbourhood_points"]["value"],
        params["gulp_library"]["value"],
    )

    output.write(
        "LCN {0:d}: Final energy = {1:.8f} eV/atom\n".format(
            structure.index, lcn_final_energy
        )
    )
    output.write(
        "LCN {0:d}: Change in energy = {1:.8f} eV/atom\n".format(
            structure.index, lcn_final_energy - lcn_initial_energy
        )
    )
    output.write("LCN {0:d}: Time = {1:.8f}s\n".format(structure.index, lcn_time))
    output.write(
        "LCN {0:d}: Number of atom loops = {1:d}\n".format(structure.index, atom_loops)
    )
    output.write(
        "LCN {0:d}: Number of single-point energy calculations = {1:d}\n".format(
            structure.index, sp_calcs
        )
    )
    output.write(
        "LCN {0:d} Data: {1:.8f} {2:.8f} {3:d} {4:d}\n".format(
            structure.index,
            lcn_final_energy - lcn_initial_energy,
            lcn_time,
            atom_loops,
            sp_calcs,
        )
    )
    output.write("\n")

    structure.atoms = lcn_structure.atoms

    return structure


# =============================================================================
def write_restart_file(
    best_structure,
    current_structure,
    atomic_numbers_list,
    positions_list,
    basins,
    outcomes,
    structure_count,
    structure_index,
    magnetic_moments,
):
    """
    This routine writes the file with all information needed to restart a calculation.

    Parameters
    ----------
    best_structure, current_structure : Structure
        ASE atoms object and associated data for the best and current structures.
    atomic_numbers_list : int
        Atomic numbers of each atom for all unique structures considered so far.
    positions_list : float
        Positions of each atom for all unique structures considered so far.
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    structure_index : int
        Number of structures considered so far.
    magnetic_moments: list
        A List of the final magnetic moments of the structure

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    np.savez(
        "restart",
        best_structure=best_structure,
        current_structure=current_structure,
        atomic_numbers_list=atomic_numbers_list,
        positions_list=positions_list,
        basins=basins,
        outcomes=outcomes,
        structure_count=structure_count,
        structure_index=structure_index,
        magnetic_moments=magnetic_moments,
    )

    return None


#  ########### NEED TO INCLUDE TRAJECTORIES IN RESTART FILE #############

#  ########### ALSO OVERRIDE OUTPUT_TRAJECTORY WHEN VERSION OF ASE IS INSUFFICIENT #################
