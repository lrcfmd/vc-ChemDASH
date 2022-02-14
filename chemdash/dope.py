try:
    import chemdash.gulp_calc as gulp_calc
except ModuleNotFoundError:
    import initialise as initialise
    import inputs as inputs
    import neighbourhood as neighbourhood
    import rngs as rngs
    import symmetry as symmetry
    import vasp_calc as vasp_calc
    from inputs import initialise_default_params

from ase import Atom
from ase.io import read
from copy import deepcopy


def random_atom_dope(structure, dopant_pool, dopable_atoms, rng, moments=None):
    """
    A function that randomly dopes a single atom into a structure

    Parameters
    ----------
    structure: ASE atoms
        The ASE atoms object for the current structure
    dopant_pool: list
        List of atoms that can be doped into the structure
    dopable_atoms: list
        List of atoms that can be doped out of the structure
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.
    moments: list
        List of magnetic moments to be assigned to the current structure


    Returns
    -------
    new_atoms: ASE atoms
        An ASE atoms structure of the doped structure
    new_dopant_pool: list
        List of new atoms that can be doped into a structure with doped out atom replacing an atom from dopant_pool
    dope_in.symbol: str
        Doped in element
    dope_out.symbol: str
        Doped out element
    """

    dope_in = "X"
    dope_out = "X"
    new_structure = None
    dope_out_index = None

    # while loop exited when a non trivial dope is achieved
    while dope_in == dope_out:

        # get current atom indices for atoms that can be doped
        dope_out_indices = [
            i for i, atom in enumerate(structure) if atom.symbol in dopable_atoms
        ]

        # randomly choose an index to swap out
        dope_out_index = dope_out_indices[rng.int_range(0, len(dope_out_indices))]

        # converts a list of dopants to ase atom objects
        if type(dopant_pool[0]) == str:
            dopant_pool_atoms = [Atom(dopant) for dopant in dopant_pool]
        else:
            dopant_pool_atoms = dopant_pool.copy()

        # generates a random index to use a random dopant atom
        dopant_pool_index = rng.int_range(0, len(dopant_pool_atoms))

        # copy structure
        new_structure = deepcopy(structure)
        new_dopant_pool = deepcopy(dopant_pool_atoms)

        dope_in = dopant_pool_atoms[dopant_pool_index]
        dope_out = structure[dope_out_index]

    # set dopant in place of current species
    new_structure.numbers[dope_out_index] = dopant_pool_atoms[dopant_pool_index].number

    # set magnetic moments
    new_structure.set_initial_magnetic_moments(moments)

    # set current species into dopant pool
    new_dopant_pool[dopant_pool_index].number = structure.numbers[dope_out_index]

    return new_structure, new_dopant_pool, dope_in.symbol, dope_out.symbol


def directed_atom_dope(structure, dopant_pool, dopant_atoms, rng, moments=None):
    pass


def get_energy(species_outcar_dir):
    """
    This function gets the final energy from a given OUTCAR file.
    NOTE: This function is defunct now due to solid solution energies being used

    Args:
        species_outcar_dir (str):
            Directory where OUTCAR file is located

    Returns:
        energy (float):
            Final energy from OUTCAR file
    """
    energy = read(f"./{species_outcar_dir}O/OUTCAR").get_potential_energy()
    return energy


def get_solid_solution_energy_difference(new_structure, end_members):
    """
    Gets energy difference between new structure and end members

    Example:
        single doping of Mn into Zn8Fe16O32

            xMnFe2O4 + (8-x)ZnFe2O4 --> Mn(x)Zn(8-x)Fe2O4

             ΔE = E(Mn8xZn8(1-x)Fe16O32)
    Parameters
    ----------
    new_structure: ASE atoms

    end_members: dict

    Returns
    -------

    """

    # get labels, energies from keys, values
    keys, values = list(end_members.keys()), list(end_members.values())

    # get a list of atoms in the structure
    list_new_structure_atoms = [a.symbol for a in new_structure.atoms]

    # get energies as floats
    end_member_1_energy, end_member_2_energy = float(values[0]), float(values[1])

    # get a count of each differentiating species of each end member in the structure
    end_member_1_count = list_new_structure_atoms.count(keys[0])
    end_member_2_count = list_new_structure_atoms.count(keys[1])

    # get a sum of the number of end members
    end_member_sum = end_member_1_count + end_member_2_count

    # get proportion of each end member in the structure
    x = end_member_1_count / end_member_sum
    y = end_member_2_count / end_member_sum

    # calculate solid solution energy difference
    solid_solution_energy = (
        new_structure.energy - (x * end_member_1_energy) - (y * end_member_2_energy)
    )

    return solid_solution_energy


def get_doped_energy_difference(
    current_structure, new_structure, dope_in_oxide_species, dope_out_oxide_species
):
    """

    example 4 energies needed:

        Mn2Fe4O8 + ZnO - MnO --> MnZnFe4O8

        with ΔE showing relative stability for the metropolis criterion

    Parameters
    ----------
    structure: ase atoms

    dopant: str
        string of the dopant atom into the old structure to form new structure



    Returns
    -------

    """
    if not dope_in_oxide_species or not dope_out_oxide_species:
        raise ValueError("No doped species is specified!")

    new_structure_energy = new_structure.energy * len(new_structure.atoms)
    current_structure_energy = current_structure.energy * len(current_structure.atoms)

    dope_in_oxide_energy = get_energy(dope_in_oxide_species)
    dope_out_oxide_energy = get_energy(dope_out_oxide_species)

    reaction_energy = (
        new_structure_energy
        - (
            current_structure_energy
            + (dope_in_oxide_energy / 2)
            - (dope_out_oxide_energy / 2)
        )
    ) / len(new_structure.atoms)

    return reaction_energy


def compare_energy_above_hull(structure, hull_data):
    return structure
