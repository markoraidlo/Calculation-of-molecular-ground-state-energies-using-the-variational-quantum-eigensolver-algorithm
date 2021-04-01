import sys
import logging
import time as time
import datetime as datetime
import multiprocessing as mp

import cirq as cirq
import numpy as numpy
import pandas as pandas
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4
from scipy.optimize import minimize

from functions import *


def calculate_minimum(molecule_name, basis, multiplicity, charge):
    """[summary]

    Args:
        molecule_name ([type]): [description]
        basis ([type]): [description]
        multiplicity ([type]): [description]
        charge ([type]): [description]
    """
    logging.info("Calculating %s minimum.", molecule_name)

    min_geometry_dict = {'H2': [[ 'H', [ 0, 0, 0]],
                            [ 'H', [ 0, 0, 0.74]]], 
                    'LiH': [['Li', [0, 0, 0]] ,
                            ['H', [0, 0, 1.5949]]],
                    'BeH2': [['Be', [ 0, 0, 0 ]],
                            ['H', [ 0, 0, 1.3264]],
                            ['H', [ 0, 0, -1.3264]]],
                    'H2O': [['O', [-0.053670056908, -0.039737675589, 0]],
                            ['H', [-0.028413670411,  0.928922556351, 0]],
                            ['H', [0.880196420813,  -0.298256807934, 0]]]}
    geometry = min_geometry_dict[molecule_name]

    #Should always work for known geometry
    molecular_data = MolecularData(geometry, basis, multiplicity,
        charge, filename = './data/{}_min_molecule.data'.format(molecule_name))

    molecular_data = run_psi4(molecular_data,
                            run_scf=True,
                            run_mp2=True,
                            run_cisd=True,
                            run_ccsd=True,
                            run_fci=True)

    # Do the calculations.
    min_result = single_point_calculation(molecular_data)

    # Result save.
    file = open("./results/VQE_min_{}_{}.csv".format(molecule_name, datetime.datetime.now()), "a")
    file.write("{}, {}, {}, {}, \n".format(min_result[0], min_result[1]
                                         , min_result[2], min_result[3]))
    file.close()
    logging.info("Results saved.")

        #TODO: Make into log
    """
    molecule_name = molecular_data.name
    electron_count = molecular_data.n_electrons
    qubit_count = molecular_data.n_qubits
    orbital_count = molecular_data.n_orbitals
    qubit_op_count = len(qubit_operator_list)

    UCCSD = initial_hartree_fock(electron_count, qubit_count)
    UNITARY = create_uccsd(qubit_operator_list, qubit_count, 't')
    UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)
    uccsd_len = len(UCCSD)
    print("#####################################################################")
    print(molecule_name)
    print("Electron count: {}".format(electron_count))
    print("Qubit count: {}".format(qubit_count))
    print("Orbital count: {}".format(orbital_count))
    print("Qubit operator count: {}".format(qubit_op_count))
    print("Length of UCCSD circuit: {}".format(uccsd_len))
    print("#####################################################################")
    print("Minimum energy: {}".format(energy_min))
    print("Time elapsed: {} s".format(end_time))
    print("Number of evaluations of the objective function: {}".format(nfev))
    print("Number of iterations performed by the optimizer: {}".format(nit))
    print("#####################################################################")
    """
    

def calculate_scan(molecule_name, basis, multiplicity, charge, counts):
    length_bounds = [0.5, 3]
    logging.info("Calculating %s scan.", molecule_name)
    file_name = "./results/VQE_scan_{}_{}.csv".format(molecule_name, datetime.datetime.now())
    
    for length in numpy.linspace(length_bounds[0], length_bounds[1], counts):
        while True:
            geometry_dict = {'H2': [[ 'H', [ 0, 0, 0]],
                                [ 'H', [ 0, 0, length]]], 
                        'LiH': [['Li', [0, 0, 0]] ,
                                ['H', [0, 0,  length]]],
                        'BeH2': [['Be', [ 0, 0, 0 ]],
                                ['H', [ 0, 0,  length]],
                                ['H', [ 0, 0, - length]]],
                        'H2O': [['O', [0, 0, 0]],
                                ['H', [numpy.sin(0.9163) / length,  numpy.cos(0.9163) / length, 0]],
                                ['H', [- numpy.sin(0.9163) / length,  numpy.cos(0.9163) / length, 0]]]}

            geometry = geometry_dict[molecule_name]

            try:
                logging.info("Trying psi4 calculation at length %s.", length)
                molecular_data = MolecularData(geometry, basis, multiplicity,
                    charge, filename = './data/{}_{}_molecule.data'.format(molecule_name, length))

                molecular_data = run_psi4(molecular_data,
                                        run_scf=True,
                                        run_mp2=True,
                                        run_cisd=True,
                                        run_ccsd=True,
                                        run_fci=True)
                
                logging.info("Psi4 calculations were succesful.")
                # Do the calculations.
                scan_result = single_point_calculation(molecular_data)

                # Result save.
                file = open(file_name, "a")
                file.write("{}, {}, {}, {}, {}, \n".format(scan_result[0], scan_result[1], 
                                                       scan_result[2], scan_result[3], 
                                                       length))
                file.close()
                logging.info("Result at %s saved.", length)

                break
            except Exception as exc:
                logging.error(exc)
                length += 0.000000000001
                logging.info("New length set: %s", length)
                        
def main():
    """
    -min molecule \n
    -scan counts molecule \n
    Molecules: H2, LiH, BeH2, H2O
    """
    start = time.time()
    #Program args
    args = sys.argv[1:]
    assert len(args) >= 2
    min_mode = ('-min' in args)
    scan_mode = ('-scan' in args)
    molecule_name = args[-1]
    if scan_mode:
        counts = int(args[-2])

    #Logging
    logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/LOG_{}_{}.log".format(molecule_name, datetime.datetime.now())),
        logging.StreamHandler()
    ])

    #Main variables
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    # Calculation modes:
    if min_mode:
        calculate_minimum(molecule_name, basis, multiplicity, charge)
    elif scan_mode:
        calculate_scan(molecule_name, basis, multiplicity, charge, counts)
    else:
        print("No mode selected!")
        exit()
    
    elapsed_time = time.time() - start
    logging.info("Total programm run time: %s s.", elapsed_time)
    #Log elapsed time

if __name__ == "__main__":
    main()