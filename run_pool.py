import datetime as datetime
import logging
import multiprocessing as mp
import sys
import time as time

import cirq as cirq
import numpy as numpy
import pandas as pandas
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4
from scipy.optimize import minimize

from vqe_functions import *


def calculate_scan(molecule_name, basis, multiplicity, charge, counts):

    length_bounds = [0.2, 3]
    logging.info("Calculating %s scan.", molecule_name)
    file_name = "./results/VQE_scan_{}_{}.csv".format(molecule_name, datetime.datetime.now())
    
    molecular_data_list = list()

    for length in numpy.linspace(length_bounds[0], length_bounds[1], counts):
        length = round(length, 3)
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
                    charge, filename = './data/{}_{}_molecule.data'.format(molecule_name, length), description=str(length))

                molecular_data = run_psi4(molecular_data,
                                        run_scf=True,
                                        run_mp2=True,
                                        run_cisd=True,
                                        run_ccsd=True,
                                        run_fci=True)
                
                logging.info("Psi4 calculations were succesful.")
                # Do the calculations.
                molecular_data_list.append([molecular_data, file_name])
                break
            except Exception as exc:
                logging.error(exc)
                length += 0.001
                logging.info("New length set: %s", length)

    #Pool
    pool = mp.Pool(processes= 10)
    result = pool.map(single_point_pool, molecular_data_list)



def single_point_pool(val):
    molecular_data = val[0]
    file_name = val[1]
    length = molecular_data.description
    logging.info("Starting expectation value calculations for length %s.", length)
    
    electron_count = molecular_data.n_electrons
    qubit_count = molecular_data.n_qubits
    
    qubit_operator_list = get_qubit_operators(molecular_data)
    uccsd = initial_hartree_fock(electron_count, qubit_count)
    unitary = create_uccsd(qubit_operator_list, qubit_count, 't')
    uccsd.append(unitary, strategy = cirq.InsertStrategy.NEW)

    hamiltonian = get_measurement_hamiltonian(molecular_data)
    
    #Simulation
    options = {'t': 16}
    simulator = qsimcirq.QSimSimulator(options)
    cirq.DropEmptyMoments().optimize_circuit(circuit = uccsd)

    qubit_map = get_qubit_map(qubit_count)
    pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

    bounds = list()
    for i in range(len(qubit_operator_list)):
        bounds.append([-numpy.pi, numpy.pi])

    start_time = time.time()
    
    optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                        , method = 'Nelder-Mead',
                        args = (simulator, uccsd, pauli_sum, qubit_map),
                        options = {'disp' : True, 'ftol': 1e-4})


    elapsed_time = time.time() - start_time

    logging.info(optimize_result)
    logging.info("Elapsed time: %s", elapsed_time)

    energy_min = optimize_result.fun
    nfev = optimize_result.nfev
    nit = optimize_result.nit

    # Result save.
    file = open(file_name, "a")
    file.write("{}, {}, {}, {}, {}, \n".format(energy_min, nfev, 
                                            nit, elapsed_time, 
                                            length))
    file.close()
    logging.info("Result at %s saved.", length)

    return energy_min, nfev, nit, elapsed_time

                        

def main():
    """
    -min molecule \n
    -scan counts molecule \n
    Molecules: H2, LiH, BeH2
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
        #calculate_minimum(molecule_name, basis, multiplicity, charge)
        pass
    elif scan_mode:
        calculate_scan(molecule_name, basis, multiplicity, charge, counts)
    else:
        print("No mode selected!")
        exit()
    
    elapsed_time = time.time() - start
    logging.info("Total programm run time: %s s.", elapsed_time)
    exit()


if __name__ == "__main__":
    main()
