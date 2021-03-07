import sys
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


def calculate_minimum(geometry, basis, multiplicity, charge, molecule_name, results):
    """[summary]

    Args:
        geometry ([type]): [description]
        basis ([type]): [description]
        multiplicity ([type]): [description]
        charge ([type]): [description]
        molecule_name ([type]): [description]
        results ([type]): [description]
    """
    #Should always work for known geometry
    molecular_data = MolecularData(geometry, basis, multiplicity,
        charge, filename = './data/{}_min_molecule.data'.format(molecule_name))

    molecular_data = run_psi4(molecular_data,
                            run_scf=True,
                            run_mp2=True,
                            run_cisd=True,
                            run_ccsd=True,
                            run_fci=True)

    qubit_operator_list = get_qubit_operators(molecular_data)

    #Do the calculations
    electron_count, orbital_count, qubit_count, qubit_op_count,uccsd_len, energy_min, end_time, nfev, nit = temp([-1, molecular_data, qubit_operator_list])

    ############ Molecule output
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

    ############ Data save
    results.loc[len(results)] = [molecule_name, " ",electron_count, orbital_count, qubit_count, qubit_op_count, 
                    uccsd_len, energy_min, end_time, nfev, nit]
    results.to_csv("./results/VQE_min_{}_{}.csv".format(molecule_name, datetime.datetime.now()))


def get_data(geometry, basis, multiplicity, charge, molecule_name, counts):
    data_list = list()
    length_bounds = [0.2, 3]
    #Lineaarne 2 aatomit
    if molecule_name == 'H2' or molecule_name == 'LiH':
        #Contains length, molecular data and qubit op for every calculation:
        for length in numpy.linspace(length_bounds[0], length_bounds[1], counts):
            while True:
                if molecule_name == 'H2':
                    geometry = [[ 'H', [ 0, 0, 0]],
                                [ 'H', [ 0, 0, length]]]
                else:
                    geometry = [['Li', [0, 0, 0]] ,
                                ['H', [0, 0, length]]]

                try:
                    molecular_data = MolecularData(geometry, basis, multiplicity,
                        charge, filename = './data/{}_{}_molecule.data'.format(molecule_name, length))

                    molecular_data = run_psi4(molecular_data,
                                            run_scf=True,
                                            run_mp2=True,
                                            run_cisd=True,
                                            run_ccsd=True,
                                            run_fci=True)

                    qubit_operator_list = get_qubit_operators(molecular_data)
                    data_list.append([length, molecular_data, qubit_operator_list])
                    break
                except Exception as exc:
                    print(exc)
                    length += 0.000000000001
                    print("########################################")
                    print("ERROR FIX: length = {}".format(length))
                    print("########################################")
                        
    #Lineaarne 3 aatomit
    elif molecule_name == 'BeH2':
        length_bounds = [0.5, 3.5]
        for length_1 in numpy.linspace(length_bounds[0], length_bounds[1], counts):
            for length_2 in numpy.linspace(length_bounds[0], length_bounds[1], counts):
                while True:
                    try:
                        geometry = [['Be', [ 0, 0, 0 ]],
                                ['H', [ 0, 0, length_1]],
                                ['H', [ 0, 0, -length_2]]]
                        molecular_data = MolecularData(geometry, basis, multiplicity,
                            charge, filename = './data/{}_{}_{}molecule.data'.format(molecule_name, length_1, length_2))

                        molecular_data = run_psi4(molecular_data,
                                                run_scf=True,
                                                run_mp2=True,
                                                run_cisd=True,
                                                run_ccsd=True,
                                                run_fci=True)

                        qubit_operator_list = get_qubit_operators(molecular_data)
                        data_list.append([[length_1, -length_2], molecular_data, qubit_operator_list])
                        break

                    except Exception as exc:
                        print(exc)
                        length_1 += 0.000000000001
                        length_2 -= 0.000000000001
                        print("########################################")
                        print("ERROR FIX: lengths = {}, - {}".format(length_1, length_2))
                        print("########################################")
          
    #Mitte lineaarne 3 aatomit
    elif molecule_name == 'H2O':
        #angle_bounds = [-numpy.pi + 0.1, numpy.pi - 0.1] 
        length_bounds = [0.5, 3.5]
        #Length scan
        for length_1 in numpy.linspace(length_bounds[0], length_bounds[1], counts):
            for length_2 in numpy.linspace(length_bounds[0], length_bounds[1], counts):
                while True:
                    try:
                        geometry = [['O', [0, 0, 0]],
                                    ['H', [numpy.sin(0.9163) / length_1,  numpy.cos(0.9163) / length_1, 0]],
                                    ['H', [- numpy.sin(0.9163) / length_2,  numpy.cos(0.9163) / length_2, 0]]]
                        molecular_data = MolecularData(geometry, basis, multiplicity,
                            charge, filename = './data/{}_{}_{}molecule.data'.format(molecule_name, length_1, length_2))

                        molecular_data = run_psi4(molecular_data,
                                                run_scf=True,
                                                run_mp2=True,
                                                run_cisd=True,
                                                run_ccsd=True,
                                                run_fci=True)

                        qubit_operator_list = get_qubit_operators(molecular_data)
                        data_list.append([(length_1, length_2), molecular_data, qubit_operator_list])
                        break

                    except Exception as exc:
                        print(exc)
                        length_1 += 0.000000000001
                        length_2 -= 0.000000000001
                        print("########################################")
                        print("ERROR FIX: lengths = {}, - {}".format(length_1, length_2))
                        print("########################################")
                    
        
        #Angle scan
        """
        for angle_1 in numpy.linspace(angle_bounds[0], angle_bounds[1], counts):
            for angle_2 in numpy.linspace(angle_bounds[0], angle_bounds[1], counts):
                angle_list.append([angle_1, angle_2, basis, multiplicity, charge, molecule_name])
        """

    return data_list



def scan_scheduler(data_list):
    pool = mp.Pool(processes = 20)
    results = pool.map(temp, data_list)
    print(results)
    return results

def main():
    """
    -min molecule \n
    -scan counts molecule \n
    -min -scan counts molecule \n
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

    #Main variables
    geometry_dict = {'H2': [[ 'H', [ 0, 0, 0]],
                                [ 'H', [ 0, 0, 0.74]]], 
                        'LiH': [['Li', [0, 0, 0]] ,
                                ['H', [0, 0, 1.5949]]],
                        'BeH2': [['Be', [ 0, 0, 0 ]],
                                ['H', [ 0, 0, 1.3264]],
                                ['H', [ 0, 0, -1.3264]]],
                        'H2O': [['O', [-0.053670056908, -0.039737675589, 0]],
                                ['H', [-0.028413670411,  0.928922556351, 0]],
                                ['H', [0.880196420813,  -0.298256807934, 0]]]}
    geometry = geometry_dict[molecule_name]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    columns = ['Molecule', 'Scan','Electrons', 'Orbitals', 'Qubits', 'Qubit op', 'Circ len', 
                'Min energy', 'Time', 'nfev', 'nit']
    results = pandas.DataFrame(columns = columns)
    
    #Finds min value:
    if min_mode:
        calculate_minimum(geometry, basis, multiplicity, charge, molecule_name, results)
    elif scan_mode:
        data_list = get_data(geometry, basis, multiplicity, charge, molecule_name, counts)
        results = scan_scheduler(data_list)
        end = time.time() - start
        numpy.savetxt("./results/VQE_scan_{}_{}_{}.csv".format(
            molecule_name, counts, datetime.datetime.now()), results, delimiter=",")
        print("####################################")
        print("Total time: {} s".format(end))
        print("####################################")
    else:
        print("No mode selected!")
        exit()
    

if __name__ == "__main__":
    main()