import sys
import time as time

import cirq as cirq
import numpy as numpy
import pandas as pandas
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4
from scipy.optimize import minimize

from functions import *

#TODO: See fail ilusaks
#calc.py molekul / min or skan / count

def main():
    """
    -min molecule \n
    -scan counts molecule \n
    -min -scan counts molecule \n
    Molecules: H2, LiH, BeH2, H2O
    """
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
        molecular_data = MolecularData(geometry, basis, multiplicity,
            charge, filename = './data/{}_min_molecule.data'.format(molecule_name))

        molecular_data = run_psi4(molecular_data,
                                run_scf=True,
                                run_mp2=True,
                                run_cisd=True,
                                run_ccsd=True,
                                run_fci=True)

        qubit_operator_list = get_qubit_operators(molecular_data)

        electron_count = molecular_data.n_electrons
        qubit_count = molecular_data.n_qubits
        orbital_count = molecular_data.n_orbitals
        qubit_op_count = len(qubit_operator_list)

        UCCSD = initial_hartree_fock(electron_count, qubit_count)
        UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
        UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

        hamiltonian = get_measurement_hamiltonian(molecular_data)


        ############ Molecule output
        print(molecule_name)
        print("Electron count: {}".format(electron_count))
        print("Qubit count: {}".format(qubit_count))
        print("Orbital count: {}".format(orbital_count))
        print("Qubit operator count: {}".format(qubit_op_count))
        print("Length of UCCSD circuit: {}".format(len(UCCSD)))
        #print(UCCSD.to_text_diagram(transpose=True))


        ############ Simulation
        options = {'t': 64}
        simulator = qsimcirq.QSimSimulator(options)
        cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

        qubit_map = get_qubit_map(qubit_count)
        pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

        bounds = list()
        for i in range(len(qubit_operator_list)):
            bounds.append([-numpy.pi, numpy.pi])

        start = time.process_time()

        optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                            , method = 'TNC', bounds = bounds,
                            args = (simulator, UCCSD, pauli_sum, qubit_map),
                            options = {'disp' : True})

        end_time = time.process_time() - start


        energy_min = optimize_result.fun
        nfev = optimize_result.nfev
        nit = optimize_result.nit
        print("Minimum energy: {}".format(energy_min))
        print("Time elapsed: {} s".format(end_time))
        print("Number of evaluations of the objective function: {}".format(nfev))
        print("Number of iterations performed by the optimizer: {}".format(nit))
        print("#####################################################################")

        ############ Data save
        results.loc[len(results)] = [molecule_name, " ",electron_count, orbital_count, qubit_count, qubit_op_count, 
                        len(UCCSD), energy_min, end_time, nfev, nit]
        results.to_csv("VQE_min_{}.csv".format(molecule_name))

    #Scans for different bond lengths
    if scan_mode:
        #H2, LiH scan
        #BeH2 scan
        #H2O scan 2
        pass
        

if __name__ == "__main__":
    main()



#For all:


########################################################################
###         H2 min
########################################################################
"""
############ Molecular data and prep
molecule_name = "H2"
geometry = [[ 'H', [ 0, 0, 0]],
            [ 'H', [ 0, 0, 0.74]]]

molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_scf=True,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

qubit_operator_list = get_qubit_operators(molecular_data)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits
orbital_count = molecular_data.n_orbitals
qubit_op_count = len(qubit_operator_list)

UCCSD = initial_hartree_fock(electron_count, qubit_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)


############ Molecule output
print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Orbital count: {}".format(orbital_count))
print("Qubit operator count: {}".format(qubit_op_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
#print(UCCSD.to_text_diagram(transpose=True))


############ Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

qubit_map = get_qubit_map(qubit_count)
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

bounds = list()
for i in range(len(qubit_operator_list)):
    bounds.append([-numpy.pi, numpy.pi])

start = time.process_time()

optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})

end_time = time.process_time() - start


energy_min = optimize_result.fun
nfev = optimize_result.nfev
nit = optimize_result.nit
print("Minimum energy: {}".format(energy_min))
print("Time elapsed: {} s".format(end_time))
print("Number of evaluations of the objective function: {}".format(nfev))
print("Number of iterations performed by the optimizer: {}".format(nit))
print("#####################################################################")

############ Data save
results.loc[len(results)] = [molecule_name, electron_count, orbital_count, qubit_count, qubit_op_count, 
                len(UCCSD), energy_min, end_time, nfev, nit]
results.to_csv("VQE_results.csv")


########################################################################
###         LiH min
########################################################################

############ Molecular data and prep
molecule_name = "LiH"
geometry= [['Li', [0, 0, 0]] ,
            ['H', [0, 0, 1.5949]]]

molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_scf=True,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

qubit_operator_list = get_qubit_operators(molecular_data)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits
orbital_count = molecular_data.n_orbitals
qubit_op_count = len(qubit_operator_list)

UCCSD = initial_hartree_fock(electron_count, qubit_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)


############ Molecule output
print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Orbital count: {}".format(orbital_count))
print("Qubit operator count: {}".format(qubit_op_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
#print(UCCSD.to_text_diagram(transpose=True))


############ Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

qubit_map = get_qubit_map(qubit_count)
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

bounds = list()
for i in range(len(qubit_operator_list)):
    bounds.append([-numpy.pi, numpy.pi])

start = time.process_time()

optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})

end_time = time.process_time() - start


energy_min = optimize_result.fun
nfev = optimize_result.nfev
nit = optimize_result.nit
print("Minimum energy: {}".format(energy_min))
print("Time elapsed: {} s".format(end_time))
print("Number of evaluations of the objective function: {}".format(nfev))
print("Number of iterations performed by the optimizer: {}".format(nit))
print("#####################################################################")

############ Data save
results.loc[len(results)] = [molecule_name, electron_count, orbital_count, qubit_count, qubit_op_count, 
                len(UCCSD), energy_min, end_time, nfev, nit]
results.to_csv("VQE_results.csv")


########################################################################
###         H2O min
########################################################################

############ Molecular data and prep
molecule_name = "H2O"
geometry = [['O', [-0.053670056908, -0.039737675589, 0]],
                    ['H', [-0.028413670411,  0.928922556351, 0]],
                    ['H', [0.880196420813,  -0.298256807934, 0]]]


molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_scf=True,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

qubit_operator_list = get_qubit_operators(molecular_data)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits
orbital_count = molecular_data.n_orbitals
qubit_op_count = len(qubit_operator_list)

UCCSD = initial_hartree_fock(electron_count, qubit_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)


############ Molecule output
print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Orbital count: {}".format(orbital_count))
print("Qubit operator count: {}".format(qubit_op_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
#print(UCCSD.to_text_diagram(transpose=True))


############ Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

qubit_map = get_qubit_map(qubit_count)
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

bounds = list()
for i in range(len(qubit_operator_list)):
    bounds.append([-numpy.pi, numpy.pi])

start = time.process_time()

optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})

end_time = time.process_time() - start


energy_min = optimize_result.fun
nfev = optimize_result.nfev
nit = optimize_result.nit
print("Minimum energy: {}".format(energy_min))
print("Time elapsed: {} s".format(end_time))
print("Number of evaluations of the objective function: {}".format(nfev))
print("Number of iterations performed by the optimizer: {}".format(nit))
print("#####################################################################")

############ Data save
results.loc[len(results)] = [molecule_name, electron_count, orbital_count, qubit_count, qubit_op_count, 
                len(UCCSD), energy_min, end_time, nfev, nit]
results.to_csv("VQE_results.csv")
                        


########################################################################
###         BeH2 min
########################################################################

############ Molecular data and prep
molecule_name = "BeH2"
geometry= [['Be', [ 0, 0, 0 ]],
                ['H', [ 0, 0, 1.3264]],
                ['H', [ 0, 0, -1.3264]]]


molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_scf=True,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

qubit_operator_list = get_qubit_operators(molecular_data)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits
orbital_count = molecular_data.n_orbitals
qubit_op_count = len(qubit_operator_list)

UCCSD = initial_hartree_fock(electron_count, qubit_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)


############ Molecule output
print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Orbital count: {}".format(orbital_count))
print("Qubit operator count: {}".format(qubit_op_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
#print(UCCSD.to_text_diagram(transpose=True))


############ Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

qubit_map = get_qubit_map(qubit_count)
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

bounds = list()
for i in range(len(qubit_operator_list)):
    bounds.append([-numpy.pi, numpy.pi])

start = time.process_time()

optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})

end_time = time.process_time() - start


energy_min = optimize_result.fun
nfev = optimize_result.nfev
nit = optimize_result.nit
print("Minimum energy: {}".format(energy_min))
print("Time elapsed: {} s".format(end_time))
print("Number of evaluations of the objective function: {}".format(nfev))
print("Number of iterations performed by the optimizer: {}".format(nit))
print("#####################################################################")

############ Data save
results.loc[len(results)] = [molecule_name, electron_count, orbital_count, qubit_count, qubit_op_count, 
                len(UCCSD), energy_min, end_time, nfev, nit]
results.to_csv("VQE_results.csv")

"""
"""
########################################################################
###         H2 scan
########################################################################

result_length = list()
result_energy = list()
molecule_name = "H2"
for length in numpy.linspace(0.2, 3, 60):

    while True:
        geometry = [[ 'H', [ 0, 0, 0]],
                    [ 'H', [ 0, 0, length]]]

        print(length)
        
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
            break
        except Exception as exc:
            print(exc)
            length += 0.000000000001
            print("########################################")
            print("ERROR FIX: length = {}".format(length))
            print("########################################")
            pass

    electron_count = molecular_data.n_electrons
    qubit_count = molecular_data.n_qubits

    UCCSD = initial_hartree_fock(electron_count, qubit_count)
    UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
    UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

    hamiltonian = get_measurement_hamiltonian(molecular_data)

    #Simulation
    options = {'t': 64}
    simulator = qsimcirq.QSimSimulator(options)
    cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)
    qubit_map = get_qubit_map(qubit_count)

    pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

    bounds = list()
    for i in range(len(qubit_operator_list)):
        bounds.append([-numpy.pi, numpy.pi])

    res = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map))
    print(res)

    result_length.append(length)
    result_energy.append(res.fun)

print("###########################################")
print("H2 Scan complete.")
print("###########################################")       

dict_h2 = {'length': result_length, 'energy': result_energy}
df = pandas.DataFrame(dict_h2)
df.to_csv('H2_scan.csv')



########################################################################
###         LiH scan
########################################################################

result_length = list()
result_energy = list()

for length in numpy.linspace(0.2, 3, 50):

    molecule_name = "LiH"
    geometry= [['Li', [0, 0, 0]] ,
            ['H', [0, 0, length]]]

    molecular_data = MolecularData(geometry, basis, multiplicity,
        charge, filename = './data/{}_{}_molecule.data'.format(molecule_name, length))

    molecular_data = run_psi4(molecular_data,
                            run_scf=True,
                            run_mp2=True,
                            run_cisd=True,
                            run_ccsd=True,
                            run_fci=True)

    #UnboundLocalError: local variable 'single_amplitudes_list' referenced before assignment
    while True:
        try:
            qubit_operator_list = get_qubit_operators(molecular_data)
            break
        except UnboundLocalError as error:
            print(error)
            pass
        except Exception as exc:
            print(exc)
            pass

    electron_count = molecular_data.n_electrons
    qubit_count = molecular_data.n_qubits

    UCCSD = initial_hartree_fock(electron_count, qubit_count)
    UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
    UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

    hamiltonian = get_measurement_hamiltonian(molecular_data)

    #Simulation
    options = {'t': 64}
    simulator = qsimcirq.QSimSimulator(options)
    cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)
    qubit_map = get_qubit_map(qubit_count)

    pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

    bounds = list()
    for i in range(len(qubit_operator_list)):
        bounds.append([-numpy.pi, numpy.pi])

    res = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'TNC', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map))

    print(res)

    result_length.append(length)
    result_energy.append(res.fun)

print("###########################################")
print("LiH Scan complete.")
print("###########################################")       

dict_LiH = {'length': result_length, 'energy': result_energy}
df = pandas.DataFrame(dict_LiH)
df.to_csv('LiH_scan.csv')

########################################################################
###         BeH2 scan
########################################################################

shape_1, shape_2 = 10, 10
qubit_map = get_qubit_map(qubit_count)
BeH2_scan = numpy.zeros((shape_1, shape_2), dtype = complex) 

for i in range(shape_1):
    for j in range(shape_1):

        molecule_name = "BeH2"
        geometry= [['Be', [ 0, 0, 0 ]],
                        ['H', [ 0, 0, (0.3 + i * 0.28)]],
                        ['H', [ 0, 0, -(0.3 + i * 0.28)]]]
                        #linspace(0.2, 3, 50)

        molecular_data = MolecularData(geometry, basis, multiplicity,
            charge, filename = './data/{}_{}_molecule.data'.format(molecule_name, length))

        molecular_data = run_psi4(molecular_data,
                                run_scf=True,
                                run_mp2=True,
                                run_cisd=True,
                                run_ccsd=True,
                                run_fci=True)

        #UnboundLocalError: local variable 'single_amplitudes_list' referenced before assignment
        while True:
            try:
                qubit_operator_list = get_qubit_operators(molecular_data)
                break
            except UnboundLocalError as error:
                print(error)
                pass
            except Exception as exc:
                print(exc)
                pass

        electron_count = molecular_data.n_electrons
        qubit_count = molecular_data.n_qubits

        UCCSD = initial_hartree_fock(electron_count, qubit_count)
        UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
        UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

        hamiltonian = get_measurement_hamiltonian(molecular_data)

        #Simulation
        options = {'t': 64}
        simulator = qsimcirq.QSimSimulator(options)
        cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)
        qubit_map = get_qubit_map(qubit_count)

        pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

        bounds = list()
        for i in range(len(qubit_operator_list)):
            bounds.append([-numpy.pi, numpy.pi])

        res = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                        , method = 'TNC', bounds = bounds,
                        args = (simulator, UCCSD, pauli_sum, qubit_map))

        print(res)

        BeH2_scan[i][j] = res.fun

print("###########################################")
print("BeH2 Scan complete.")
print("###########################################")       
BeH2_scan = BeH2_scan.real
numpy.savetxt('BeH2_scan.csv', BeH2_scan, delimiter=',')

"""
