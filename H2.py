import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from functions import *

#For results
import matplotlib.pyplot as plt
import time

molecule_name = "H2"
geometry = [[ 'H', [ 0, 0, 0]],
            [ 'H', [ 0, 0, 0.74]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0

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

UCCSD = initial_hartree_fock(electron_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)
#Võib olla pole enam vaja
#coefficients, measurement_circuits = get_measurement_circuits(hamiltonian, qubit_count)

print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
print("Qubit operator count: {}".format(len(qubit_operator_list)))
#print("Number of measurement circuits: {}".format(len(measurement_circuits)))
#print(UCCSD.to_text_diagram(transpose=True))

#Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

shape_1, shape_2 = 20, 20
qubit_map = get_qubit_map(qubit_count)

################################################################################################
###Test one single PauliSum:
################################################################################################
results_sum = numpy.zeros((shape_1, shape_2), dtype = complex) 
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)
start = time.process_time()

for i in range(shape_1):
    for j in range(shape_1):
        #ParamResolver
        param_list = [2 * numpy.pi * i / (shape_1 - 1) - numpy.pi,
        2 * numpy.pi * j / (shape_2 - 1)  - numpy.pi]
        resolver_dict = dict()
        for k in range(len(param_list)):
            resolver_dict.update({'t{}'.format(k): param_list[k]})
        resolver = cirq.ParamResolver(resolver_dict)
        
        #Main simulation call
        result = simulator.simulate(UCCSD, resolver)
        state_vector = result.final_state_vector

        #Finding expectation value sum
        expectation_value = pauli_sum.expectation_from_state_vector(state_vector, qubit_map)
        results_sum[i][j] = expectation_value
        

elapsed = time.process_time() - start
#Results:
results_sum = results_sum.real
print(results_sum)
print("Time elapsed finding expectation values: {} s".format(elapsed))

print(hamiltonian)
print(pauli_sum)


#Imgae plot
plt.matshow(results_sum);
plt.colorbar()
plt.savefig('result_sum.png')

print("Minimum single sum: {}".format(results_sum.min()))
#-1.13