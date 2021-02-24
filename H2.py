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

#Needed for expectation values
expectation_value_results = numpy.zeros((20, 20), dtype = complex)
qubit_map = get_qubit_map(qubit_count)
coefficients, pauli_sum_list = get_measurement_pauli_sums(hamiltonian, qubit_count)

start = time.process_time()
#Both parameters 0 to 2*pi
#i and j for putting result in matrix, later going to be removed
i = 0
for t0 in numpy.linspace(0, 2 * numpy.pi, 20):
    j = 0
    for t1 in numpy.linspace(0, 2 * numpy.pi, 20):
        #ParamResolver
        param_list = [t0, t1]
        resolver_dict = dict()
        for k in range(len(param_list)):
            resolver_dict.update({'t{}'.format(k): param_list[k]})
        resolver = cirq.ParamResolver(resolver_dict)
      
        #Main simulation call
        result = simulator.simulate(UCCSD, resolver)
        state_vector = result.final_state_vector

        #Finding expectation value sum
        expectation_value_sum = 0
        for k in range(len(pauli_sum_list)):
            if pauli_sum_list[k] is not None:
                expectation = pauli_sum_list[k].expectation_from_state_vector(state_vector, qubit_map)
                expectation_value_sum += coefficients[k] * expectation
            else:
                expectation_value_sum += coefficients[k]
        expectation_value_results[i][j] = expectation_value_sum
        j += 1

    i += 1

elapsed = time.process_time() - start
#Results:
print(expectation_value_results)
print("Time elapsed finding expectation values: {} s".format(elapsed))
expectation_value_results = expectation_value_results.real

#Imgae plot
plt.matshow(expectation_value_results);
plt.colorbar()
plt.savefig('result.png')