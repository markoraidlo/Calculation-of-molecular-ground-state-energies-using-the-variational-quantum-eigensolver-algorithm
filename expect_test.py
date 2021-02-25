import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from functions import *

#Expectation value arvutamise kontroll:

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
#print(UCCSD.to_text_diagram(transpose=True))

#Simulation
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

#ParamResolver
param_list = [1, 2]
resolver_dict = dict()
for k in range(len(param_list)):
    resolver_dict.update({'t{}'.format(k): param_list[k]})
resolver = cirq.ParamResolver(resolver_dict)

#Needed for expectation values
qubit_map = get_qubit_map(qubit_count)
coefficients, pauli_sum_list = get_measurement_pauli_sums(hamiltonian, qubit_count)

#Main simulation call
result = simulator.simulate(UCCSD, resolver)
state_vector = result.final_state_vector


#EXPECTATION VALUE TEST OSA:
#########################################################################
print(state_vector)

#Finding expectation value sum
expectation_value_sum = 0
for k in range(len(pauli_sum_list)):
    if pauli_sum_list[k] is not None:
        #print(pauli_sum_list[k])
        expectation = pauli_sum_list[k].expectation_from_state_vector(state_vector, qubit_map)
        #print(coefficients[k])
        expectation_value_sum += expectation.real
        #print(expectation)
    else:
        expectation_value_sum += coefficients[k].real
    #print(expectation_value_sum)
    
print(expectation_value_sum)
#print(hamiltonian)

coefficients, pauli_sum_list = single_sum(hamiltonian, qubit_count)
#print(hamiltonian)
#print(pauli_sum_list)
#print(pauli_sum_list)
expectation_value = pauli_sum_list.expectation_from_state_vector(state_vector, qubit_map)
print(expectation_value)

