import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from functions import *

#For results
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize


#####################################################################
#####################################################################
#####################################################################
molecule_name = "H2"
geometry = [[ 'H', [ 0, 0, 0]],
            [ 'H', [ 0, 0, 0.74]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0

molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

#####################################################################
#####################################################################
#####################################################################

qubit_operator_list = get_qubit_operators(molecular_data)
print(qubit_operator_list)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits

UCCSD = initial_hartree_fock(electron_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)
print(hamiltonian)

#print(UCCSD.to_text_diagram(transpose=True))



#####################################################################
#Simulation 
#####################################################################
options = {'t': 64}
simulator = qsimcirq.QSimSimulator(options)
cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)

qubit_map = get_qubit_map(qubit_count)
pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

print(pauli_sum)

expectation = minimize(get_expectation_value, x0 =[0, 0], 
                args = (simulator, UCCSD, pauli_sum, qubit_map))
#expectation = get_expectation_value([0, 0], simulator, UCCSD, pauli_sum, qubit_map)


#####################################################################
#End
#####################################################################

print(expectation)
