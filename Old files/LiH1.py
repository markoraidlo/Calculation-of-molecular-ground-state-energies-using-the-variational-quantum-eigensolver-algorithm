import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from functions import *

#For results
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time as time


molecule_name = "LiH"
geometry= [['Li', [0, 0, 0]] ,
            ['H', [0, 0, 1.5949]]]
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

bounds = list()
for i in range(len(qubit_operator_list)):
    bounds.append([-numpy.pi, numpy.pi])

res = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                    , method = 'L-BFGS-B', bounds = bounds,
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})


print("Minimum single sum: {}".format(res))
#-1.13
end = time.process_time() - start
print("Time: {}".format(end))

#Result save kood:

#Save plot

#Save min

#Save data.