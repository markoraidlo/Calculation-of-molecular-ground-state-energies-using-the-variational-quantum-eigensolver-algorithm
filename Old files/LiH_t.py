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
"""
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
"""
###

        
"""
elapsed = time.process_time() - start
#Results:
results_sum = results_sum.real
print(results_sum)
print("Time elapsed finding expectation values: {} s".format(elapsed))
"""


"""
#Imgae plot
plt.matshow(results_sum);
plt.colorbar()
plt.savefig('result_sum.png')
"""

res = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list)), method = 'Nelder-Mead',
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})
"""
res = get_expectation_value(numpy.zeros(len(qubit_operator_list)), simulator, UCCSD, pauli_sum, qubit_map)
"""
print("Minimum single sum: {}".format(res))
#-1.13
end = time.process_time() - start
print("Time: {}".format(end))


"""
molecule_name = "LiH"
basis = 'sto-3g'
multiplicity = 1
charge = 0

result_x = list()
result_y = list()

for length in numpy.linspace(0.3, 3, 40):

    
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

    UCCSD = initial_hartree_fock(electron_count)
    UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
    UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

    hamiltonian = get_measurement_hamiltonian(molecular_data)

    #Simulation
    options = {'t': 64}
    simulator = qsimcirq.QSimSimulator(options)
    cirq.DropEmptyMoments().optimize_circuit(circuit = UCCSD)
    qubit_map = get_qubit_map(qubit_count)

    pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)

    res = minimize(get_expectation_value, x0 = [1, -1], method = 'Nelder-Mead',
                    args = (simulator, UCCSD, pauli_sum, qubit_map),
                    options = {'disp' : True})

    result_x.append(length )
    result_y.append(res.fun )

    print("Length: {}. Minimum: {}".format(length, res.fun))
    #print(res)
            

plt.plot(result_x, result_y)
plt.xlabel("H2 bond length")
plt.savefig('result_LiH.png')
"""