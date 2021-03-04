import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from functions import *

#For results
import matplotlib.pyplot as plt
from scipy.optimize import minimize


molecule_name = "H2"
basis = 'sto-3g'
multiplicity = 1
charge = 0

result_x = list()
result_y = list()

for length in numpy.linspace(0.3, 3, 40):

    geometry = [[ 'H', [ 0, 0, 0]],
                [ 'H', [ 0, 0, length]]]

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
plt.savefig('result_xy.png')