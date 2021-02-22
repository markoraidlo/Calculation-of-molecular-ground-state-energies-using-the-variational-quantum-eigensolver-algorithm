import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
import sympy as sympy
from openfermion import (FermionOperator, MolecularData, bravyi_kitaev,
                         get_fermion_operator, jordan_wigner,
                         uccsd_convert_amplitude_format)
from openfermionpsi4 import run_psi4

from functions import *

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
coefficients, measurement_circuits = get_measurement_circuits(hamiltonian, qubit_count)

print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
print("Qubit operator count: {}".format(len(qubit_operator_list)))
print("Number of measurement circuits: {}".format(len(measurement_circuits)))
print(UCCSD.to_text_diagram(transpose=True))

#Lõpp Hamiltoniaani väärtus
result_value = 0

#Generic ParamResolver
param_list = [1, 2]
resolver_dict = dict()
for i in range(len(param_list)):
    resolver_dict.update({'t{}'.format(i): param_list[i]})
resolver = cirq.ParamResolver(resolver_dict)

simulator = qsimcirq.QSimSimulator()

for i in range(len(coefficients)):
    if measurement_circuits[i] is None:
        result_value += coefficients[i]
    else:
        simulation_circuit = UCCSD + measurement_circuits[i]
        
        #Remove empty moments
        cirq.DropEmptyMoments().optimize_circuit(circuit = simulation_circuit)

        result = simulator.simulate(simulation_circuit, resolver)
        state_vector = result.final_state_vector
        #print(state_vector)
        #TODO: Toore skänn parameetrite jaoks 0 - 2pi
        #TODO:




