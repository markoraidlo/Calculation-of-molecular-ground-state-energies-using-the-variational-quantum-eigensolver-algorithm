import logging
import time as time

import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import (FermionOperator, MolecularData, bravyi_kitaev,
                         get_fermion_operator, jordan_wigner,
                         uccsd_convert_amplitude_format)
from openfermionpsi4 import run_psi4
from scipy.optimize import minimize, shgo
from sympy import Symbol


def get_qubit_operators(molecular_data):
    """Leiab kvantbittoperaatorid psi4 arvutatud andmetest.

    Args:
        molecular_data (MolecularData): MolecularData, millele on tehtud psi4 arvutused

    Returns:
        list: Kvantbittoperaatorite järjend
    """

    # Üksik ja kaksik ergastuste amplituudid psi4 arvutusest.
    single_amplitudes = molecular_data.ccsd_single_amps
    double_amplitudes = molecular_data.ccsd_double_amps

    if (isinstance(single_amplitudes, numpy.ndarray) or isinstance(double_amplitudes, numpy.ndarray)):
        single_amplitudes_list, double_amplitudes_list = uccsd_convert_amplitude_format( single_amplitudes, double_amplitudes) 
    
    fermion_operator_list = list()
    # Üksik ergastused.
    for (i, j), t_ik in single_amplitudes_list:
        i, j = int(i), int(j)
        ferm_op = FermionOperator(((i, 1), (j, 0)), 1.) - FermionOperator(((j, 1), (i, 0)), 1.)
        fermion_operator_list.append(ferm_op)
        
    # Kaksik ergastused.
    for (i, j, k, l), t_ijkl in double_amplitudes_list:
        i, j, k, l = int(i), int(j), int(k), int(l)
        ferm_op = (FermionOperator(((i, 1), (j, 0), (k, 1), (l, 0)), 1.) - FermionOperator(((l, 1), (k, 0), (j, 1), (i, 0)), 1.))
        fermion_operator_list.append(ferm_op)

    # Jordan-Wigneri teisendus.
    qubit_operator_list = list()
    for fermion_operator in fermion_operator_list:
        qubit_operator_list.append(jordan_wigner(fermion_operator))

    return qubit_operator_list


def create_uccsd(qubit_operator_list, qubit_count, param):
    """Loob kvantbittoperaatorite järjendist UCCSD kvantahela.

    Args:
        qubit_operator_list (list[QubitOperator]): Kvantbittoperaatorite järjend
        qubit_count (int): Kvantbittide arv
        param (String)): Optimiseerimis parameetri tähistus

    Returns:
        Circuit: UCCSD kvantahel
    """

    circuit = cirq.Circuit()
    # Erinevate kvantbittoperaatorite loomine.
    for i in range(len(qubit_operator_list)):

        qubits = cirq.LineQubit.range(qubit_count)
        sub_circuit = cirq.Circuit()
        terms_list = qubit_operator_list[i].terms

        # Tundmatu parameeteri loomine.
        param_string = param + "{}".format(i)
        temp_param = Symbol(param_string)

        # Erinevate eksponentide loomine.
        for term in terms_list:

            # Baasi vahetus.
            moment1 = cirq.Moment()
            moment2 = cirq.Moment()

            for basis in term:
                if basis[1] == 'X':
                    q = qubits[int(basis[0])]
                    moment1 = moment1.with_operation(cirq.H(q))

                if basis[1] == 'Y':
                    q = qubits[int(basis[0])]
                    moment1 = moment1.with_operation(cirq.S(q)**(-1))
                    moment2 = moment2.with_operation(cirq.H(q))
                    

            sub_circuit.append(moment1)
            sub_circuit.append(moment2)

            # Eksponendi kvantahela loomine.
            exponent = cirq.Circuit()
            max_qubit = 0
            for basis in term:
                if basis[0] > max_qubit:
                    max_qubit = basis[0]

            for basis in term:
                if basis[0] < max_qubit:
                    exponent.append(cirq.CNOT(qubits[basis[0]], qubits[max_qubit]),
                                    strategy = cirq.InsertStrategy.NEW)     

            rotate_z = cirq.rz(-2 * terms_list[term].imag *  temp_param)

            exponent_reverse = exponent**(-1)
            exponent.append([rotate_z(qubits[max_qubit]), exponent_reverse],
                                strategy = cirq.InsertStrategy.NEW)

            sub_circuit.append(exponent)

            # Baasi vahetus tagasi.
            moment3 = cirq.Moment()
            moment4 = cirq.Moment()

            for basis in term:
                if basis[1] == 'X':
                    q = qubits[int(basis[0])]
                    moment3 = moment3.with_operation(cirq.H(q))
                
                if basis[1] == 'Y':
                    q = qubits[int(basis[0])]
                    moment3 = moment3.with_operation(cirq.H(q))
                    moment4 = moment4.with_operation(cirq.S(q))
                    
                
            sub_circuit.append(moment3)
            sub_circuit.append(moment4)

        circuit.append(sub_circuit, strategy = cirq.InsertStrategy.NEW)
    
    return circuit


def initial_hartree_fock(electron_count, qubit_count):
    """Loob Hartree-Focki põhi oleku |11..100...0>.

    Args:
        electron_count (int): Elektronide arv molekulis
        qubit_count (int): Kvantbittide arv kvantahelas

    Returns:
        Circuit: Hartree-Focki põhiseisund UCCSD ahela algusesse
    """

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(qubit_count)

    i = 0
    while i < electron_count:
        circuit.append(cirq.X(qubits[i]), strategy = cirq.InsertStrategy.INLINE)
        i += 1

    return circuit


def get_measurement_hamiltonian(molecular_data):
    """Loob mõõtmis hamiltoniaani.

    Args:
        molecular_data (MolecularData): MolecularData koos psi4 arvutustega

    Returns:
        QubitOperator: Mõõtmis hamiltoniaan
    """

    one_body_integrals = molecular_data.one_body_integrals
    two_body_integrals = molecular_data.two_body_integrals
    orbitals = molecular_data.canonical_orbitals
    molecular_data.save()
    molecule_qubit_hamiltonian = jordan_wigner(get_fermion_operator(molecular_data.get_molecular_hamiltonian()))
    
    return molecule_qubit_hamiltonian


def get_measurement_pauli_sum(molecule_qubit_hamiltonian, qubit_count):
    """Loob Pauli summa mõõtmis hamiltoniaanist.

    Args:
        molecule_qubit_hamiltonian (QubitOperator): Mõõtmis hamiltoniaan
        qubit_count (int): Kvantbittide arv

    Returns:
        PauliSum: Pauli summa
    """
    pauli_sum = None
    qubits = cirq.LineQubit.range(qubit_count)
    terms = molecule_qubit_hamiltonian.terms
    
    # Erinevad hamiltoniaani osad.
    for term in terms:
        pauli_string = None

        # Identiteedi operaatori erand.
        if len(term) == 0:
            if pauli_sum is  None:
                pauli_sum = cirq.I(qubits[0]) * terms[term].real
            else:
                pauli_sum += cirq.I(qubits) * terms[term].real
            
            continue
        
        # Kvantbittoperaatorite lisamine Pauli summasse.
        for basis in term:
            if basis[1] == 'X':
                # Kui pauli_string-i pole lisatud väravat.
                if pauli_string is None:
                    pauli_string = cirq.X(qubits[basis[0]])
                # Kui värav on juba olemas.
                else:
                    pauli_string = pauli_string * cirq.X(qubits[basis[0]])

            elif basis[1] == 'Y':
                if pauli_string is None:
                    pauli_string = cirq.Y(qubits[basis[0]])
                else:
                    pauli_string = pauli_string * cirq.Y(qubits[basis[0]])
                
            elif basis[1] == 'Z':
                if pauli_string is None:
                    pauli_string = cirq.Z(qubits[basis[0]])
                else:
                    pauli_string = pauli_string * cirq.Z(qubits[basis[0]])

        # Kõik kokku liita üheks Pauli summaks.
        if pauli_sum is not None:
            pauli_sum += pauli_string * terms[term].real 
        else:
            pauli_sum =  pauli_string * terms[term].real 

    return pauli_sum


def get_qubit_map(qubit_count):
    """Loob expectation_from_state_vector() funktsiooni jaoks qubit_map-i.
    kvantbitt[j] -> arv j.

    Args:
        qubit_count (int): Kvantbittide arv

    Returns:
        [dict]: {LineQubit: int} Qubitmap
    """
    qubit_map = dict()
    qubits = cirq.LineQubit.range(qubit_count)
    i = 0

    for qubit in qubits:
        qubit_map[qubit] = i
        i += 1

    return qubit_map


def get_expectation_value(x, *args):
    """Kasutada koos scipy.minimize optimiseerijaga.
    Leiab ahelale, hamiltoniaanile ja parameetritele vastava keskväärtuse.

    Args:
        x (list): Parameetrite järjend
        *args (): simulator, UCCSD, pauli_sum, qubit_map

    Returns:
        float: Keskväärtus
    """
    start_time = time.time()

    assert len(args) == 4
    simulator, uccsd, pauli_sum, qubit_map = args

    # Parameetrite lisamine kvantahelale.
    resolver_dict = dict()
    for k in range(len(x)):
        resolver_dict.update({'t{}'.format(k): x[k]})
    resolver = cirq.ParamResolver(resolver_dict)

    # Qsim simuleerib kvantahela.
    result = simulator.simulate(uccsd, resolver)
    state_vector = result.final_state_vector

    norm = numpy.linalg.norm(state_vector)
    state_vector = state_vector / norm

    # Keskväärtuse leiab kvantahela olekuvektori ja Pauli summaga.
    expectation_value = pauli_sum.expectation_from_state_vector(state_vector, qubit_map)

    elapsed_time = time.time() - start_time
    # Aja logimine kui on vaja.
    #logging.info("get_expectation_value: time - %s s; value - %s", elapsed_time, expectation_value.real)

    return expectation_value.real


def single_point_calculation(values):
    """Arvutab ühe punkti miinimum energia väärtuse

    Args:
        values (List): [MolecularData, faili_nimi]

    Returns:
        List: Leitud väärtused
    """
    molecular_data = values[0]
    file_name = values[1]
    length = molecular_data.description
    logging.info("Starting expectation value calculations for length %s.", length)
    
    electron_count = molecular_data.n_electrons
    qubit_count = molecular_data.n_qubits
    
    # Kvantahela loomine.
    qubit_operator_list = get_qubit_operators(molecular_data)
    uccsd = initial_hartree_fock(electron_count, qubit_count)
    unitary = create_uccsd(qubit_operator_list, qubit_count, 't')
    uccsd.append(unitary, strategy = cirq.InsertStrategy.NEW)

    # Mõõtmis hamiltoniaani loomine.
    hamiltonian = get_measurement_hamiltonian(molecular_data)
    qubit_map = get_qubit_map(qubit_count)
    pauli_sum = get_measurement_pauli_sum(hamiltonian, qubit_count)
    
    # Simulaatori seaded.
    options = {'t': 6}
    simulator = qsimcirq.QSimSimulator(options)
    cirq.DropEmptyMoments().optimize_circuit(circuit = uccsd)

    start_time = time.time()
    
    # Miinimumi leidmine.
    optimize_result = minimize(get_expectation_value, x0 = numpy.zeros(len(qubit_operator_list))
                        , method = 'Nelder-Mead',
                        args = (simulator, uccsd, pauli_sum, qubit_map),
                        options = {'disp' : True, 'ftol': 1e-4, 
                        'maxiter': 100000, 'maxfev': 100000})

    elapsed_time = time.time() - start_time
    logging.info(optimize_result)
    logging.info("Elapsed time: %s", elapsed_time)

    energy_min = optimize_result.fun
    nfev = optimize_result.nfev
    nit = optimize_result.nit

    # Tulemuste salvestamine faili.
    file = open(file_name, "a")
    file.write("{}, {}, {}, {}, {}, \n".format(energy_min, nfev, 
                                            nit, elapsed_time, 
                                            length))
    file.close()
    logging.info("Result at %s saved.", length)

    return energy_min, nfev, nit, elapsed_time
