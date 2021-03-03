import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
import sympy as sympy
from openfermion import (FermionOperator, MolecularData, bravyi_kitaev,
                         get_fermion_operator, jordan_wigner,
                         uccsd_convert_amplitude_format)
from openfermionpsi4 import run_psi4


def get_qubit_operators(molecular_data):
    """Finds qubit operators for quantum circuit unitary.

    Args:
        molecular_data (MolecularData): MolecularData with psi4 calculation

    Returns:
        list: List of qubit operaors
    """

    # GET the single and double amplitudes
    single_amplitudes = molecular_data.ccsd_single_amps
    double_amplitudes = molecular_data.ccsd_double_amps

    # Convert amplitudes into lists
    if (isinstance(single_amplitudes, numpy.ndarray) or isinstance(double_amplitudes, numpy.ndarray)):
        single_amplitudes_list, double_amplitudes_list = uccsd_convert_amplitude_format( single_amplitudes, double_amplitudes)
    
    fermion_operator_list = list()
    #Single excitations
    for (i, j), t_ik in single_amplitudes_list:
        i, j = int(i), int(j)
        ferm_op = FermionOperator(((i, 1), (j, 0)), 1.) - FermionOperator(((j, 1), (i, 0)), 1.)
        fermion_operator_list.append(ferm_op)
        
    #Double excitations
    for (i, j, k, l), t_ijkl in double_amplitudes_list:
        i, j, k, l = int(i), int(j), int(k), int(l)
        ferm_op = (FermionOperator(((i, 1), (j, 0), (k, 1), (l, 0)), 1.) - FermionOperator(((l, 1), (k, 0), (j, 1), (i, 0)), 1.))
        fermion_operator_list.append(ferm_op)
        
    #Jordan-Wigner transform
    quantum_operator_list = list()
    for fermion_operator in fermion_operator_list:
        quantum_operator_list.append(jordan_wigner(fermion_operator))
    
    return quantum_operator_list


def create_UCCSD(qubit_operator_list, qubit_count, param):
    """Converts list of qubit operators into a UCCSD circuit

    Args:
        qubit_operator_list (list[QubitOperator]): List of qubit operators
        qubit_count (int): Number of qubits needed for circuit
        param (String)): Optimization parameter name

    Returns:
        Circuit: Circuit of UCCSD
    """

    circuit = cirq.Circuit()

    #Different parameter chains
    for i in range(len(qubit_operator_list)):

        qubits = cirq.LineQubit.range(qubit_count)
        sub_circuit = cirq.Circuit()

        terms_list = qubit_operator_list[i].terms

        #Tundmatu parameeter
        param_string = param + "{}".format(i)
        temp_param = sympy.Symbol(param_string)

        #Different exponents
        for term in terms_list:

            #Basis change
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

            #Exponent
            exponent = cirq.Circuit()
            #Find max qubit:
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

            #Revert basis change
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


def initial_hartree_fock(electron_count):
    """Creates circuit for initial Hartree-Fock state.

    Args:
        electron_count (int): Number of electrons in molecule

    Returns:
        Circuit: Start of the UCCSD circuit
    """

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)

    i = 0
    while i < electron_count:
        circuit.append(cirq.X(qubits[i]), strategy = cirq.InsertStrategy.INLINE)
        i += 1

    return circuit


def get_measurement_hamiltonian(molecular_data):
    """Creates Hamiltonian for VQE measurements

    Args:
        molecular_data (MolecularData): MolecularData with psi4 calculation

    Returns:
        QubitOperator: Hamiltonian
    """

    one_body_integrals = molecular_data.one_body_integrals
    two_body_integrals = molecular_data.two_body_integrals
    orbitals = molecular_data.canonical_orbitals
    molecular_data.save()
    molecule_qubit_hamiltonian = jordan_wigner(get_fermion_operator(molecular_data.get_molecular_hamiltonian()))
    
    return molecule_qubit_hamiltonian


def get_measurement_pauli_sum(molecule_qubit_hamiltonian, qubit_count):
    """Creates one single PauliSum from measurement Hamiltonian.

    Args:
        molecule_qubit_hamiltonian (QubitOperator): Hamiltonian
        qubit_count (int): Number of qubits

    Returns:
        PauliSum: Single PauliSum
    """
    pauli_sum = None
    qubits = cirq.LineQubit.range(qubit_count)
    terms = molecule_qubit_hamiltonian.terms
    
    #Different parts in Hamiltonian
    for term in terms:
        #Empty string
        pauli_string = None

        #I operator
        if len(term) == 0:
            if pauli_sum is  None:
                pauli_sum = cirq.I(qubits[0]) * terms[term].real
            else:
                pauli_sum += cirq.I(qubits) * terms[term].real
            
            continue
        
        for basis in term:
            if basis[1] == 'X':
                #If pauli_string doesnt contain a gate
                if pauli_string is None:
                    pauli_string = cirq.X(qubits[basis[0]])
                #If it does
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


        if pauli_sum is not None:
            pauli_sum += pauli_string * terms[term].real 
        else:
            pauli_sum =  pauli_string * terms[term].real 

    return pauli_sum


def get_qubit_map(qubit_count):
    """Creates qubit map for expectation_from_state_vector() function.
    Sets qubit[j] with integer j.

    Args:
        qubit_count (int): Number of qubtis needed to map

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
    """For use with scipy minimize. 

    Args:
        x (list): List of parameters for circuit
        *args (): simulator, UCCSD, pauli_sum, qubit_map

    Returns:
        float: Expectation value.
    """
    assert len(args) == 4
    simulator, UCCSD, pauli_sum, qubit_map = args

    resolver_dict = dict()
    for k in range(len(x)):
        resolver_dict.update({'t{}'.format(k): x[k]})

    resolver = cirq.ParamResolver(resolver_dict)

    #Main simulation call
    result = simulator.simulate(UCCSD, resolver)
    state_vector = result.final_state_vector

    expectation_value = pauli_sum.expectation_from_state_vector(state_vector, qubit_map)

    return expectation_value.real


#######################################
#Asjad mida pole enam vaja: 
#######################################

def get_measurement_circuits(molecule_qubit_hamiltonian, qubit_count):
    """Creates list of measurement circuits from molecule hamiltonian for VQE

    Args:
        molecule_qubit_hamiltonian (QubitOperator): Hamiltonian
        qubit_count (int): Number of qubits

    Returns:
        list: List of coefficients
        list: List of circuits
    """

    coefficients = list()
    circuit_list = list()

    terms = molecule_qubit_hamiltonian.terms

    #Creating circuits
    for term in terms:
        coefficients.append(terms[term])

        #Keskväärtus I-st
        if len(term) == 0:
            circuit_list.append(None)
            continue
        
        qubits = cirq.LineQubit.range(qubit_count)
        circuit = cirq.Circuit()
        
        moment1 = cirq.Moment()
        moment2 = cirq.Moment()
        measure_meoment = cirq.Moment()
        
        for basis in term:
            if basis[1] == 'X':
                q = qubits[basis[0]]
                moment1 = moment1.with_operation(cirq.H(q))
                measure_meoment = measure_meoment.with_operation(cirq.measure(q))
            
            elif basis[1] == 'Y':
                q = qubits[basis[0]]
                moment1 = moment1.with_operation(cirq.S(q)**(-1))
                moment2 = moment2.with_operation(cirq.H(q))
                measure_meoment = measure_meoment.with_operation(cirq.measure(q))
                
            elif basis[1] == 'Z':
                q = qubits[int(basis[0])]
                measure_meoment = measure_meoment.with_operation(cirq.measure(q))
        
        circuit.append(moment1)
        circuit.append(moment2)
        circuit.append(measure_meoment)

        circuit_list.append(circuit)
    
    return coefficients, circuit_list


def get_measurement_pauli_strings(molecule_qubit_hamiltonian, qubit_count):
    """Creates PauliStrings from measurement Hamiltonian.

    Args:
        molecule_qubit_hamiltonian (QubitOperator): Hamiltonian
        qubit_count (int): Number of qubits

    Returns:
        [list(int)]: List of coefficents.
        [list(PauliString)]: List of PauliStrings.
    """
    coefficients = list()
    pauli_string_list = list()
    qubits = cirq.LineQubit.range(qubit_count)
    terms = molecule_qubit_hamiltonian.terms
    
    #Different parts in Hamiltonian
    for term in terms:
        #Empty string
        pauli_string = None

        #No operator
        if len(term) == 0:
            pauli_string_list.append(None)
            coefficients.append(terms[term])
            continue
        
        for basis in term:
            if basis[1] == 'X':
                #If pauli_string doesnt contain a gate
                if pauli_string is None:
                    pauli_string = cirq.X(qubits[basis[0]])
                #If it does
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

        pauli_string_list.append(pauli_string)
        coefficients.append(terms[term])

    return coefficients, pauli_string_list

#######################################
#######################################
#######################################