import os
import pprint
import math
import cirq
import numpy as np
import openfermion as of
import stim
import stimcirq
import h5py
from typing import Set, List, Iterable
import warnings
import matplotlib.pyplot as plt
import numpy as np
import sympy

def restrict_to(
    pauli: cirq.PauliString, qubits: Iterable[cirq.Qid]
) -> cirq.PauliString:
    """Returns the Pauli string restricted to the provided qubits.

    Arguments:
        pauli: A Pauli string.
        qubits: A set of qubits.

    Returns:
        The provided Pauli string acting only on the provided qubits.
        Note: This could potentially be empty (identity).
    """
    return cirq.PauliString(p.on(q) for q, p in pauli.items() if q in qubits)


def commutes(pauli1: cirq.PauliString, pauli2: cirq.PauliString, blocks) -> bool:
    """Returns True if pauli1 k-commutes with pauli2, else False.

    Arguments:
        pauli1: A Pauli string.
        pauli2: A Pauli string.
        blocks: The block partitioning.

    """

    for block in blocks:
        if not cirq.commutes(restrict_to(pauli1, block), restrict_to(pauli2, block)):
            return False
    return True


def get_num_qubits(hamiltonian: cirq.PauliSum) -> int:
    return len(hamiltonian.qubits)


def get_terms_ordered_by_abscoeff(ham: cirq.PauliSum) -> List[cirq.PauliString]:
    """Returns the terms of the PauliSum ordered by coefficient absolute value.

    Args:
        ham: A PauliSum.
    Returns:
        a list of PauliStrings sorted by the absolute value of their coefficient.
    """
    return sorted([term for term in ham], key=lambda x: abs(x.coefficient), reverse=True)


def get_si_sets(ham: cirq.PauliSum, k: int = 1) -> List[List[cirq.PauliString]]:
    """Returns grouping from the sorted insertion algorithm [https://quantum-journal.org/papers/q-2021-01-20-385/].

    Args:
        op: The observable to group.
        k: The integer k in k-commutativity.
    """
    
    
    qubits = sorted(set(ham.qubits))
    blocks = compute_blocks(qubits, k)

    commuting_sets = []
    for pstring in get_terms_ordered_by_abscoeff(ham):
        found_commuting_set = False

        for commset in commuting_sets:
            cant_add = False

            for pauli in commset:
                if not commutes(pstring, pauli, blocks):
                    cant_add = True
                    break

            if not cant_add:
                commset.append(pstring)
                found_commuting_set = True
                break

        if not found_commuting_set:
            commuting_sets.append([pstring])

    return commuting_sets


def compute_blocks(qubits, k):
    return [qubits[k * i : k * (i + 1)] for i in range(math.ceil(len(qubits) / k))]


def compute_rhat(groupings):
    r_numerator = 0
    r_denominator = 0
    for group in groupings:
        if isinstance(group, cirq.PauliSum):
            a_ij = sum([term.coefficient for term in group])
            r_numerator += abs(a_ij)
            r_denominator += np.sqrt(abs(a_ij) ** 2)
        else:
            a_ij = np.array([term.coefficient for term in group])
            group_sum = np.sum(np.abs(a_ij))
            group_sum_squares = np.sum(np.abs(a_ij) ** 2)
            r_numerator += group_sum
            r_denominator += np.sqrt(group_sum_squares)
    return (r_numerator / r_denominator) ** 2


def read_openfermion_hdf5(fname_hdf5: str, key: str, optype=of.QubitOperator):
    """
    Read any openfermion operator object from HDF5 file at specified key.
    'optype' is the op class, can be of.QubitOperator or of.FermionOperator.
    """

    with h5py.File(fname_hdf5, 'r', libver='latest') as f:
        op = optype(f[key][()].decode("utf-8"))
    return op


def parse_through_hdf5(func):
    """
    Decorator function that iterates through an HDF5 file and performs
    the action specified by ‘ func ‘ on the internal and leaf nodes in the HDF5 file.
    """

    def wrapper (obj, path = '/', key = None) :
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for ky in obj.keys() :
                func(obj, path, key=ky, leaf = False)
                wrapper(obj = obj[ky], path = path + ky + ',', key = ky)
        elif type (obj) == h5py._hl.dataset.Dataset:
            func(obj, path, key = None, leaf = True)
    return wrapper


def get_hdf5_keys ( fname_hdf5 : str ) :
    """ Get a list of keys to all datasets stored in the HDF5 file .
    Args
    ----
    fname_hdf5 ( str ) : full path where HDF5 file is stored
    """

    all_keys = []
    @parse_through_hdf5
    def action(obj, path = '/', key = None, leaf = False):
        if leaf is True :
            all_keys.append(path)

    with h5py.File(fname_hdf5, 'r') as f:
        action(f['/'])
    return all_keys


def preprocess_hamiltonian(
    hamiltonian: of.QubitOperator,
    drop_term_if = None,
) -> cirq.PauliSum:
    """Drop identity terms from the Hamiltonian and convert to Cirq format.
    """
    if drop_term_if is None:
        drop_term_if = []

    new = cirq.PauliSum()

    for term in hamiltonian.terms:
        add_term = True

        for drop_term in drop_term_if:
            if drop_term(term):
                add_term = False
                break

        if add_term:
            key = " ".join(pauli + str(index) for index, pauli in term)
            new += next(iter(of.transforms.qubit_operator_to_pauli_sum(
                of.QubitOperator(key, hamiltonian.terms.get(term))
            )))
    
    return new


def get_bit(value, bit):
    return value >> bit & 1


def convert_to_stim_strings(group, k, qubits): #takes in 1 k-commuting group from si 
    """Convert the group to Stim strings that can be used to generate
    the tableau.
    
    Args
    ----
        group:
            group of k-commuting Paulis
        k:
            value of k
        qubits:
            qubits hamiltonian acts on
    """
    
    # Compute the blocks of size k
    blocks = compute_blocks(qubits, k)

    
    # Compute the Pauli strings for Stim.
    all_strings = []
    for block in blocks:
        block_strings = []
        #print(f'block: {block}')
        for i, ps in enumerate(group):
            ps = restrict_to(ps, block)
            dps = ps.dense(block)
            ss = dps.__str__().replace("I", "_")
            if any(s in ss for s in ["X", "Y", "Z"]): 
                block_strings.append(ss)
        all_strings.append(block_strings)
    return all_strings  #returns list of n/k blocks of strings, all_strings[0][3] commutes with all_strings[3][3]
    
    

def compute_measurement_circuit_depth(stim_strings):
    """Generate the measurement circuits for every block and compute
    their optimized depth.
    
    Args
    ----
        stim_strings:
            nested list generated with "convert_to_stim_strings"
    
    Returns
    -------
        optimized depth
    """
    all_depths = []
    all_circuits = []
    
    for block_strings in stim_strings:  #block strings is list of commuting blocks
        # Compute tableau and measurement circuit
        print('strings: ', block_strings)
        if not block_strings: 
            continue
        attempt = 0
        result = False

        while not result and attempt < 2**(len(block_strings)+1):
            if len(bin(attempt))-2 > len(block_strings):
                try:
                    stim_tableau = stim.Tableau.from_stabilizers(
                        [stim.PauliString(
                            ('-' if bin(attempt)[-(i+1)] == '0' else '+') + stim_str[1:]) for i, stim_str in enumerate(block_strings)
                        ],
                    allow_redundant=True,
                    allow_underconstrained=True
                    )
                    result = True
                except ValueError:
                    pass
            attempt+=1

        if result:
            stim_circuit = stimcirq.stim_circuit_to_cirq_circuit(
                stim_tableau.to_circuit(method="elimination")
            )
            # Optimize to gate set and compute depth
            opt_circuit = cirq.optimize_for_target_gateset(
                stim_circuit, gateset=cirq.CZTargetGateset()
            )
            final_ckt = cirq.Circuit(opt_circuit.all_operations())
            depth = len(final_ckt)
            all_depths.append(depth)
            all_circuits.append(final_ckt)
        else:
            raise RuntimeWarning('No independent set of stabilizers found.')
        
    return all_depths, all_circuits



def measurement_circuit_depth(groupings, k, qubits):
    """Compute the maximum circuit depth."""
    depths = []
    for group in groupings:
        blocked_stim_strings = convert_to_stim_strings(group, k, qubits)
        blocked_circuit_depths, _ = compute_measurement_circuit_depth(blocked_stim_strings)
        if blocked_circuit_depths:
            depths.append(max(blocked_circuit_depths))
    return max(depths) # Only report the maximum



def diag_circ_from_ham(hamiltonian, k):
    hamiltonian = preprocess_hamiltonian(hamiltonian, drop_term_if=[lambda term: term == ()])
    groups = get_si_sets(hamiltonian, k)
    nqubits = get_num_qubits(hamiltonian)
    qubits = sorted(set(hamiltonian.qubits))
    nterms = len(hamiltonian)
    group_block_circuits = []
    for group in groups:
        blocked_stim_strings = convert_to_stim_strings(group, k, qubits)
        all_depths, all_circuits = compute_measurement_circuit_depth(blocked_stim_strings)
        group_block_circuits.append(all_circuits)

    group_circuits = []
    for group_ckt in group_block_circuits:
        circuit = cirq.Circuit()
        all_transformed_circuits = []
        for j in range(len(group_ckt)):
            qubit_map = {}
            for i in range(0, k):
                qubit_map[cirq.LineQubit(i)] = cirq.LineQubit(i + j * k)
            transformed_ckt = group_ckt[j].transform_qubits(qubit_map=qubit_map)
            all_transformed_circuits.append(transformed_ckt)
        circuit.append(all_transformed_circuits)
        group_circuits.append(circuit)
    return (groups, group_circuits)



