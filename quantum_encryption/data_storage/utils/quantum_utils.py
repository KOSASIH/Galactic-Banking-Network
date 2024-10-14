# quantum_utils.py

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel
from qiskit.tools.monitor import job_monitor

def generate_random_bits(length):
    """
    Generate random bits.

    Parameters:
    - length: Length of the random bits.

    Returns:
    - random_bits: Random bits.
    """
    return np.random.randint(0, 2, length)

def generate_random_basis(length):
    """
    Generate random basis.

    Parameters:
    - length: Length of the random basis.

    Returns:
    - random_basis: Random basis.
    """
    return np.random.randint(0, 2, length)

def simulate_quantum_circuit(circuit, shots=1024):
    """
    Simulate a quantum circuit.

    Parameters:
    - circuit: Quantum circuit.
    - shots: Number of shots.

    Returns:
    - result: Result of the simulation.
    """
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=shots)
    job_monitor(job)
    result = job.result()
    return result

def add_noise_to_circuit(circuit, error_rate=0.01):
    """
    Add noise to a quantum circuit.

    Parameters:
    - circuit: Quantum circuit.
    - error_rate: Error rate of the quantum channel.

    Returns:
    - noisy_circuit: Noisy quantum circuit.
    """
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, len(circuit.qubits))
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    noisy_circuit = noise_model.add_noise(circuit)
    return noisy_circuit
