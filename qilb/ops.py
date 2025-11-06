import numpy as np
from typing import Iterable

I = np.eye(2, dtype=np.complex128)

P0 = np.array([
    [1, 0],
    [0, 0]
], dtype=np.complex128)

P1 = np.array([
    [0, 0],
    [0, 1]
], dtype=np.complex128)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=np.complex128)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=np.complex128)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=np.complex128)

H = np.array([
    [1/np.sqrt(2), 1/np.sqrt(2)],
    [1/np.sqrt(2), -1/np.sqrt(2)]
], dtype=np.complex128)

S = np.array([
    [1, 0],
    [0, 1j]
], dtype=np.complex128)

T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
], dtype=np.complex128)

def RX(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=np.complex128)

def RY(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=np.complex128)

def RZ(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=np.complex128)

# Useful for the dense simulation
def build_operator(num_qubits: int, qubit_index: int, U: np.ndarray) -> np.ndarray:
    operators = [I] * num_qubits
    operators[qubit_index] = U

    full_U = operators[0]
    for i in range(1, num_qubits):
        full_U = np.kron(full_U, operators[i])

    return full_U

def build_operators(num_qubits: int, qubits: Iterable[int], operators: Iterable[np.ndarray]) -> np.ndarray:
    ops = [I] * num_qubits
    for i, qubit in enumerate(qubits):
        ops[qubit] = operators[i]

    full_U = ops[0]
    for i in range(1, num_qubits):
        full_U = np.kron(full_U, ops[i])

    return full_U

def repeat_operator(qubits: int, operator: np.ndarray) -> np.ndarray:
    operators = [operator] * qubits

    full_U = operators[0]
    for i in range(1, qubits):
        full_U = np.kron(full_U, operators[i])

    return full_U