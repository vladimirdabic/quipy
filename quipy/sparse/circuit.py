from __future__ import annotations
from typing import List, Tuple, Callable, Generator
from abc import ABC
from .state import State
from dataclasses import dataclass
from enum import IntEnum, auto
from .. import ops
import numpy as np


class CircuitEntryType(IntEnum):
    OPERATOR = auto()
    GATE = auto()
    COMPOSITE = auto()
    MEASURING_DEVICE = auto()
    CALLBACK = auto()

@dataclass
class CircuitEntry:
    type: CircuitEntryType
    matrix: np.ndarray
    qubits: List[int]
    noise_model: NoiseModel
    callback: Callable[[State, List[int]], None] = None
    terms: List[Tuple[List[np.ndarray], List[int]]] = None

class Circuit:
    entries: List[CircuitEntry]
    num_qubits: int

    def __init__(self, num_qubits: int = 1):
        self.entries = []
        self.num_qubits = num_qubits

    def add_gate(self, gate: np.ndarray, qubit: int = None, qubits: Tuple[int] = None, noise_model: NoiseModel = None):
        """
        Adds a gate to the circuit.

        Parameters
        ----------
        gate : ndarray
            The unitary matrix representing the gate

        qubit : int, optional
            The qubit that the gate is applied to

        qubits : tuple of int, optional
            The qubits that the gate is applied to

        noise_model : NoiseModel, optional
            Optional gate specific noise model

        Raises
        ------
        ValueError
            If the gate or qubit(s) are not specified.
        """
        if qubit is not None:
            self.entries.append(CircuitEntry(CircuitEntryType.GATE, gate, [qubit], noise_model))
        elif qubits is not None:
            self.entries.append(CircuitEntry(CircuitEntryType.GATE, gate, qubits, noise_model))
        else:
            raise ValueError("Expected target qubit(s)")
        
    def add_controlled_gate(self, gate: np.ndarray, control_qubit: int, target_qubit: int, noise_model: NoiseModel = None):
        """
        Adds a controlled gate to the circuit.

        Parameters
        ----------
        gate : ndarray
            The unitary matrix representing the gate

        control_qubit : int
            The control qubit

        target_qubit : int
            The qubit that the gate is applied to

        noise_model : NoiseModel, optional
            Optional gate specific noise model
        """
        
        if control_qubit >= self.num_qubits or control_qubit < 0:
            raise ValueError("Control qubit is out of range")
        
        if target_qubit >= self.num_qubits or target_qubit < 0:
            raise ValueError("Target qubit is out of range")

        operator = [([ops.P0, ops.I], [control_qubit, target_qubit]), ([ops.P1, ops.X], [control_qubit, target_qubit])]
        self.add_composite_operator(operator, noise_model=noise_model)
        
    def add_operator(self, U: np.ndarray, noise_model: NoiseModel = None):
        """
        Adds a circuit operator to the circuit.

        Parameters
        ----------
        U : ndarray
            The unitary matrix representing the operator

        noise_model : NoiseModel, optional
            Optional specific noise model
        """
        self.entries.append(CircuitEntry(CircuitEntryType.OPERATOR, U, None, noise_model))

    def add_composite_operator(self, terms: List[Tuple[List[np.ndarray], List[int]]], noise_model: NoiseModel = None):
        """
        Adds a composite operator to the circuit.

        Parameters
        ----------
        terms : list of (operator_list, qubit_list) tuples
            e.g., ([P0, I], [0, 1]), ([P1, X], [0, 1])

        noise_model : NoiseModel, optional
            Optional specific noise model
        """
        self.entries.append(CircuitEntry(CircuitEntryType.COMPOSITE, None, None, noise_model, terms=terms))
        
    def run(self, initial_state: State = None, noise_model: NoiseModel = None) -> State:
        """
        Executes the circuit and returns the final state.

        Parameters
        ----------
        initial_state : State, optional
            The initial state passed into the circuit

        noise_model : NoiseModel, optional
            Optional global noise model

        Returns
        -------
        State
            The state produced by the circuit
        """
        state = initial_state.copy() if initial_state is not None else State(num_qubits = self.num_qubits)
        
        for i, circuit_entry in enumerate(self.entries):
            match circuit_entry.type:
                case CircuitEntryType.OPERATOR:
                    state.apply_operator(circuit_entry.matrix)
                case CircuitEntryType.GATE:
                    state.apply_gate(circuit_entry.matrix, qubits=circuit_entry.qubits)
                case CircuitEntryType.COMPOSITE:
                    state.apply_composite_operator(*circuit_entry.terms)
                case CircuitEntryType.MEASURING_DEVICE:
                    results = state.measure(qubits=list(circuit_entry.qubits))
                    if circuit_entry.callback: circuit_entry.callback(state, results)
                case CircuitEntryType.CALLBACK:
                    if circuit_entry.callback: circuit_entry.callback(state)

            # Apply gate-specific noise
            if circuit_entry.noise_model:
               circuit_entry.noise_model.apply(state)
            
            # Apply global noise
            if noise_model:
                noise_model.apply(state)
        
        return state
    
    def run_shots(self, initial_state: State = None, shots: int = 10, noise_model: NoiseModel = None) -> Generator[State, None, None]:
        """
        Parameters
        ----------
        initial_state : State, optional
            The initial state passed into the circuit

        shots : int, optional
            The number of shots to be performed

        noise_model : NoiseModel, optional
            Optional global noise model

        Returns
        -------
        Generator[State]
        """
        for i in range(shots):
            yield self.run(initial_state=initial_state, noise_model=noise_model) 

    def CNOT(self, control_qubit: int, target_qubit: int, noise_model: NoiseModel = None):
        """
        Adds a CNOT gate to the circuit.

        Parameters
        ----------
        control_qubit : int
            The control qubit

        target_qubit : tuple of int, optional
            The qubit that the gate is applied to

        noise_model : NoiseModel, optional
            Optional gate specific noise model
        """
        self.add_controlled_gate(ops.X, control_qubit, target_qubit, noise_model=noise_model)

    def measuring_device(self, qubit: int = None, qubits: List[int] = None, callback: Callable[[State, List[int]], None] = None):
        if qubit is not None:
            qubits = [qubit]
        
        if qubits is None or callback is None:
            raise ValueError("Must provide qubit(s) and callback function for measuring device")

        self.entries.append(CircuitEntry(CircuitEntryType.MEASURING_DEVICE, None, qubits=tuple(qubits), noise_model=None, callback=callback))

    def add_callback(self, callback: Callable[[State], None] = None):
        if callback is None:
            raise ValueError("Must provide callback function")

        self.entries.append(CircuitEntry(CircuitEntryType.CALLBACK, None, qubits=None, noise_model=None, callback=callback))

class NoiseModel(ABC):
    def apply(self, state: State):
        """
        Applies the noise model to the state

        Parameters
        ----------
        state : State
            The state to apply the model to
        """
        raise NotImplementedError("Apply method for noise model not implemented")