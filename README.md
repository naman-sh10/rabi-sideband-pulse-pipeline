# Rabi-sideband bosonic state preparation

A simulation pipeline that designs, exports and tests pulses for preparing bosonic states (Fock, cat and GKP) in a superconducting cavity memory. The cavity is dispersively coupled to a transmon. The method uses a Rabi-driven qubit and a two-tone sideband pulse on the cavity.

Open [`rabi_sideband_pipeline.ipynb`](rabi_sideband_pipeline.ipynb). It runs three stages:

1. **Effective-Hamiltonian optimizer.** Optimal control of the two sideband phases in the displaced, dressed frame. Supports Fock, cat and GKP targets, with full-fidelity or characteristic-function objectives.
2. **Export for hardware.** Rescales the solution, builds a schedule on the 4 ns grid aligned to the beat nulls, and writes a solution folder of pulse parameters.
3. **Interaction-frame simulation of the full sequence.** Compares time-dependent and constant Stark-shift correction, and searches over integer (beat-aligned) segment lengths.

## Files

| File | Contents |
| --- | --- |
| `rabi_sideband_pipeline.ipynb` | The notebook, with outputs from the default cat run |
| `rabi_sideband_pipeline.py` | Optimizer, export, simulation and plotting code |
| `data/gkp_points_Delta0340.csv` | The 32 characteristic-function points used by the GKP objective |

## Running

Requires Python 3 with NumPy, SciPy, TensorFlow ≥ 2.16, QuTiP 5 and Matplotlib. Run the notebook from this folder. The default target (an even cat state with α = 1.5) takes a few minutes on a laptop CPU. Set `TARGET = 'fock2'` or `'gkp'` for the other presets.

Two parts need lab files that are not in this repository: the hardware-format check in stage 2, and the comparison with previously exported pulses. Without those files, the notebook skips both parts and prints a message.

Author: Naman Shhade, Hacohen-Gourgy group, Technion.
