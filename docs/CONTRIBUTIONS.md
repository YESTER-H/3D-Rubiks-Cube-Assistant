# Implementation Boundaries

This document identifies functionality present in the repository and the external components it uses. Individual authorship and contributions should be described by the project author based on their actual work.

| Component | Implementation in this repository | External foundation |
|---|---|---|
| Color classification | ROIs, HSV thresholds, pixel-coverage-based labels | OpenCV, NumPy |
| Candidate-state reconstruction | Six-face rotation search, cubie-color and duplicate filters | Python itertools |
| Physical validation | Corner orientation, edge orientation, permutation parity | Established Rubik's Cube invariants |
| Solver integration | Face ordering, color serialization, move-token conversion | **kociemba.solve() computes the solution** |
| 3D replay | Rotation matrices, sticker-state updates, animation and camera controls | PyOpenGL, GLFW |
| User workflow | Capture, duplicate rejection, undo, error feedback | Tkinter, Pillow |

## Design decisions to explain

- The fixed 3×3 ROI and its limitations under arbitrary positioning or perspective.
- Why red uses two hue intervals.
- Why nine stickers of each color do not establish physical validity.
- The 4^6 orientation search space and the role of the early cubie filter.
- The distinction between finding a valid candidate and uniquely recovering the original state.
- The correspondence between image coordinates, face indices, and solver serialization order.

The project does not introduce a new cube-solving algorithm and does not use reinforcement learning or deep learning. Recognition measurements, failure analysis, and a clear account of personal contributions provide additional portfolio evidence.

## Documentation and validation additions

The supporting materials add a README, dependency files, regression tests, synthetic benchmarks, a GIF generator, and real-camera measurement templates. These synthetic validation runs are separate from the original project's development-time results.
