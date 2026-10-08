# Demo and Real-Camera Evaluation

## Included synthetic demo

`assets/synthetic_demo.gif` shows a generated legal cube being solved. It uses the project's state-update functions and a Kociemba solution, but **it is not a recording of the application or a webcam**.

```bash
python -m tools.make_demo
```

## Record the actual application: approximately 60–90 seconds

| Segment | Record | Explain |
|---|---|---|
| 0–10 s | Physical cube and application window | Camera input, state validation, solution, and 3D replay |
| 10–35 s | Capture all six faces | Center-color identification, nine labels, 90-degree orientation differences |
| 35–45 s | Click Done and show validation | Rotation candidates and physical constraints |
| 45–65 s | Forward/backward replay and camera movement | Arrow keys, mouse drag, scroll wheel |
| 65–90 s | Duplicate-face or recognition failure | Error feedback, undo, and recapture |

Align the face with the fixed 3×3 guide. Rotate the whole cube between captures without scrambling its layers. The preview is mirrored, while stored labels use the original camera orientation.

The fixed mapping is U=yellow, R=green, F=red, D=white, L=blue, B=orange. Cubes with a different color arrangement may not match this configuration.

Record both successful and failed attempts, and document the lighting, camera, and execution environment. Add the real recording's link to the README when available. Do not present the synthetic GIF as evidence of physical-camera performance.

## Measure real recognition performance

Copy `results/webcam_trials_template.csv` and record one row per attempt. An empty field means unmeasured, not zero. Define one face-level attempt as recognizing the nine stickers on one face.

- **Sticker accuracy:** correct stickers across all attempts / (number of attempts × 9).
- **Full-face accuracy:** attempts with all nine stickers correct / total attempts.
- **End-to-end success rate:** use `results/cube_trials_template.csv` for complete six-face capture through physical solution verification, including any recaptures.
- Include failed attempts in denominators and record the number of recaptures.
- Report lighting conditions separately. Frame rate and accuracy are different measurements.
- Separate threshold-tuning images from final evaluation images.

For end-to-end success, verify that the recovered state and solution correspond to the actual cube, in addition to checking that the simulated cube reaches a solved state.
