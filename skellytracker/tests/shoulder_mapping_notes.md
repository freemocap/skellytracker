# Shoulder attachment reference

The MediaPipe and RTMPose body mappings use the same sternoclavicular (SC)
geometry. These are estimated attachment points, not detector observations.

The coefficients were calculated from the standard-human reference pose in
SkellyForge revision `0c8edb21a29ba18dc7c92b81393c49b20e7ffe86`:

1. Use reference acromion positions as shoulders and hip sockets as hips.
2. Compute shoulder midpoint `M` and shoulder width `W`.
3. Normalize the hip-midpoint-to-shoulder-midpoint vector as `up`.
4. Orthogonalize left-to-right shoulder direction against `up` as `lateral`.
5. Set `anterior = cross(up, lateral)`.
6. For each reference attachment `P`, compute
   `dot(P - M, axis) / W` for each of these axes.

This gives lateral offsets of -/+0.062498631, anterior 0.234362559, and
up 0.002493029. The notch has zero lateral offset. The small up component
comes from the reference shoulder line; it does not change neck/chest midpoints.
Nine decimal places preserve the reference construction without implying
anatomical measurement precision.

Previously MediaPipe used unequal forward offsets (0.15 and 0.234362), while
RTMPose used 0.1 on both sides. These constructed landmarks also influence the
downstream torso pose fit. Correcting them does not solve shoulder/trunk
observability or introduce variable-length spine segments.

Tracker tests cover bilateral symmetry, detector agreement, rigid transformation
and scale equivariance, and omission when evidence is missing. They do not import
Forge. Reference-skeleton round trips and real-recording reconstruction belong
in FreeMoCap after the human commits/pushes this repository and updates its
consumer dependency. Existing prepared recordings must be reconstructed with
the updated mapping before evaluating its effects; old saved rotations will
not change automatically.

Focused validation (no image/video fixtures are needed):

```powershell
.\.venv\Scripts\python.exe -B -m pytest --noconftest skellytracker/tests/test_shoulder_mapping.py skellytracker/tests/test_directly_measured_landmarks.py skellytracker/tests/test_mapping_quality.py skellytracker/tests/test_partial_mean_omission.py -q
```

`--noconftest` avoids the existing session-wide image downloads. These tests
provide their own evidence and do not use fixtures from that conftest.
