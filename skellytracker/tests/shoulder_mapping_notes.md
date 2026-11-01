# Shoulder attachment reference

The MediaPipe and RTMPose body mappings use the same sternoclavicular (SC)
geometry. These are estimated attachment points, not detector observations.

The coefficients were recalculated on 2026-11-01 from the editable SkellyForge
reference pose: SC joints/notch at 70% anterior depth and xiphoid at 60%.
This supersedes the coefficients from revision
`0c8edb21a29ba18dc7c92b81393c49b20e7ffe86`. The derivation is unchanged:

1. Use reference acromion positions as shoulders and hip sockets as hips.
2. Compute shoulder midpoint `M` and shoulder width `W`.
3. Normalize the hip-midpoint-to-shoulder-midpoint vector as `up`.
4. Orthogonalize left-to-right shoulder direction against `up` as `lateral`.
5. Set `anterior = cross(up, lateral)`.
6. For each reference attachment `P`, compute
   `dot(P - M, axis) / W` for each of these axes.

SC lateral offsets are -/+0.062498631, anterior 0.234184536150, and up
-0.009467177190. The notch has zero lateral offset and identical anterior/up.
Xiphoid anterior is 0.283314074153 and up is -0.496223613273. These include the
reference shoulder-to-hip tilt; neck/chest midpoints are unchanged. Decimal places
preserve the construction without implying anatomical measurement precision.
Do not multiply these ratios by the depth percentage: changing the SC attachment
also moves the reference acromia, changing the mapping's origin and axes.

Previously MediaPipe used unequal forward offsets (0.15 and 0.234362), while
RTMPose used 0.1 on both sides. These constructed landmarks also influence the
downstream torso pose fit. Correcting them does not solve shoulder/trunk
observability or introduce variable-length spine segments.

Tracker tests cover bilateral symmetry, detector agreement, rigid transformation
and scale equivariance, and omission when evidence is missing. They do not import
Forge. Reference-skeleton round trips and real-recording reconstruction belong
in FreeMoCap through its configured editable dependencies. Core's
`freemocap/tests/test_model_presentation.py` checks scalar and prepared mapping
against current Forge reference landmarks under rotation, translation and three
body scales for all three body presets. Existing recordings keep their saved
mappings and geometry; fresh processing uses these corrected definitions.

Focused validation (no image/video fixtures are needed):

```powershell
.\.venv\Scripts\python.exe -B -m pytest --noconftest skellytracker/tests/test_shoulder_mapping.py skellytracker/tests/test_directly_measured_landmarks.py skellytracker/tests/test_mapping_quality.py skellytracker/tests/test_partial_mean_omission.py -q
```

`--noconftest` avoids the existing session-wide image downloads. These tests
provide their own evidence and do not use fixtures from that conftest.
