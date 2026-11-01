# SkellyTracker ownership

Follow the containing workspace's authorization and Git handoff rules.

Tracker owns detector recipes, crop rules, keypoint schemas and connections,
output assembly, preset-specific mapping selection, and mapping evaluation.
These features must work without FreeMoCap or SkellyForge imports, including tests.
Applications select a preset and supply settings and images; they must not need
to reconstruct its detector tree or keypoint schema.

Detector outputs are keypoints. Mappings relate keypoints to named anatomical
landmarks; anatomical landmark definitions and reconstruction belong to Forge.
The `Observation` tracking container is distinct from mapped anatomical observations.
Prepare numeric indices once and operate on dense arrays. Preserve observation
ownership: reusing scratch memory must never change previously returned results.

For combined RTMW/MediaPipe work, read `TRACKER_COMPOSITION.md` before changing
presets, output schemas, crop rules, or mapping selection.
