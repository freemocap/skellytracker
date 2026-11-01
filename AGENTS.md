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

## Recover the current task

At milestones, preserve the task objective, decisions and rationale, existing
interfaces, exact checkout, validation results and next action in its checkpoint.
Name the relevant ontology terms, plan stage, and a short recovery reading list.
After compaction or handoff, read that checkpoint and this ownership guidance,
then inspect only the source and diff relevant to the next action:

- Presets/crops/assembly: [composition](TRACKER_COMPOSITION.md) and
  [tracker implementation](skellytracker/core/tracker/).
- Mapping semantics: [mapping interface](skellytracker/core/io/tracker_mapping.py)
  and the relevant mapping definition under `skellytracker/core/io/`.
- Output ownership and shapes: [data primitives](skellytracker/core/data_primitives/).
- Execution provider behavior: [sessions](skellytracker/core/sessions/) and the
  current [README](README.md), rather than a GPU assumption from an installed extra.

Search these existing interfaces before creating a new schema or recipe.
Treat dated implementation descriptions in CLAUDE.md and rearchitecture plans as
context to verify, not permission to reopen settled decisions. Continue the
approved work after targeted recovery; ask only for consequential ambiguity
about intent that inspection cannot resolve. Preserve existing edits and leave
Git state to the human. Component tests remain independent of the other Skellies;
cross-repository contract checks belong in Core.
