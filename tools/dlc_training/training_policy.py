"""Explicit training choices; retain the installed DLC augmentation recipe."""

from copy import deepcopy


def configure_training_policy(pose: dict, augmentation_policy: str) -> None:
    """Separate production augmentation from the explicitly unaugmented diagnostic.

    DLC owns transformation construction and keypoint transforms. Its documented
    default recipe supplies affine/crop/noise/blur; do not reproduce it here.
    Fixed geometric identities have no declared mirror correspondence, so disable
    horizontal flips. See [@dlcPytorchConfiguration] and [@dlcTrainingSettings302].
    """
    if augmentation_policy not in ("dlc-default", "diagnostic-none"):
        raise ValueError("Unknown augmentation policy")
    if augmentation_policy == "diagnostic-none":
        pose["data"]["train"] = deepcopy(pose["data"]["inference"])
    pose["data"]["train"]["hflip"] = False
