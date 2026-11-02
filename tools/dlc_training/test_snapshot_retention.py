"""Exercise actual DLC retention, including promotion of previous best checkpoints."""

import pytest


@pytest.mark.parametrize("retained_count,keeps_epoch_ten", [(5, False), (12, True)])
def test_budget_comparison_retains_fixed_epochs(tmp_path, monkeypatch, retained_count, keeps_epoch_ten):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("DLClight", "True")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import torch
    from deeplabcut.pose_estimation_pytorch.runners.snapshots import TorchSnapshotManager

    manager = TorchSnapshotManager(
        "snapshot", tmp_path, key_metric="test.mAP",
        max_snapshots=retained_count, save_epochs=10,
    )
    for epoch in range(10, 101, 10):
        manager.update(epoch, {
            "metadata": {"epoch": epoch, "metrics": {"metrics/test.mAP": float(epoch)}},
            "model": {"test_weight": torch.tensor([epoch], device="cpu")},
        }, last=epoch == 100)
    saved_epochs = {
        torch.load(path, map_location="cpu", weights_only=True)["metadata"]["epoch"]
        for path in tmp_path.glob("snapshot-*.pt")
    }
    assert (10 in saved_epochs) is keeps_epoch_ten
    assert 100 in saved_epochs
