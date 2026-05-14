import torch

from src.utils.checkpoint import load_pt_checkpoint


def test_load_pt_checkpoint_dict(tmp_path):
    path = tmp_path / "stub.pt"
    torch.save({"config": {"n": 1}, "model_state": {}}, path)
    obj = load_pt_checkpoint(path, map_location="cpu")
    assert obj["config"]["n"] == 1
    assert "model_state" in obj
