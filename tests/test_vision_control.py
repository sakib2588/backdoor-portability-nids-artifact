import numpy as np
import torch

from src import vision_control as vc


def test_add_patch_sets_corner():
    x = torch.zeros(2, 1, 28, 28)
    out = vc.add_patch(x, size=4, value=1.0)
    # bottom-right 4x4 is all ones; a far pixel stays zero
    assert torch.all(out[:, :, 24:28, 24:28] == 1.0)
    assert out[0, 0, 0, 0] == 0.0
    # original is not mutated
    assert torch.all(x == 0.0)


def test_poison_trainset_relabels_and_marks():
    x = torch.zeros(100, 1, 28, 28)
    y = torch.arange(100) % 10  # 10 of each class
    x_p, y_p, idx = vc.poison_trainset(x, y, target=0, frac=0.1, seed=42)
    # 10% of the 90 non-target samples = 9 poisoned
    assert len(idx) == 9
    # every poisoned sample now has label 0 and a patched corner
    assert torch.all(y_p[idx] == 0)
    assert torch.all(x_p[idx][:, :, 24:28, 24:28] == 1.0)
    # no originally-target sample was poisoned
    assert all(y[i] != 0 for i in idx)


def test_smallcnn_shapes():
    from src.vision_control import SmallCNN
    m = SmallCNN()
    x = torch.zeros(5, 1, 28, 28)
    assert m.features(x).shape == (5, 128)
    assert m(x).shape == (5, 10)
