"""Two silent-corruption guards for the architecture-parameterised NC pilot: a transformer run must
not overwrite the committed MLP artifacts, and must not resume from an MLP checkpoint.

Both failures would be invisible at runtime. Overwriting destroys a finished 102-row null; resuming
across architectures pools two victim families into one population and every downstream statistic
would be computed on a mixture nobody intended. Neither raises, so neither is caught without a test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(script_name: str, module_name: str):
    """Import a numbered script by path -- `scripts/65_...` is not a legal module name.

    Deliberately identical to the loader in tests/test_rung2_pilot_statistics.py, including the
    `sys.modules` registration before exec_module. Two different loaders for the same script would
    be a trap for whoever edits one of them next."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / script_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pilot():
    return _load("65_nc_mask_geometry_pilot.py", "nc_mask_geometry_pilot_archtest")


CFG = {"nc_steps": 300, "nc_sample": 400, "mlp_epochs": 20, "n_replicas": 34}


def test_output_paths_differ_between_architectures(pilot):
    mlp = pilot.output_paths("mlp")
    ft = pilot.output_paths("ft_transformer")
    assert mlp["out"] != ft["out"]
    assert mlp["ckpt"] != ft["ckpt"]
    assert mlp["masks"] != ft["masks"]


def test_mlp_paths_are_unchanged_so_the_committed_run_stays_reproducible(pilot):
    mlp = pilot.output_paths("mlp")
    assert mlp["out"].name == "nc_mask_geometry_pilot.json"
    assert mlp["ckpt"].name == "nc_mask_geometry_pilot.checkpoint.json"
    assert mlp["masks"].name == "nc_pilot_masks"


def test_config_key_separates_architectures(pilot):
    a = pilot._config_key([42], CFG, "calibration", "mlp")
    b = pilot._config_key([42], CFG, "calibration", "ft_transformer")
    assert a != b, "an ft_transformer run would resume from the MLP checkpoint"


def test_config_key_separates_transformer_sizes(pilot):
    """The one authorised gate-0 retry changes model size. That must also invalidate the key, or a
    d=32 run would resume from d=16 rows."""
    small = pilot._config_key([42], CFG, "calibration", "ft_transformer",
                              arch_kwargs={"d_token": 16, "n_layers": 2, "n_heads": 4})
    big = pilot._config_key([42], CFG, "calibration", "ft_transformer",
                            arch_kwargs={"d_token": 32, "n_layers": 3, "n_heads": 8})
    assert small != big


def test_mlp_key_is_byte_identical_to_the_committed_checkpoint(pilot):
    """The load-bearing check of the whole seam.

    The committed MLP checkpoint was written before this parameterisation existed. If adding the
    architecture fields changed the default key, that finished run would silently stop matching its
    own artifact and any resume would start from zero. Skips rather than fails when the checkpoint
    is absent, so the suite still runs on a fresh clone."""
    import json

    ckpt = ROOT / "results" / "nc_mask_geometry_pilot.checkpoint.json"
    if not ckpt.exists():
        pytest.skip("committed MLP checkpoint not present in this working tree")
    committed = json.loads(ckpt.read_text())["config_key"]
    rebuilt = pilot._config_key(committed["seeds"],
                                {"nc_steps": committed["nc_steps"],
                                 "nc_sample": committed["nc_sample"],
                                 "mlp_epochs": committed["mlp_epochs"],
                                 "n_replicas": committed["n_replicas"]},
                                committed["stage"])
    assert rebuilt == committed, "the MLP key drifted; the committed run is no longer resumable"


def test_arch_fields_are_absent_from_the_mlp_key_and_present_otherwise(pilot):
    """States the asymmetry explicitly so a future tidy-up does not "simplify" it away."""
    mlp = pilot._config_key([42], CFG, "calibration", "mlp")
    ft = pilot._config_key([42], CFG, "calibration", "ft_transformer", arch_kwargs={"d_token": 16})
    assert "arch" not in mlp and "arch_kwargs" not in mlp
    assert ft["arch"] == "ft_transformer" and ft["arch_kwargs"] == {"d_token": 16}


def test_train_victim_rejects_an_unknown_architecture(pilot):
    with pytest.raises(ValueError, match="unknown arch"):
        pilot.train_victim("resnet", None, None, 42, 20, "cpu", {})
