from __future__ import annotations

import warnings

from safemoe.data.datamodule import MultiDataLoader


def test_effective_num_workers_caps_per_training_loader(monkeypatch):
    monkeypatch.setattr("safemoe.data.datamodule.os.cpu_count", lambda: 24)
    data = MultiDataLoader(num_workers=64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        effective = data._effective_num_workers(loader_count=3)

    assert effective == 8
    assert any("Reducing MultiDataLoader num_workers" in str(w.message) for w in caught)


def test_effective_num_workers_keeps_safe_value(monkeypatch):
    monkeypatch.setattr("safemoe.data.datamodule.os.cpu_count", lambda: 24)
    data = MultiDataLoader(num_workers=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        effective = data._effective_num_workers(loader_count=3)

    assert effective == 4
    assert not caught


def test_effective_num_workers_caps_even_on_large_hosts(monkeypatch):
    monkeypatch.setattr("safemoe.data.datamodule.os.cpu_count", lambda: 64)
    data = MultiDataLoader(num_workers=64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        effective = data._effective_num_workers(loader_count=3)

    assert effective == 8
    assert any("caps streaming workers at 8 per loader" in str(w.message) for w in caught)


def test_effective_num_workers_accounts_for_world_size(monkeypatch):
    monkeypatch.setattr("safemoe.data.datamodule.os.cpu_count", lambda: 24)
    monkeypatch.setenv("WORLD_SIZE", "4")
    data = MultiDataLoader(num_workers=64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        effective = data._effective_num_workers(loader_count=3)

    assert effective == 2
    assert any("WORLD_SIZE=4" in str(w.message) for w in caught)


def test_effective_num_workers_allows_zero():
    data = MultiDataLoader(num_workers=0)
    assert data._effective_num_workers(loader_count=3) == 0
