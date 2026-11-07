import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.rl.models.entity_projector import EntityProjector
from src.rl.schema.state import COL_START_CHARACTER
from src.rl.schema.state import COL_START_HEALTH_BLOCK
from src.rl.schema.state import COL_START_MONSTER
from src.rl.schema.state import COL_START_POTION
from src.rl.schema.state import COL_START_PRICE
from src.rl.schema.state import COL_START_RELIC
from src.rl.schema.state import COL_START_STASIS_SLOT
from src.rl.schema.state import COL_START_TAG


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("src.rl.schema", ("src.rl.encoding", "src.rl.models", "src.rl.algorithms")),
        ("src.rl.models", ("src.rl.encoding", "src.rl.algorithms")),
        ("src.rl.actions", ("src.rl.encoding", "src.rl.models", "src.rl.algorithms")),
        ("src.rl.batches", ("src.rl.encoding", "src.rl.models", "src.rl.algorithms")),
        (
            "src.rl.algorithms.actor_critic.curriculum",
            ("src.rl.algorithms.actor_critic.checkpoint", "src.rl.models"),
        ),
    ],
)
def test_imports_stay_within_their_dependency_boundaries(module, forbidden):
    code = f"""
import importlib
import sys
importlib.import_module({module!r})
loaded = [name for name in sys.modules if any(
    name == prefix or name.startswith(prefix + '.') for prefix in {forbidden!r}
)]
assert not loaded, loaded
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_projector_initializes_each_feature_range_independently():
    projector = EntityProjector(dim_entity=16, dim_map=8)
    projector.init_token_weights()
    for x_range in torch.tensor_split(
        projector._pre_token.weight,
        (
            COL_START_RELIC,
            COL_START_POTION,
            COL_START_MONSTER,
            COL_START_CHARACTER,
            COL_START_HEALTH_BLOCK,
            COL_START_PRICE,
            COL_START_STASIS_SLOT,
            COL_START_TAG,
        ),
        dim=1,
    ):
        x_gram = (
            x_range @ x_range.T if x_range.shape[0] <= x_range.shape[1] else x_range.T @ x_range
        )
        torch.testing.assert_close(x_gram, 2 * torch.eye(len(x_gram)), rtol=1e-5, atol=1e-5)
