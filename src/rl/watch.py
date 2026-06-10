"""Watch a trained agent play in the engine's curses TUI.

Reuses the engine `play/` renderer (so the agent plays in the same nice UI a
human does) but replaces the human keypress with the model's greedy action —
the exact inference path the trainer uses (encode → mask from legal actions →
forward). Dependency direction stays trainer → engine.

Controls: [space] pause/resume, [n] single step, [r] new run, [+/-] speed,
[q] quit.

Usage:
    python -m src.rl.watch --exp-path experiments/ppo/NEWERA
    python -m src.rl.watch --random            # untrained model, no checkpoint
"""

import curses
import random
import sys
import time
from pathlib import Path

import click
import slai
import torch


# The engine's `play/` package lives at the engine repo root (sibling of the
# installed `slai` python package): .../slai/python/slai/__init__.py -> .../slai
_ENGINE_ROOT = Path(slai.__file__).resolve().parents[2]
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from play.__main__ import Cursor  # noqa: E402
from play.__main__ import _probe_acs  # noqa: E402
from play.__main__ import init_colors  # noqa: E402
from play.__main__ import render  # noqa: E402
from play.__main__ import reset_if_phase_changed  # noqa: E402
from play.__main__ import write_segments  # noqa: E402

from src.rl.constants import FAST_MODE  # noqa: E402
from src.rl.models import ActorCritic  # noqa: E402
from src.rl.test_agent import _fmt_action  # noqa: E402
from src.rl.test_agent import get_action_from_model  # noqa: E402
from src.rl.utils import load_config  # noqa: E402


def _random_seed() -> int:
    return random.randint(0, 2**31 - 1)


def _loop(stdscr, model, device, ascension: int, delay: float, fast_mode: bool) -> None:
    curses.curs_set(0)
    init_colors()
    _probe_acs(stdscr)
    stdscr.keypad(True)
    stdscr.nodelay(True)  # non-blocking input so we can auto-advance on a timer

    seed = _random_seed()
    env = slai.GameEnv(ascension, fast_mode=fast_mode)
    view = env.reset(seed=seed)
    cursor = Cursor()
    paused = False
    pending = None  # cached agent action for the current state

    while True:
        reset_if_phase_changed(cursor, view)
        legal = env.get_legal_actions()

        if pending is None and legal and not view.game_over:
            pending, _ = get_action_from_model(model, view, legal, device, greedy=True)

        stdscr.erase()
        render(stdscr, view, cursor, legal, ascension)
        _maxy, maxx = stdscr.getmaxyx()
        state = "PAUSED" if paused else f"{delay:.2f}s"
        act = _fmt_action(pending) if pending is not None else "—"
        status = (
            f" agent: {act}   [{state}]   "
            f"[space]pause [n]step [+/-]speed [r]run [q]quit   seed {seed} "
        )
        write_segments(stdscr, 0, maxx - 1, [(status, {"dim": True})])
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in (ord("r"), ord("R")):
            seed = _random_seed()
            view = env.reset(seed=seed)
            cursor = Cursor()
            pending = None
            continue
        if key == ord(" "):
            paused = not paused
        if key in (ord("+"), ord("=")):
            delay = max(0.0, delay - 0.1)
        if key in (ord("-"), ord("_")):
            delay = min(5.0, delay + 0.1)
        step_now = key in (ord("n"), ord("N"))

        if view.game_over or pending is None:
            time.sleep(0.05)
            continue
        if paused and not step_now:
            time.sleep(0.02)
            continue

        try:
            view, _terminated = env.step(pending)
            cursor.error = None
        except Exception as e:  # surface engine rejection in the UI
            cursor.error = str(e)
        pending = None
        if not step_now:
            time.sleep(delay)


@click.command()
@click.option(
    "--exp-path",
    default="experiments/ppo/NEWERA",
    help="Experiment dir with config.yml + checkpoint.pth",
)
@click.option(
    "--random", "use_random", is_flag=True, help="Use a fresh untrained model (no checkpoint)"
)
@click.option("--ascension", default=0, type=int, help="Ascension level")
@click.option("--delay", default=0.6, type=float, help="Seconds between agent moves")
@click.option("--device", default="cpu", type=str)
@click.option(
    "--no-fast-mode", is_flag=True, help="Show every state (don't auto-skip trivial nodes)"
)
def main(exp_path, use_random, ascension, delay, device, no_fast_mode):
    """Watch a trained agent play in the curses TUI."""
    if use_random:
        model = ActorCritic(
            dim_entity=32,
            dim_global=64,
            transformer_dim_ff=64,
            transformer_num_heads=2,
            transformer_num_blocks=1,
            map_encoder_kernel_size=3,
            map_encoder_dim=16,
            dim_ff_primary=32,
            dim_ff_card=32,
            dim_ff_monster=32,
            dim_ff_map=32,
            dim_ff_value=32,
        )
    else:
        config = load_config(f"{exp_path}/config.yml")
        model = ActorCritic(**config["model"])
        ckpt = torch.load(f"{exp_path}/checkpoint.pth", weights_only=True)
        model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    fast_mode = FAST_MODE and not no_fast_mode
    curses.wrapper(_loop, model, dev, ascension, delay, fast_mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
