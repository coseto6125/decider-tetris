import gymnasium as gym
import pytest
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import COUNT, current, phrase, placements, surface

PIECES = {name: index for index, name in enumerate("IOTSZJL")}


@pytest.fixture
def env():
    env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
    env.reset(seed=1)
    return env


def hold(env, name):
    env.active_tetromino = env.tetrominoes[PIECES[name]]
    env.reset_tetromino_position()


def fill(env, row, columns):
    """Put blocks on one row of the playfield, counting rows up from the floor."""
    for column in columns:
        env.board[env.height - row, env.padding + column] = 2


def test_placements_offer_every_column_to_a_square_piece(env):
    hold(env, "O")
    assert sorted(int(p.id.split("c")[1]) for p in placements(env)) == list(range(env.width - 1))


def test_placements_offer_one_landing_per_distinct_stack(env):
    hold(env, "I")
    # Flat and upright reach 7 and 10 columns; the other two turns repeat those stacks.
    assert len(placements(env)) == 17


def test_placement_filling_the_last_gap_clears_the_row(env):
    fill(env, 1, range(env.width - 2))
    hold(env, "O")
    landing = next(p for p in placements(env) if p.id.endswith("c8"))
    assert landing.lines == 1


def test_placement_beside_a_deep_notch_traps_a_cell(env):
    for row in (1, 2, 3):
        fill(env, row, [0])
    hold(env, "O")
    landing = next(p for p in placements(env) if p.id.endswith("c0"))
    assert landing.holes == 3  # the square bridges the notch and seals the column beside it
    assert "Traps three cells" in phrase(landing, current(env))


def test_phrase_leads_with_the_trapped_cells(env):
    fill(env, 1, range(env.width - 2))
    hold(env, "O")
    landing = next(p for p in placements(env) if p.id.endswith("c8"))
    assert phrase(landing, current(env)).startswith("Traps nothing. Clears one row.")


def test_surface_reports_zero_for_an_empty_column(env):
    fill(env, 1, [3])
    heights = surface(env.crop_padding(env.board.copy()))
    assert heights[3] == 1 and heights[4] == 0
    assert COUNT[0] == "no"


def test_phrase_reports_the_rise_of_the_stack_not_a_fixed_band(env):
    # Leave one column empty: a full row would clear on the next placement.
    fill(env, 1, range(env.width - 1))
    fill(env, 2, range(env.width - 1))
    before = current(env)
    hold(env, "O")
    landing = next(p for p in placements(env) if p.id.endswith("c4"))
    assert "It raises the top of the stack by two rows." in phrase(landing, before)


def test_phrase_keeps_its_edge_when_the_stack_is_high(env):
    for row in range(1, 14):
        fill(env, row, range(env.width - 1))
    before = current(env)
    hold(env, "O")
    raised = next(p for p in placements(env) if p.id.endswith("c4"))
    # Both landings sit in a high stack; a fixed band called them the same, the change does not.
    assert phrase(raised, before) != phrase(before, before)
