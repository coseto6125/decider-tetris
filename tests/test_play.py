import random
import sys

import numpy as np
import pytest

from decider_tetris import play
from decider_tetris.placements import Outcome, buried, doomed, placements, settled
from decider_tetris.play import survivors


def landing(identity, x):
    return Outcome(id=identity, rotation=0, x=x, lines=0, holes=0, height=4, bumpiness=6,
                   landing=2, row_transitions=0, column_transitions=0, wells=0, well_depth=0,
                   deep_gaps=0, splits=0, eroded=0)


def test_settle_breaks_a_tie_in_landing_order_not_by_string_hash():
    """Equal probabilities must give the same pick in every process.

    The totals used to be keyed by a set of option names, whose iteration order follows the string
    hash. That hash is randomised per process, so two runs of one seed took different landings on
    the first tie and the games parted company: one seed gave 3894, 1863 and 322 rows.
    """
    outcomes = [landing("r0c3", 3), landing("r0c7", 7)]
    phrases = {outcome.id: "Traps nothing." for outcome in outcomes}
    tied = {"r0c3": 0.5, "r0c7": 0.5}

    def judge(url, state, pool, texts, rng, wording, group, weight=None, order="shuffle",
              turn=0, turns=1, reverse=False):
        if weight is not None:
            weight.update(tied)
        return pool[0], [1], 0.5

    original, play.judge = play.judge, judge
    try:
        picks = {play.settle("", {}, outcomes, phrases, random.Random(1), "walls", 20, "enum", 0.15)[0].id
                 for _ in range(5)}
    finally:
        play.judge = original
    assert picks == {outcomes[0].id}


def test_settle_returns_the_pick_when_the_top_two_are_far_apart():
    outcomes = [landing("r0c3", 3), landing("r0c7", 7)]
    phrases = {outcome.id: "Traps nothing." for outcome in outcomes}

    def judge(url, state, pool, texts, rng, wording, group, weight=None, order="shuffle",
              turn=0, turns=1, reverse=False):
        if weight is not None:
            weight.update({"r0c3": 0.1, "r0c7": 0.9})
        return pool[-1], [1], 0.9

    original, play.judge = play.judge, judge
    try:
        pick, calls, confidence = play.settle("", {}, outcomes, phrases, random.Random(1), "walls",
                                             20, "enum", 0.15)
    finally:
        play.judge = original
    assert (pick.id, len(calls), confidence) == ("r0c7", 1, 0.9)


def buries(identity, holes, lines=0):
    return Outcome(id=identity, rotation=0, x=0, lines=lines, holes=holes, height=6, bumpiness=8,
                   landing=3, row_transitions=0, column_transitions=0, wells=0, well_depth=0,
                   deep_gaps=0, splits=0, eroded=0)


def test_survivors_keeps_only_the_fewest_traps_when_every_landing_buries():
    before = buries("now", 2)
    offered = survivors([buries("a", 3), buries("b", 14), buries("c", 3)], before, least=True)
    assert [outcome.id for outcome in offered] == ["a", "c"]


def test_survivors_keeps_a_row_clear_even_when_it_buries_more():
    before = buries("now", 2)
    offered = survivors([buries("a", 3), buries("b", 6, lines=1)], before, least=True)
    assert [outcome.id for outcome in offered] == ["a", "b"]


def test_survivors_ignores_least_while_a_clean_landing_is_on_offer():
    before = buries("now", 2)
    offered = survivors([buries("a", 2), buries("b", 3)], before, least=True)
    assert [outcome.id for outcome in offered] == ["a"]


def test_survivors_adds_the_cheapest_dirty_landing_when_clean_ones_run_out():
    before = buries("now", 1)
    found = [buries("clean", 1), buries("one", 2), buries("many", 9)]
    assert [o.id for o in survivors(found, before, relax=3)] == ["clean", "one"]


def test_survivors_keeps_the_red_line_while_enough_clean_landings_remain():
    before = buries("now", 1)
    found = [buries("a", 1), buries("b", 1), buries("c", 1), buries("dirty", 2)]
    assert [o.id for o in survivors(found, before, relax=3)] == ["a", "b", "c"]


def welled(identity, depth, holes=0, deep=None):
    return Outcome(id=identity, rotation=0, x=0, lines=0, holes=holes, height=6, bumpiness=8,
                   landing=3, row_transitions=0, column_transitions=0, wells=depth,
                   well_depth=depth, deep_gaps=depth >= 3 if deep is None else deep,
                   splits=0, eroded=0)


def test_survivors_hides_a_deep_gap_while_a_shallow_landing_is_on_offer():
    before = welled("now", 0)
    offered = survivors([welled("deep", 4), welled("flat", 1), welled("dip", 2)], before, gap="any")
    assert [outcome.id for outcome in offered] == ["flat", "dip"]


def test_survivors_stands_aside_when_every_landing_leaves_a_deep_gap():
    before = welled("now", 0)
    offered = survivors([welled("a", 4), welled("b", 5)], before, gap="any")
    assert [outcome.id for outcome in offered] == ["a", "b"]


def test_survivors_prefers_one_deep_gap_over_two_under_fewest():
    """`fewest` acts on a board that already carries a deep gap, where `any` stands aside.

    It measured worse all the same: 2267 rows against 5893 on seed 6. Forcing the gap to be filled
    buries cells and cuts the option list, so the code plays instead of the model.
    """
    before = welled("now", 0)
    offered = survivors([welled("two", 4, deep=2), welled("one", 5, deep=1)], before, gap="fewest")
    assert [outcome.id for outcome in offered] == ["one"]


def test_survivors_prefers_a_clean_landing_over_a_shallow_gap():
    before = welled("now", 0)
    offered = survivors([welled("clean", 1), welled("dirty", 1, holes=2)], before, gap="any")
    assert [outcome.id for outcome in offered] == ["clean"]


def test_survivors_never_offers_a_burying_landing_to_avoid_a_deep_gap():
    """The gap rule runs inside the buried-cell rule, never instead of it.

    Run outside it, a board where every clean landing left a deep gap handed back the burying
    landings, and the buried cells per piece went from 0.08 to 3.77 over the first 431 pieces.
    """
    before = welled("now", 0)
    deep_but_clean = welled("clean", 5)
    shallow_but_dirty = welled("dirty", 1, holes=3)
    assert [o.id for o in survivors([deep_but_clean, shallow_but_dirty], before, gap="any")] == ["clean"]


def test_survivors_lets_a_burying_landing_through_only_at_the_traded_row_count():
    before = buries("now", 1)
    one_row, two_rows = buries("one", 3, lines=1), buries("two", 3, lines=2)
    found = [buries("clean", 1), one_row, two_rows]
    assert [o.id for o in survivors(found, before, trading=2)] == ["clean", "two"]
    assert [o.id for o in survivors(found, before, trading=0)] == ["clean"]


def test_relax_waits_for_the_stack_to_reach_the_given_height():
    low, high_stack = buries("now", 1), Outcome(
        id="now", rotation=0, x=0, lines=0, holes=1, height=17, bumpiness=8, landing=3,
        row_transitions=0, column_transitions=0, wells=0, well_depth=0, deep_gaps=0, splits=0,
        eroded=0)
    found = [buries("clean", 1), buries("dirty", 2)]
    assert [o.id for o in survivors(found, low, relax=3, high=15)] == ["clean"]
    assert [o.id for o in survivors(found, high_stack, relax=3, high=15)] == ["clean", "dirty"]


def stack(env, rows):
    """Put a stack into a fresh environment; `rows` reads top to bottom, '#' is a filled cell."""
    env.reset(seed=0)
    field = np.array([[2 if cell == "#" else 0 for cell in row] for row in rows], dtype=env.board.dtype)
    env.board[:] = 1
    env.board[:env.height, env.padding:env.padding + env.width] = 0
    env.board[env.height - field.shape[0]:env.height, env.padding:env.padding + env.width] = field
    return env.crop_padding(env.board.copy())


def fresh():
    import gymnasium as gym
    from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401
    return gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped


def test_doomed_is_false_when_the_next_piece_still_has_a_clean_landing():
    env = fresh()
    field = stack(env, ["##########", "##########"])
    square = env.tetrominoes[1].matrix
    assert doomed(env, field, square) is False


def test_doomed_is_true_when_every_landing_of_the_next_piece_buries_a_cell():
    """A comb surface leaves a square nowhere to sit: each landing spans a gap and covers it."""
    env = fresh()
    field = stack(env, [".#.#.#.#.#", ".#.#.#.#.#"])
    square = env.tetrominoes[1].matrix
    assert doomed(env, field, square) is True


def test_settled_returns_the_stack_the_landing_leaves():
    env = fresh()
    env.reset(seed=6)
    env.active_tetromino = env.tetrominoes[1]  # the square
    env.reset_tetromino_position()
    found = placements(env)
    landing_here = next(outcome for outcome in found if outcome.id == "r0c0")
    assert buried(settled(env, landing_here)) == landing_here.holes


def test_main_refuses_seed_zero_because_the_environment_ignores_it():
    """The randomizer reads `if seed and seed > 0`, so seed 0 draws a fresh bag per process.

    A run under seed 0 looks reproducible and is not: two processes play different pieces from the
    first one, and every comparison drawn from them is a comparison of two different games.
    """
    argv = sys.argv
    sys.argv = ["play", "--seed", "0", "--no-window"]
    try:
        with pytest.raises(SystemExit):
            play.main()
    finally:
        sys.argv = argv


def test_doomed_is_true_when_the_next_piece_cannot_spawn():
    """The environment spawns the next piece unrotated at the centre and ends the game if it does
    not fit. A tower in the middle with clean room at the sides is that board, and searching the
    landings alone answered "not doomed" for five of the seven pieces."""
    env = fresh()
    field = stack(env, ["....##....", "....##....", "....##....", "....##...."] * 5)
    for piece in env.tetrominoes:
        assert doomed(env, field, piece.matrix) is True, piece.id
