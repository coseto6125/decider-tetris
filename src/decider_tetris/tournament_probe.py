"""Does a group tournament lose information? Ask the same decision both ways.

  uv run python -m decider_tetris.tournament_probe --decider http://127.0.0.1:8000

A game's row count cannot answer this: the two ways differ on a few per cent of the pieces, and
after the first difference the boards diverge into two unrelated games. So the boards come from
one game, and on every piece with more than `group` landings the same question is put twice, as
a tournament of groups and as one question holding them all. Agreement with the published
heuristic's pick is the yardstick, because it is the same for both.
"""

import argparse
import json
import random

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import current, placements, score
from decider_tetris.play import judge, options, stack_words, survivors


def probe(env, url, rng, wording, group, pieces):
    seen = both = single_hits = tournament_hits = 0
    single_calls, tournament_calls = [], []
    while not env.game_over and seen < pieces:
        before = current(env)
        found = placements(env)
        if not found:
            break
        choices, phrases = options(survivors(found, before), before)
        best = max(found, key=score)
        if len(choices) > group:
            state = {"piece": "IOTSZJL"[env.active_tetromino.id - len(env.base_pixels)], **stack_words(before)}
            grouped, calls_a, _ = judge(url, state, choices, phrases, rng, wording, group)
            whole, calls_b, _ = judge(url, state, choices, phrases, rng, wording, len(choices))
            seen += 1
            both += grouped.id == whole.id
            tournament_hits += grouped.id == best.id
            single_hits += whole.id == best.id
            tournament_calls += calls_a
            single_calls += calls_b
        # The heuristic drives the board so both ways keep meeting the same positions.
        for _ in range(best.rotation):
            env.active_tetromino = env.rotate(env.active_tetromino, True)
        env.x, env.y = best.x, 0
        env.step(env.actions.hard_drop)
    return {"decisions": seen, "same_pick": both, "tournament_matches_heuristic": tournament_hits,
            "one_question_matches_heuristic": single_hits,
            "tournament_ms": sum(tournament_calls), "one_question_ms": sum(single_calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--group", type=int, default=10)
    parser.add_argument("--wording", default="gaps")
    parser.add_argument("--decisions", type=int, default=60)
    args = parser.parse_args()
    env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
    env.reset(seed=args.seed)
    result = probe(env, args.decider, random.Random(args.seed), args.wording, args.group, args.decisions)
    print(json.dumps({"seed": args.seed, "group": args.group, **result}, indent=1))


if __name__ == "__main__":
    main()
