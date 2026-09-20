"""Which disagreements with the heuristic actually cost the game?

  uv run python -m decider_tetris.regret --decider http://127.0.0.1:8000 --seed 6

The model and the published heuristic pick a different landing on about a third of the pieces, yet
the model clears almost as many rows, so most of those differences cannot matter. This finds the
ones that do: at every disagreement the board is cloned twice, each side's landing is played, and
the heuristic then plays both clones forward over the same pieces. The difference in rows cleared,
and in whether the clone survives, is what that one decision cost.

The facts of the costly decisions are printed last: they say which fact is missing or too weak.
"""

import argparse
import copy
import json
import random

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import current, phrase, placements, score
from decider_tetris.play import options, settle, stack_words, survivors


def drop(env, landing):
    for _ in range(landing.rotation):
        env.active_tetromino = env.rotate(env.active_tetromino, True)
    env.x, env.y = landing.x, 0
    return int(env.step(env.actions.hard_drop)[4]["lines_cleared"])


def rollout(env, horizon):
    """Let the heuristic play this board on, so both clones are judged by the same player."""
    rows = 0
    for _ in range(horizon):
        if env.game_over:
            break
        found = placements(env)
        if not found:
            break
        rows += drop(env, max(found, key=score))
    return rows, env.game_over


def run(env, url, rng, wording, group, order, threshold, pieces, horizon):
    costly, seen, disagreements = [], 0, 0
    while not env.game_over and seen < pieces:
        before = current(env)
        found = placements(env)
        if not found:
            break
        choices, phrases = options(survivors(found, before), before)
        baseline = max(found, key=score)
        state = {"piece": "IOTSZJL"[env.active_tetromino.id - len(env.base_pixels)], **stack_words(before)}
        pick, _, _ = settle(url, state, choices, phrases, rng, wording, group, order, threshold)
        if pick.id != baseline.id:
            disagreements += 1
            outcomes = []
            for landing in (pick, baseline):
                clone = copy.deepcopy(env)
                rows = drop(clone, landing)
                more, died = rollout(clone, horizon)
                outcomes.append((rows + more, died))
            (model_rows, model_died), (base_rows, base_died) = outcomes
            if base_rows - model_rows > 0 or (model_died and not base_died):
                costly.append({"piece": seen, "cost_rows": base_rows - model_rows,
                               "only_the_model_died": bool(model_died and not base_died),
                               "model": phrases.get(pick.id, phrase(pick, before)),
                               "heuristic": phrase(baseline, before)})
        drop(env, pick)
        seen += 1
    return seen, disagreements, costly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", required=True)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--pieces", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=30, help="pieces the heuristic plays on each clone")
    parser.add_argument("--wording", default="gaps")
    parser.add_argument("--group", type=int, default=20)
    parser.add_argument("--order", default="enum")
    parser.add_argument("--settle", type=float, default=0.15)
    args = parser.parse_args()
    env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
    env.reset(seed=args.seed)
    seen, disagreements, costly = run(env, args.decider, random.Random(args.seed), args.wording, args.group,
                                      args.order, args.settle, args.pieces, args.horizon)
    costly.sort(key=lambda row: (-row["only_the_model_died"], -row["cost_rows"]))
    print(json.dumps({"pieces": seen, "disagreements": disagreements, "costly": len(costly),
                      "rows_behind": sum(row["cost_rows"] for row in costly),
                      "deaths_the_heuristic_avoided": sum(row["only_the_model_died"] for row in costly)},
                     indent=1))
    for row in costly[:8]:
        print(f"\npiece {row['piece']}  cost {row['cost_rows']} rows"
              f"{'  and only the model died' if row['only_the_model_died'] else ''}")
        print(f"  model:     {row['model']}")
        print(f"  heuristic: {row['heuristic']}")


if __name__ == "__main__":
    main()
