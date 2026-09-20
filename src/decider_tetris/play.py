"""Play Tetris with a decision model: Python enumerates the landings, the model judges them.

  uv run python -m decider_tetris.play --decider http://127.0.0.1:8000 --games 1
  uv run python -m decider_tetris.play --games 5            # the heuristic baseline alone

Every landing the piece can reach becomes one option, worded as what it does to the stack. Ten
options go into one question and the group winners meet again, the way the browser agent's
ladder narrows a long list. Dellacherie's published heuristic scores the same landings on every
piece, so the model's pick can be compared against it without a second run.
"""

import argparse
import json
import random
import time
import urllib.request

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import COUNT, buried, current, phrase, placements, score
from decider_tetris.view import Window

# The wire contract takes up to 255 options, but the model was trained on at most ten candidates
# per example, and its own card reports the drop on full label sets (CLINC 151-way: 0.88 against
# 0.98 with ten sampled options). A group of ten is therefore the default, not a limit.
GROUP = 10
NAMES = "IOTSZJL"
# Three wordings of the same four goals, kept so the comparison can be rerun. Measured over five
# seeds, the numbers are in docs/tetris.md; an ordered list of the goals plays far worse than one
# short sentence, which is why the wording is a flag and not a guess.
WORDINGS = {
    "original": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, buries "
                 "no cell, keeps the stack low and keeps the surface even. A buried cell stays trapped "
                 "until every row above it is cleared, so burying cells loses the game."),
    "short": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, traps no "
              "cell, keeps the stack low and keeps the surface even. A trapped cell stays trapped until "
              "every row above it is cleared."),
    "gaps": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, traps no "
             "cell, keeps the stack low, keeps the surface flat and opens no one-wide gap. A trapped "
             "cell stays trapped until every row above it is cleared."),
    "ordered": ("Pick where to drop this piece in a game of Tetris, in this order of importance. First, "
                "trap no cell: a trapped cell stays trapped until every row above it is cleared. Second, "
                "clear rows. Third, keep the surface flat and open no one-wide gap, because only an I "
                "piece fills one. Fourth, prefer the lowest part of the stack, and prefer a landing "
                "against a wall over one in the middle."),
}


def ask(url, body):
    request = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        answers = json.load(response)["answers"]
    return answers, round((time.perf_counter() - started) * 1000)


def options(outcomes, before):
    """Landings the model can tell apart; identical wording is the same option to it."""
    chosen, phrases = [], {}
    for outcome in outcomes:
        text = phrase(outcome, before)
        if text not in phrases.values():
            chosen.append(outcome)
            phrases[outcome.id] = text
    return chosen, phrases


def survivors(outcomes, before):
    """Landings that trap no cell, when any exist.

    A trapped cell cannot be filled until every row above it clears, so a landing that traps one
    while a clean landing is on offer is never the right move. The browser agent keeps the same
    kind of red line in code: options that must never be taken are removed before the model sees
    them. Ranking the rest stays with the model.
    """
    clean = [outcome for outcome in outcomes if outcome.holes <= before.holes]
    return clean or outcomes


def undominated(outcomes):
    """Drop a landing that another one beats on every fact at once.

    This is not a weighting: no fact is traded against another. A landing that traps more cells,
    clears fewer rows, stands higher, leaves a rougher surface and opens more gaps than some other
    landing cannot be the right move under any preference, so the model never sees it.
    """
    def worse(a, b):
        return (a.holes >= b.holes and a.lines <= b.lines and a.height >= b.height
                and a.bumpiness >= b.bumpiness and a.wells >= b.wells
                and (a.holes, -a.lines, a.height, a.bumpiness, a.wells)
                != (b.holes, -b.lines, b.height, b.bumpiness, b.wells))
    return [a for a in outcomes if not any(worse(a, b) for b in outcomes)] or list(outcomes)


def stack_words(before):
    """The state the model reads: the stack in words, never numbers to compare."""
    return {
        "piece_fits_into": ("an empty board" if before.height == 0 else
                            "a low stack" if before.height <= 6 else
                            "a stack up to the middle" if before.height <= 11 else
                            "a high stack"),
        "buried_cells": ("none" if before.holes == 0 else
                         COUNT[before.holes] if before.holes < len(COUNT) else "several"),
        "surface": "even" if before.bumpiness <= 4 else "uneven",
    }


RATING = "Is this a good place to drop the piece?"
LEVELS = ["a losing placement", "a poor placement", "an acceptable placement", "a strong placement"]


def rate(url, state, outcomes, phrases, kind):
    """One question per landing, all in one call: every landing gets its own number, not one winner.

    The server scores each question in its own row, so the landings do not see each other. A
    choice question returns one winner and throws the runners-up away; this keeps the ranking.
    """
    question = ({"type": "noul", "instructions": f"{RATING} {{}}"} if kind == "noul" else
                {"type": "score", "criteria": LEVELS, "instructions": "How good is this placement? {}"})
    answers, ms = ask(url, {"state": state, "questions": {
        outcome.id: {**question, "instructions": question["instructions"].format(phrases[outcome.id])}
        for outcome in outcomes}})
    key = "noul" if kind == "noul" else "score"
    best = max(outcomes, key=lambda outcome: answers[outcome.id][key])
    return best, [ms], float(answers[best.id][key])


def board_rows(env):
    """The stack as text, to test whether the model reads a grid better than the facts."""
    return ["".join("#" if cell else "." for cell in row) for row in env.crop_padding(env.board.copy())]


ORDERS = {
    "shuffle": None,  # a fresh order every time, so no position keeps an advantage
    "enum": lambda outcome: (outcome.rotation, outcome.x),  # as enumerated: every upright landing leads
    "column": lambda outcome: (outcome.x, outcome.rotation),  # fixed and unrelated to how good a landing is
    "facts": lambda outcome: (outcome.holes, -outcome.lines, outcome.height, outcome.bumpiness),  # good first
}


def arrange(outcomes, rng, order, turn=0, turns=1, reverse=False):
    """Put the options in the order the model will read them.

    With `turns` above one the fixed orders are cycled: round `turn` starts the same sorted list at
    a different option. Every option then reads in a different position across the rounds, which
    cancels the pull of the leading position without a random draw, so the run stays reproducible.
    """
    if ORDERS[order] is None:
        pool = list(outcomes)
        rng.shuffle(pool)
        return pool
    pool = sorted(outcomes, key=ORDERS[order], reverse=reverse)
    shift = turn * len(pool) // max(turns, 1)
    return pool[shift:] + pool[:shift]


def settle(url, state, outcomes, phrases, rng, wording, group, order, threshold):
    """Ask once; when the top two options are close, ask again in the opposite order and add up.

    Measured over twelve orders on fourteen positions: a pick held through every order when the top
    two probabilities were 0.34 apart on average, and changed with the order when they were 0.12
    apart. So only the close calls need a second reading, and reading the list backwards is the
    cheapest way to move every option to a different position without a random draw.
    """
    first = {}
    pick, calls, confidence = judge(url, state, outcomes, phrases, rng, wording, group, first, order)
    ranked = sorted(first.values(), reverse=True)
    if len(ranked) < 2 or ranked[0] - ranked[1] >= threshold:
        return pick, calls, confidence
    second = {}
    _, again, _ = judge(url, state, outcomes, phrases, rng, wording, group, second, order, reverse=True)
    total = {name: first.get(name, 0.0) + second.get(name, 0.0) for name in set(first) | set(second)}
    best = max(total, key=total.get)
    return {outcome.id: outcome for outcome in outcomes}[best], calls + again, total[best] / 2


def vote(url, state, outcomes, phrases, rng, wording, group, rounds, order="shuffle"):
    """Ask the same question under several option orders and add the probabilities up.

    Measured on twelve positions with eight to eleven landings each, asking twelve orders, the pick
    changed with the order on eight of them, twice splitting six to six. The model returns a
    calibrated distribution, so adding the distributions uses more of its answer than a majority
    vote over the argmax does.
    """
    if rounds <= 1:
        return judge(url, state, outcomes, phrases, rng, wording, group, order=order)
    weight, latencies = {outcome.id: 0.0 for outcome in outcomes}, []
    for turn in range(rounds):
        pick, calls, _ = judge(url, state, outcomes, phrases, rng, wording, group, weight, order, turn, rounds)
        latencies += calls
    best = max(weight, key=weight.get)
    by_id = {outcome.id: outcome for outcome in outcomes}
    return by_id[best], latencies, weight[best] / rounds


def judge(url, state, outcomes, phrases, rng, wording, group=GROUP, weight=None, order="shuffle",
          turn=0, turns=1, reverse=False):
    """Ten options per question; the winners of each group meet until one landing is left.

    Landings are shuffled first. Enumerated in order, the first group holds only unrotated
    landings, and whichever option leads a group wins it more often than it should.
    """
    pool, latencies, picks = arrange(outcomes, rng, order, turn, turns, reverse), [], []
    while len(pool) > 1:
        groups = [pool[i:i + group] for i in range(0, len(pool), group)]
        answers, ms = ask(url, {"state": state, "questions": {
            f"g{n}": {"type": "choice", "instructions": WORDINGS[wording],
                      "criteria": {outcome.id: phrases[outcome.id] for outcome in group}}
            for n, group in enumerate(groups)}})
        latencies.append(ms)
        by_id = {outcome.id: outcome for outcome in pool}
        picks = [answers[f"g{n}"] for n in range(len(groups))]
        if weight is not None:
            for answer in picks:
                for name, probability in answer["probabilities"].items():
                    weight[name] = weight.get(name, 0.0) + probability
        pool = [by_id[pick["choice"]] for pick in picks]
    return pool[0], latencies, picks[-1]["confidence"] if picks else 1.0


def drop(env, outcome, window, notes, hold_ms):
    """Turn and move the piece to the chosen landing, then let it fall."""
    for _ in range(outcome.rotation):
        env.active_tetromino = env.rotate(env.active_tetromino, True)
    env.x, env.y = outcome.x, 0
    while not env.collision(env.active_tetromino, env.x, env.y + 1):
        if window:
            window.show(env, notes, hold_ms)
        env.y += 1
    return int(env.step(env.actions.hard_drop)[4]["lines_cleared"])  # numpy counts are not JSON


def play(env, url, kind, board, wording, group, clean, best, rounds, order, threshold, window, hold_ms,
         max_pieces, log, rng):
    """One game. Returns the pieces placed, the rows cleared and how often the baseline agreed."""
    placed = cleared = agreed = 0
    latencies = []
    while not env.game_over and placed < max_pieces:
        before = current(env)
        outcomes = placements(env)
        if not outcomes:
            break
        offered = survivors(outcomes, before) if clean else outcomes
        if best:
            offered = undominated(offered)
        forced = clean and all(outcome.holes > before.holes for outcome in outcomes)
        choices, phrases = options(offered, before)
        baseline = max(outcomes, key=score)
        piece = NAMES[env.active_tetromino.id - len(env.base_pixels)]
        state = {"piece": piece, **stack_words(before)}
        if board:
            state["board"] = board_rows(env)
        if url and kind == "choice":
            pick, calls, confidence = (
                settle(url, state, choices, phrases, rng, wording, group, order, threshold) if threshold > 0
                else vote(url, state, choices, phrases, rng, wording, group, rounds, order))
        elif url:
            pick, calls, confidence = rate(url, state, choices, phrases, kind)
        else:
            pick, calls, confidence = baseline, [], 1.0
        latencies += calls
        agreed += pick.id == baseline.id
        notes = [f"{cleared} rows", f"piece {piece}   placed {placed}",
                 f"{pick.id}   p={confidence:.2f}   {sum(calls)} ms   {len(choices)} options",
                 phrases.get(pick.id, phrase(pick, before)),
                 f"baseline {baseline.id}" + ("  (agrees)" if pick.id == baseline.id else "  (differs)")]
        lines = drop(env, pick, window, notes, hold_ms)
        # The env is the authority: a landing that lands elsewhere than simulated shows up here.
        landed = buried(env.crop_padding(env.board.copy()))
        if landed != pick.holes:
            print(json.dumps({"desync": pick.id, "piece": piece, "predicted_holes": pick.holes,
                              "actual_holes": landed}), flush=True)
        placed, cleared = placed + 1, cleared + lines
        if window:
            window.show(env, notes, hold_ms)
        if log:
            log.write(json.dumps({"piece": piece, "state": state, "options": len(choices),
                                  "forced_trap": bool(forced), "holes": before.holes,
                                  "height": before.height, "bumpiness": before.bumpiness,
                                  "baseline_offered": any(o.id == baseline.id for o in choices),
                                  "baseline_phrase": phrase(baseline, before),
                                  "choice": pick.id, "confidence": confidence, "latency_ms": calls,
                                  "phrase": phrases.get(pick.id), "baseline": baseline.id,
                                  "lines": int(lines), "rows_total": int(cleared)}) + "\n")
    return placed, cleared, agreed, latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", help="base URL of the decision model; omitted runs the heuristic alone")
    parser.add_argument("--rate", choices=("choice", "noul", "score"), default="choice",
                        help="choice runs a tournament; noul and score give every landing its own number")
    parser.add_argument("--wording", choices=tuple(WORDINGS), default="gaps")
    parser.add_argument("--group", type=int, default=GROUP, help="options per question; 255 asks them all at once")
    parser.add_argument("--order", choices=tuple(ORDERS), default="shuffle")
    parser.add_argument("--settle", type=float, default=0.0,
                        help="when the top two options are closer than this, read the list backwards and add up")
    parser.add_argument("--rounds", type=int, default=1,
                        help="ask each decision under this many option orders and add the probabilities")
    parser.add_argument("--best", action="store_true",
                        help="hide landings another landing beats on every fact at once")
    parser.add_argument("--clean", action="store_true",
                        help="hide landings that trap a cell while a clean landing exists")
    parser.add_argument("--board", action="store_true", help="also send the stack as a grid of text")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-pieces", type=int, default=200)
    parser.add_argument("--hold-ms", type=int, default=1, help="milliseconds per animation frame")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--log")
    args = parser.parse_args()
    env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
    window = None if args.no_window else Window()
    log = open(args.log, "w") if args.log else None
    try:
        for game in range(args.games):
            env.reset(seed=args.seed + game)
            placed, cleared, agreed, latencies = play(
                env, args.decider, args.rate, args.board, args.wording, args.group, args.clean, args.best,
                args.rounds, args.order, args.settle, window,
                args.hold_ms,
                args.max_pieces,
                log,
                random.Random(args.seed))
            print(json.dumps({"game": game, "rate": args.rate, "wording": args.wording, "group": args.group, "clean": args.clean, "best": args.best, "rounds": args.rounds, "order": args.order, "settle": args.settle, "seed": args.seed + game, "pieces": placed, "rows_cleared": cleared,
                              "baseline_agreement": round(agreed / max(placed, 1), 3),
                              "calls": len(latencies),
                              "median_call_ms": sorted(latencies)[len(latencies) // 2] if latencies else None}),
                  flush=True)
    finally:
        if window:
            window.close()
        if log:
            log.close()
        env.close()


if __name__ == "__main__":
    main()
