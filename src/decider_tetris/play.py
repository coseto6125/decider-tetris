"""Play Tetris with a decision model: Python enumerates the landings, the model judges them.

  uv run python -m decider_tetris.play --decider http://127.0.0.1:8000 --games 1
  uv run python -m decider_tetris.play --games 5            # the heuristic baseline alone

Every landing the piece can reach becomes one option, worded as what it does to the stack. Ten
options go into one question and the group winners meet again, the way the browser agent's
ladder narrows a long list. Dellacherie's published heuristic scores the same landings on every
piece, so the model's pick can be compared against it without a second run.
"""

import argparse
import collections
import json
import random
import time
import urllib.request

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import (COUNT, buried, current, doomed, phrase, placements,
                                            score, settled)
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
    # The costly disagreements share one shape: the heuristic's landing sat against a wall and the
    # model's did not. The option already says so; only the instructions never said it was good.
    "walls": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, traps no "
              "cell, keeps the stack low, keeps the surface flat, opens no one-wide gap and sits against a "
              "wall rather than in the middle. A trapped cell stays trapped until every row above it is "
              "cleared."),
    # The options have always said "It opens a one-wide gap" and "It fills in a one-wide gap", and
    # the instructions listed the gap beside five other goals without saying what it costs. The
    # game that this was written for died with three one-wide channels open the height of the
    # board, each waiting for its own I piece. This is the same shape of fix as the wall clause:
    # the fact was already in the option, the reason for it was in no sentence the model read.
    "wells": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, traps no "
              "cell, keeps the stack low, keeps the surface flat, opens no one-wide gap and sits against a "
              "wall rather than in the middle. A trapped cell stays trapped until every row above it is "
              "cleared. Only an I piece fills a one-wide gap and one arrives every seventh piece, so a "
              "second open gap costs more than a high stack, and filling one in is worth more than "
              "clearing a row."),
    # Cloning the board at each disagreement showed the heuristic taking a landing that buries one
    # cell to clear a row, which the red line had been hiding. This says when that trade is on.
    "trade": ("Pick where to drop this piece in a game of Tetris. A good placement clears rows, traps no "
              "cell, keeps the stack low, keeps the surface flat, opens no one-wide gap and sits against a "
              "wall rather than in the middle. A trapped cell stays trapped until every row above it is "
              "cleared, so trapping one is only worth it when the same landing clears a row."),
    # The same clauses as `walls`, reordered. High up the stack dies by rising: through the fatal
    # climb of the 14338-row game every piece still had a clean landing while the top went from 9
    # rows to 19. Six goals listed flat never said that clearing outranks the rest once the stack
    # is high, and a measured rule of this project is that rewriting a sentence gains where adding
    # one loses.
    "urgent": ("Pick where to drop this piece in a game of Tetris. The stack is high, so clearing a row is "
               "worth more than everything else: take the landing that clears the most rows. When no "
               "landing clears, keep the stack low, trap no cell, keep the surface flat, open no one-wide "
               "gap and sit against a wall rather than in the middle. A trapped cell stays trapped until "
               "every row above it is cleared."),
    # The same clauses again, with the top of the stack first and clearing second. Which of the two
    # orders is right is a measurement, not a guess: on the boards recorded before each death a
    # clearing landing survived the red lines in only 10 to 20 per cent of the moments above eight
    # rows, while a landing that does not raise the top survived in 67 to 89 per cent. The one the
    # model can almost always act on may beat the one that is worth more when it is there.
    "holding": ("Pick where to drop this piece in a game of Tetris. The stack is high, so a landing that "
                "does not raise the top of the stack is worth more than everything else. Clearing a row "
                "comes next. Then trap no cell, keep the surface flat, open no one-wide gap and sit "
                "against a wall rather than in the middle. A trapped cell stays trapped until every row "
                "above it is cleared."),
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


def options(outcomes, before, grade=False, fills=False, count=False, depth=False, blocks=False, rest=False, terse=False, where=False, skip=()):
    """Landings the model can tell apart; identical wording is the same option to it."""
    chosen, phrases = [], {}
    for outcome in outcomes:
        text = phrase(outcome, before, grade, fills, count, depth, blocks, rest, terse, where, skip)
        if text not in phrases.values():
            chosen.append(outcome)
            phrases[outcome.id] = text
    return chosen, phrases


def survivors(outcomes, before, trading=0, least=False, relax=0, gap="off", high=0):
    """Landings that trap no cell, when any exist; with `least`, the ones that trap fewest.

    With `trading` a landing that clears at least that many rows survives even though it traps: the
    heuristic takes that trade, and hiding it cost rows at four of the fourteen costly
    disagreements measured.

    A trapped cell cannot be filled until every row above it clears, so a landing that traps one
    while a clean landing is on offer is never the right move. The browser agent keeps the same
    kind of red line in code: options that must never be taken are removed before the model sees
    them. Ranking the rest stays with the model.
    """
    # `trading` at one row was measured and lost: on this board almost every burying landing clears
    # something, so it relaxes the line everywhere. At two rows it fires on the trade the attributed
    # disagreements kept showing, where the heuristic buries one cell to empty a one-wide gap and
    # takes two rows out of the stack with the same piece.
    clean = [outcome for outcome in outcomes
             if outcome.holes <= before.holes or (trading and outcome.lines >= trading)]
    if clean:
        offered = clean
        # A red line that leaves one option is the code playing, not the model. Measured on a death
        # at 8520 pieces: over the last twelve landings it hid 33 of 34, and the one it left raised
        # the stack two rows and opened a deep gap, twelve times over, until the stack reached the
        # top. Above a bad surface, burying one cell often beats piling two more rows, so when the
        # clean landings run out the cheapest dirty ones come back and the model judges the trade.
        # Relaxing this line at any height cost thousands of rows: 288 against 5893 on seed 6. The
        # gain it was after is only in the endgame, where rebuilding the probe over the last twelve
        # landings of a death found that every landing surviving longer than the offered ones was
        # one this line had hidden. `high` keeps the relaxation for that regime alone.
        if relax and len(clean) < relax and before.height >= high:
            dirty = [outcome for outcome in outcomes if outcome.holes > before.holes]
            if dirty:
                fewest = min(outcome.holes for outcome in dirty)
                offered = clean + [outcome for outcome in dirty if outcome.holes == fewest]
    elif least:
        # Nothing clean is left, so the piece must bury something. One piece laid across the mouth
        # of a twelve-deep one-wide gap buries the whole column: the measured games ended on exactly
        # that, holes going from twelve to twenty-five on one Z while landings that buried one cell
        # were on offer. How many cells a landing buries is arithmetic, so the count decides and the
        # model ranks what is left. A landing that clears a row stays: the row takes its cells away.
        fewest = min(outcome.holes for outcome in outcomes)
        offered = [outcome for outcome in outcomes if outcome.holes == fewest or outcome.lines > 0]
    else:
        offered = list(outcomes)
    # A one-wide gap three rows deep is filled by an I piece and nothing else, so one comes along
    # every seventh piece: leaving one is close to burying the cells under it. The count decides and
    # `any` only asks whether a landing avoids leaving one, and stands aside when none of them can:
    # 5893 rows. `fewest` keeps the ones leaving the fewest, which also acts on a board that already
    # carries two, and cost half the game: 2267 rows. Forcing a deep gap to be filled buries cells
    # and cuts the option list, and the code plays instead of the model, the same way relaxing the
    # first red line did.
    #
    # This runs inside the buried-cell rule and never outside it. Run outside, it handed back
    # landings that bury cells whenever every clean landing left a deep gap, and the buried cells
    # went from 0.08 to 3.77 a piece over the first 431 pieces.
    if gap == "any":
        shallow = [outcome for outcome in offered if outcome.well_depth < 3]
        if shallow:
            offered = shallow
    elif gap == "fewest":
        fewest = min(outcome.deep_gaps for outcome in offered)
        offered = [outcome for outcome in offered if outcome.deep_gaps == fewest]
    return offered


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


def stack_words(before, grade=False):
    """The state the model reads: the stack in words, never numbers to compare."""
    # A bumpiness of four was the threshold when the stack rarely passed it. Over a long game the
    # surface sits at ten, so the one word "uneven" is true of every position the game is decided
    # in; three grades keep it informative at the bumpiness the run actually reaches.
    if grade:
        return {
            "piece_fits_into": ("an empty board" if before.height == 0 else
                                "a low stack" if before.height <= 6 else
                                "a stack up to the middle" if before.height <= 11 else
                                "a high stack"),
            "buried_cells": ("none" if before.holes == 0 else
                             COUNT[before.holes] if before.holes < len(COUNT) else "several"),
            "surface": ("flat" if before.bumpiness <= 4 else
                        "a little uneven" if before.bumpiness <= 9 else
                        "jagged" if before.bumpiness <= 15 else "very jagged"),
        }
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
    "enum_down": lambda outcome: (-outcome.rotation, -outcome.x),  # the same list read from the other end
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


def settle(url, state, outcomes, phrases, rng, wording, group, order, threshold, flip=False):
    """Ask once; when the top two options are close, ask again in the opposite order and add up.

    Measured over twelve orders on fourteen positions: a pick held through every order when the top
    two probabilities were 0.34 apart on average, and changed with the order when they were 0.12
    apart. So only the close calls need a second reading, and reading the list backwards is the
    cheapest way to move every option to a different position without a random draw.
    """
    first = {}
    pick, calls, confidence = judge(url, state, outcomes, phrases, rng, wording, group, first, order,
                                    reverse=flip)
    ranked = sorted(first.values(), reverse=True)
    if len(ranked) < 2 or ranked[0] - ranked[1] >= threshold:
        return pick, calls, confidence
    second = {}
    _, again, _ = judge(url, state, outcomes, phrases, rng, wording, group, second, order, reverse=not flip)
    # Two landings can reach the same total. `max` then keeps whichever the dict yields first, so
    # the totals are built in the enumeration order of the landings: a set of names would order them
    # by string hash, which is randomised per process, and two runs of one seed would part company on
    # the first tie. That is how one seed produced 3894, 1863 and 322 rows under the same flags.
    total = {outcome.id: first.get(outcome.id, 0.0) + second.get(outcome.id, 0.0) for outcome in outcomes}
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


def play(env, args, window, log, rng):
    """One game. Returns the pieces placed, the rows cleared and how often the baseline agreed.

    Every flag reaches this through `args`: the wordings and red lines are twelve switches now, and
    a positional chain that long dropped one silently the first time it grew.
    """
    url, kind, threshold = args.decider, args.rate, args.settle
    # The switches that change the option text travel together, into `options` and into `phrase`.
    words = dict(grade=args.grade, fills=args.fills, count=args.count,
                 blocks=args.blocks, rest=args.rest, terse=args.terse, where=args.where)
    placed = cleared = agreed = milestone = 0
    latencies, opened = [], time.perf_counter()
    # The last moves before a death are the only ones that explain it, so they are kept in full:
    # every option the model was offered, what it picked and what the heuristic would have picked.
    recent = collections.deque(maxlen=25)
    while not env.game_over and placed < args.max_pieces:
        before = current(env)
        outcomes = placements(env)
        if not outcomes:
            break
        offered = (survivors(outcomes, before, args.trading, args.least, args.relax, args.gap, args.high)
                   if args.clean else outcomes)
        if args.best:
            offered = undominated(offered)
        # One piece of foresight, only where the games end. Below the gate the board is low and
        # almost every landing leaves the next piece a clean move, so the check would cost time and
        # remove nothing; the deaths all begin between ten and fifteen rows, where a quarter of the
        # pieces already have one landing left.
        if args.foresee and len(offered) > 1 and before.height >= args.foresee:
            next_piece = env.tetrominoes[env.queue.get_queue()[0]].matrix
            open_ended = [outcome for outcome in offered
                          if not doomed(env, settled(env, outcome), next_piece)]
            offered = open_ended or offered
        forced = args.clean and all(outcome.holes > before.holes for outcome in outcomes)
        # Stating the depth of the gap a landing leaves on every piece cost five points of
        # agreement with the baseline: one more sentence on an option that is already six long.
        # `crowded` spends it only where the games end, on a board already carrying two gaps
        # that each need their own I piece.
        words["depth"] = args.depth == "always" or (args.depth == "crowded" and before.deep_gaps >= 2)
        choices, phrases = options(offered, before, **words)
        baseline = max(outcomes, key=score)
        piece = NAMES[env.active_tetromino.id - len(env.base_pixels)]
        state = {"piece": piece, **stack_words(before, args.grade)}
        if args.board:
            state["board"] = board_rows(env)
        # Alternating the reading direction costs nothing and cancels the pull of the leading
        # option over a game: enumerated low column first, the model used columns 8 and 9 five
        # points less often than the heuristic did.
        speaks = args.above if args.urgent and before.height >= args.urgent else args.wording
        flip = bool(args.alternate and placed % 2)
        if url and kind == "choice":
            pick, calls, confidence = (
                settle(url, state, choices, phrases, rng, speaks, args.group, args.order,
                       threshold, flip)
                if threshold > 0
                else vote(url, state, choices, phrases, rng, speaks, args.group, args.rounds,
                          args.order))
        elif url:
            pick, calls, confidence = rate(url, state, choices, phrases, kind)
        else:
            pick, calls, confidence = baseline, [], 1.0
        latencies += calls
        agreed += pick.id == baseline.id
        notes = [f"{cleared} rows", f"piece {piece}   placed {placed}",
                 f"{pick.id}   p={confidence:.2f}   {sum(calls)} ms   {len(choices)} options",
                 phrases.get(pick.id, phrase(pick, before, **words)),
                 f"baseline {baseline.id}" + ("  (agrees)" if pick.id == baseline.id else "  (differs)")]
        recent.append({"piece": placed, "kind": piece, "board": board_rows(env),
                       "holes": before.holes, "height": before.height,
                       "forced_trap": bool(forced), "picked": pick.id,
                       "options": {outcome.id: phrases[outcome.id] for outcome in choices},
                       "hidden": len(outcomes) - len(choices),
                       "heuristic": baseline.id, "heuristic_phrase": phrase(baseline, before, **words)})
        lines = drop(env, pick, window, notes, args.hold_ms)
        # The env is the authority: a landing that lands elsewhere than simulated shows up here.
        landed = buried(env.crop_padding(env.board.copy()))
        if landed != pick.holes:
            print(json.dumps({"desync": pick.id, "piece": piece, "predicted_holes": pick.holes,
                              "actual_holes": landed}), flush=True)
        placed, cleared = placed + 1, cleared + lines
        if cleared // 1000 > milestone:
            milestone = cleared // 1000
            print(json.dumps({"rows_total": cleared, "pieces": placed,
                              "minutes": round((time.perf_counter() - opened) / 60, 1)}), flush=True)
        if window:
            window.show(env, notes, args.hold_ms)
        if log:
            log.write(json.dumps({"piece": piece, "state": state, "options": len(choices),
                                  "forced_trap": bool(forced), "holes": before.holes,
                                  "height": before.height, "bumpiness": before.bumpiness,
                                  "baseline_offered": any(o.id == baseline.id for o in choices),
                                  "baseline_phrase": phrase(baseline, before, **words),
                                  "choice": pick.id, "confidence": confidence, "latency_ms": calls,
                                  "phrase": phrases.get(pick.id), "baseline": baseline.id,
                                  "lines": int(lines), "rows_total": int(cleared)}) + "\n")
    if args.postmortem and env.game_over:
        with open(args.postmortem, "w") as handle:
            json.dump({"pieces": placed, "rows": cleared, "final_board": board_rows(env),
                       "recent": list(recent)}, handle, indent=1)
    return placed, cleared, agreed, latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", help="base URL of the decision model; omitted runs the heuristic alone")
    parser.add_argument("--rate", choices=("choice", "noul", "score"), default="choice",
                        help="choice runs a tournament; noul and score give every landing its own number")
    parser.add_argument("--wording", choices=tuple(WORDINGS), default="gaps")
    parser.add_argument("--urgent", type=int, default=0, metavar="ROWS",
                        help="above this stack height, read the `--above` instructions instead")
    parser.add_argument("--above", choices=tuple(WORDINGS), default="urgent",
                        help="the instructions to read once the stack passes --urgent")
    parser.add_argument("--group", type=int, default=GROUP, help="options per question; 255 asks them all at once")
    parser.add_argument("--order", choices=tuple(ORDERS), default="shuffle")
    parser.add_argument("--alternate", action="store_true",
                        help="read the options forwards on one piece and backwards on the next")
    parser.add_argument("--settle", type=float, default=0.0,
                        help="when the top two options are closer than this, read the list backwards and add up")
    parser.add_argument("--rounds", type=int, default=1,
                        help="ask each decision under this many option orders and add the probabilities")
    parser.add_argument("--trading", type=int, default=0, metavar="ROWS",
                        help="let a landing that traps a cell through when it clears at least this many rows; "
                             "one row measured worse, two is the trade the attribution kept showing")
    parser.add_argument("--best", action="store_true",
                        help="hide landings another landing beats on every fact at once")
    parser.add_argument("--clean", action="store_true",
                        help="hide landings that trap a cell while a clean landing exists")
    parser.add_argument("--grade", action="store_true",
                        help="say how much flatter or rougher a landing leaves the surface, not just which way")
    parser.add_argument("--fills", action="store_true",
                        help="say when a landing fills in a one-wide gap, not only when it opens one")
    parser.add_argument("--count", action="store_true",
                        help="count trapped cells past four instead of calling them several")
    parser.add_argument("--depth", choices=("off", "always", "crowded"), default="off",
                        help="state how deep the one-wide gap a landing leaves is: always, or only once the "
                             "stack already carries two gaps that each need an I piece")
    parser.add_argument("--least", action="store_true",
                        help="when every landing buries a cell, offer only those that bury the fewest")
    parser.add_argument("--relax", type=int, default=0,
                        help="when fewer clean landings than this are left, also offer the cheapest dirty ones")
    parser.add_argument("--blocks", action="store_true",
                        help="say whether a landing joins the stack together or leaves it in separate towers")
    parser.add_argument("--rest", action="store_true",
                        help="say how far below the top of the stack the piece comes to rest")
    parser.add_argument("--terse", action="store_true",
                        help="leave out \"Traps nothing\" and \"Clears no rows\", which nearly every option carries")
    parser.add_argument("--gap", choices=("off", "any", "fewest"), default="off",
                        help="any: hide a landing leaving a gap three rows deep while another does not "
                             "(5893 rows); fewest: keep those leaving the fewest such gaps (2267)")
    parser.add_argument("--where", action="store_true",
                        help="say how many rows above the lowest part of the stack a landing sits, not only whether it is the lowest")
    parser.add_argument("--foresee", type=int, default=0, metavar="ROWS",
                        help="above this stack height, drop landings that leave the next piece no "
                             "landing that buries nothing")
    parser.add_argument("--high", type=int, default=0, metavar="ROWS",
                        help="only let --relax act once the stack reaches this height")
    parser.add_argument("--board", action="store_true", help="also send the stack as a grid of text")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-pieces", type=int, default=200)
    parser.add_argument("--hold-ms", type=int, default=1, help="milliseconds per animation frame")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--log")
    parser.add_argument("--postmortem", help="on a death, write the last 25 moves and the final board here")
    args = parser.parse_args()
    # The environment's randomizer reads `if seed and seed > 0`, so seed 0 is taken as no seed at
    # all and the bag is drawn from the operating system instead. The run then looks reproducible
    # and is not: two processes play different pieces from the first one. Refuse it at the door
    # rather than let a comparison rest on it.
    if args.seed < 1:
        parser.error("--seed must be 1 or more: the environment ignores seed 0 and draws at random")
    env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
    window = None if args.no_window else Window()
    log = open(args.log, "w") if args.log else None
    try:
        for game in range(args.games):
            env.reset(seed=args.seed + game)
            placed, cleared, agreed, latencies = play(env, args, window, log, random.Random(args.seed))
            # Every switch is reported, so a result line says which run produced it without a note.
            settings = {name: value for name, value in vars(args).items()
                        if name not in ("decider", "games", "hold_ms", "no_window", "log", "postmortem")}
            print(json.dumps({"game": game, **settings, "seed": args.seed + game,
                              "pieces": placed, "rows_cleared": cleared,
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
