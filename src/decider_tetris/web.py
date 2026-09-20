"""Play the published react-tetris page with the decision model.

  uv run python -m decider_tetris.web --decider http://127.0.0.1:8000

The page stores its whole Redux state in localStorage, so the board, the piece in flight and
the score are read from the page itself; moves go back in as key events. The page is never
modified. The rules below mirror the page's own source (src/unit/const.js, src/unit/block.js,
src/unit/index.js): a placement is only offered if the page would accept it.
"""

import argparse
import base64
import json
import random
import time
import urllib.parse
import urllib.request

import numpy as np
from websockets.sync.client import connect

from decider_tetris.placements import describe, phrase, score, surface
from decider_tetris.play import judge, stack_words

PAGE = "https://chvin.github.io/react-tetris/"
STORAGE = "REACT_TETRIS"
WIDTH, HEIGHT = 10, 20
KEYS = {"left": 37, "rotate": 38, "right": 39, "down": 40, "space": 32}
# src/unit/const.js
SHAPES = {"I": [[1, 1, 1, 1]], "L": [[0, 0, 1], [1, 1, 1]], "J": [[1, 0, 0], [1, 1, 1]],
          "Z": [[1, 1, 0], [0, 1, 1]], "S": [[0, 1, 1], [1, 1, 0]], "O": [[1, 1], [1, 1]],
          "T": [[0, 1, 0], [1, 1, 1]]}
ORIGIN = {"I": [[-1, 1], [1, -1]], "L": [[0, 0]], "J": [[0, 0]], "Z": [[0, 0]], "S": [[0, 0]],
          "O": [[0, 0]], "T": [[0, 0], [1, 0], [-1, 1], [0, -1]]}


class Page:
    """One CDP connection to the open tab: read the state, send keys."""

    def __init__(self, cdp):
        tabs = json.load(urllib.request.urlopen(cdp + "/json"))
        tab = next(t for t in tabs if t["type"] == "page" and "react-tetris" in t["url"])
        self.socket = connect(tab["webSocketDebuggerUrl"], max_size=None)
        self.sent = 0

    def call(self, method, **params):
        self.sent += 1
        self.socket.send(json.dumps({"id": self.sent, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.sent:
                return message.get("result", {})

    def evaluate(self, expression):
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get("value")

    def state(self):
        """The page's own Redux state: btoa(encodeURIComponent(JSON)) under one localStorage key."""
        raw = self.evaluate(f"localStorage.getItem({STORAGE!r})")
        if not raw:
            return None
        return json.loads(urllib.parse.unquote(base64.b64decode(raw).decode()))

    def press(self, name, times=1, gap=0.05):
        for _ in range(times):
            for kind in ("rawKeyDown", "keyUp"):
                self.call("Input.dispatchKeyEvent", type=kind, windowsVirtualKeyCode=KEYS[name],
                          nativeVirtualKeyCode=KEYS[name])
            time.sleep(gap)

    def settle(self, expected, timeout=3.0):
        """Wait until the page records the locked board.

        src/unit/index.js stops recording while `lock` is true, so during a clearing animation
        localStorage still holds the board from before the drop. Acting on it drops a second
        piece onto a stale stack.
        """
        deadline, state = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            state = self.state()
            if state is not None and np.array_equal(np.array(state["matrix"]), expected):
                return state, True
            time.sleep(0.1)
        return state, False

    def close(self):
        self.socket.close()


def turn(shape):
    """src/unit/block.js rotate(): the page's own turn, not a numpy rotation."""
    width = len(shape[0])
    return [[row[width - 1 - index] for row in shape] for index in range(width)]


def fits(shape, y, x, matrix):
    """src/unit/index.js want(): the page refuses any move this rejects."""
    if x < 0 or x + len(shape[0]) > WIDTH:
        return False
    for row, cells in enumerate(shape):
        if y + row < 0:
            continue
        if y + row >= HEIGHT:
            return False
        for column, cell in enumerate(cells):
            if cell and matrix[y + row][x + column]:
                return False
    return True


def turns(piece):
    """Each distinct turn of the piece, with the sideways shift the page applies to it."""
    shape, shift, index, seen = SHAPES[piece], 0, 0, {}
    for count in range(4):
        key = json.dumps(shape)
        if key not in seen:
            seen[key] = (count, shape, shift)
        shift += ORIGIN[piece][index][1]
        index = (index + 1) % len(ORIGIN[piece])
        shape = turn(shape)
    return list(seen.values())


def landings(matrix, piece, heights):
    """Every landing the page would accept, described the same way as in the offline game."""
    field = np.array(matrix)
    found = {}
    for count, shape, shift in turns(piece):
        for x in range(WIDTH - len(shape[0]) + 1):
            y = -len(shape)
            while fits(shape, y + 1, x, matrix):
                y += 1
            after = field.copy()
            for row, cells in enumerate(shape):
                if y + row < 0:
                    continue  # a piece resting partly above the board keeps only its visible cells
                for offset, cell in enumerate(cells):
                    if cell:
                        after[y + row][x + offset] = 1
            full = after.all(1)
            cleared = int(full.sum())
            covered = [index for index, cells in enumerate(shape) if any(cells)]
            eroded = cleared * sum(sum(shape[row]) for row in covered if y + row >= 0 and full[y + row])
            after = np.vstack((np.zeros((cleared, WIDTH), dtype=after.dtype), after[~full]))
            spanned = [x + offset for offset in range(len(shape[0]))
                       if any(shape[row][offset] for row in range(len(shape)))]
            outcome = describe(after, f"r{count}c{spanned[0]}", count, x, cleared,
                               HEIGHT - (y + covered[-1]), eroded, where(spanned, heights))
            found.setdefault(after.tobytes(), (outcome, count, x, shift, after))
    return list(found.values())


def where(spanned, heights):
    notes = []
    if heights.max() > heights.min():
        covered = heights[spanned]
        if covered.min() == heights.min():
            notes.append("It fills the lowest part of the stack.")
        elif covered.max() == heights.max():
            notes.append("It piles onto the highest part of the stack.")
    if spanned[0] == 0:
        notes.append("It sits against the left wall.")
    elif spanned[-1] == WIDTH - 1:
        notes.append("It sits against the right wall.")
    return " ".join(notes)


def options(found, before):
    chosen, phrases = [], {}
    for outcome, count, x, shift, after in found:
        text = phrase(outcome, before)
        if text not in phrases.values():
            chosen.append(outcome)
            phrases[outcome.id] = text
    return chosen, phrases, {outcome.id: (count, x, shift, after) for outcome, count, x, shift, after in found}


def move(page, piece, turns_needed, target, current_x, shift):
    """Turn the piece, walk it to the target column, then drop it."""
    page.press("rotate", turns_needed)
    steps = target - (current_x + shift)
    page.press("right" if steps > 0 else "left", abs(steps))
    page.press("space")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", required=True)
    parser.add_argument("--cdp", default="http://127.0.0.1:9333")
    parser.add_argument("--pieces", type=int, default=100)
    parser.add_argument("--log")
    args = parser.parse_args()
    page = Page(args.cdp)
    log = open(args.log, "w") if args.log else None
    rng = random.Random(1)
    placed = 0
    try:
        while placed < args.pieces:
            state = page.state()
            if state is None:
                time.sleep(0.2)
                continue
            if state["cur"] is None:
                page.press("space")  # src/control/todo/space.js starts the game when no piece is in flight
                time.sleep(0.5)
                continue
            matrix, piece = state["matrix"], state["cur"]["type"]
            heights = surface(np.array(matrix))
            before = describe(np.array(matrix), "now", 0, 0, 0, 0, 0)
            found = landings(matrix, piece, heights)
            if not found:
                break
            choices, phrases, moves = options(found, before)
            baseline = max((outcome for outcome, *_ in found), key=score)
            pick, calls, confidence = judge(args.decider, {"piece": piece, **stack_words(before)},
                                            choices, phrases, rng)
            count, x, shift, expected = moves[pick.id]
            move(page, piece, count, x, state["cur"]["xy"][1], shift)
            placed += 1
            # The page is the authority: if its board never becomes ours, the turn or shift maths is wrong.
            landed, settled = page.settle(expected)
            desync = not settled
            print(json.dumps({"piece": piece, "points": state["points"], "choice": pick.id,
                              "baseline": baseline.id, "confidence": round(confidence, 3),
                              "ms": sum(calls), **({"desync": True} if desync else {})}), flush=True)
            if log:
                log.write(json.dumps({"piece": piece, "points": state["points"], "choice": pick.id,
                                      "baseline": baseline.id, "phrase": phrases[pick.id],
                                      "confidence": confidence, "latency_ms": calls,
                                      "desync": desync}) + "\n")
    finally:
        final = page.state() or {}
        print(json.dumps({"pieces": placed, "points": final.get("points"), "best": final.get("max")}))
        page.close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
