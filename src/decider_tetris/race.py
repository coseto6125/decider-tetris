"""Two games side by side in one window: the heuristic against the decision model.

  uv run python -m decider_tetris.race --decider http://127.0.0.1:8000 --seed 1

Both boards get the same seed, so they see the same pieces in the same order and the only
difference is who chooses the landing. The run ends when both boards are dead, or at the cap.
"""

import argparse
import json
import random
import threading
import time

import cv2
import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris  # noqa: F401  (registers the environment)

from decider_tetris.placements import current, phrase, placements, score
from decider_tetris.play import options, settle, stack_words, survivors
from decider_tetris.view import TEXT, frame

TITLE = "Dellacherie vs Decider-2B"


class Side:
    """One board, its player and its running tally."""

    def __init__(self, name, url, seed, wording, clean, rng, group=20, order="enum", threshold=0.15):
        self.name, self.url, self.wording, self.clean, self.rng = name, url, wording, clean, rng
        self.group, self.order, self.threshold = group, order, threshold
        self.env = gym.make("tetris_gymnasium/Tetris", gravity=False).unwrapped
        self.env.reset(seed=seed)
        self.placed = self.cleared = 0
        self.note = "ready"
        self.latencies = []
        # The window reads this snapshot instead of taking a lock: at 140 pieces a second the
        # heuristic holds its board almost continuously, and a locking renderer freezes.
        self.snapshot = self.env.board.copy()
        self.started, self.stopped = time.perf_counter(), None  # lockstep play never calls run()

    @property
    def alive(self):
        return not self.env.game_over

    def choose(self):
        """Pick a landing and turn the piece towards it; the fall is animated by the caller."""
        before = current(self.env)
        found = placements(self.env)
        if not found:
            self.env.game_over = True
            return None
        if self.url:
            choices, phrases = options(survivors(found, before) if self.clean else found, before)
            pick, calls, confidence = settle(self.url, {"piece": self.piece(), **stack_words(before)},
                                             choices, phrases, self.rng, self.wording, self.group,
                                             self.order, self.threshold)
            self.latencies += calls
            self.note = f"{pick.id}  p={confidence:.2f}  {sum(calls)} ms"
        else:
            pick = max(found, key=score)
            self.note = f"{pick.id}  weighted score"
        self.phrase = phrase(pick, before)
        for _ in range(pick.rotation):
            self.env.active_tetromino = self.env.rotate(self.env.active_tetromino, True)
        self.env.x, self.env.y = pick.x, 0
        return pick

    def piece(self):
        return "IOTSZJL"[self.env.active_tetromino.id - len(self.env.base_pixels)]

    def publish(self):
        """Refresh what the window draws; the window never touches the live board."""
        self.snapshot = self.env.project_tetromino() if self.env.active_tetromino is not None \
            else self.env.board.copy()

    def falling(self):
        return not self.env.collision(self.env.active_tetromino, self.env.x, self.env.y + 1)

    def commit(self):
        self.cleared += int(self.env.step(self.env.actions.hard_drop)[4]["lines_cleared"])
        self.placed += 1
        if not self.alive:
            self.stopped = time.perf_counter()

    def run(self, cap):
        """Play at full speed, independent of the other board."""
        self.started = time.perf_counter()

        while self.alive and self.placed < cap:
            if self.choose() is None:
                break
            self.commit()
            self.publish()
        self.stopped = time.perf_counter()

    def seconds(self):
        return (self.stopped or time.perf_counter()) - (self.started or time.perf_counter())

    def notes(self):
        elapsed = self.seconds()
        rate = f"{self.placed / elapsed:.1f}" if elapsed > 0 else "0.0"
        return [f"{self.cleared} rows", f"{self.placed} pieces   {elapsed:.0f}s   {rate}/s",
                "DEAD" if not self.alive else self.note, getattr(self, "phrase", "")]


def show(sides, hold_ms, live=False):
    if live:  # lockstep play has no worker thread to refresh the snapshot
        for side in sides:
            side.publish()
    panels = [frame(side.snapshot, side.notes(), side.name, side.env) for side in sides]
    board = np.hstack(panels)
    cv2.line(board, (panels[0].shape[1], 0), (panels[0].shape[1], board.shape[0]), TEXT, 2)
    cv2.imshow(TITLE, board)
    cv2.waitKey(hold_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decider", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-pieces", type=int, default=0, help="0 plays until both boards are dead")
    parser.add_argument("--rows-per-frame", type=int, default=6,
                        help="rows a piece drops between drawn frames; drawing costs more than deciding")
    parser.add_argument("--frame-ms", type=int, default=1, help="pause on each drawn frame")
    parser.add_argument("--wording", default="gaps")
    parser.add_argument("--hold-ms", type=int, default=12)
    parser.add_argument("--plain", action="store_true", help="drop the no-trap red line from the model's side")
    parser.add_argument("--free", action="store_true",
                        help="each board plays at its own speed instead of one piece each in turn")
    args = parser.parse_args()
    sides = [Side("Dellacherie heuristic", None, args.seed, args.wording, False, random.Random(args.seed)),
             Side("Decider-2B", args.decider, args.seed, args.wording, not args.plain, random.Random(args.seed))]
    cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE)
    started, first_dead = time.perf_counter(), None
    try:
        if args.free:
            threads = [threading.Thread(target=side.run, args=(args.max_pieces,), daemon=True) for side in sides]
            for thread in threads:
                thread.start()
            while any(thread.is_alive() for thread in threads):
                show(sides, 30)
                if first_dead is None and any(not side.alive for side in sides):
                    first_dead = next(side.name for side in sides if not side.alive)
                    print(json.dumps({"first_dead": first_dead, "after_seconds": round(time.perf_counter() - started),
                                      "at": {side.name: [side.placed, side.cleared] for side in sides}}), flush=True)
            for thread in threads:
                thread.join()
            raise SystemExit(report(sides, args.seed, started))

        while any(side.alive for side in sides) and (not args.max_pieces
                                                     or max(side.placed for side in sides) < args.max_pieces):
            for side in sides:
                if side.alive:
                    side.choose()
            while any(side.alive and side.falling() for side in sides):
                for side in sides:
                    for _ in range(args.rows_per_frame):
                        if side.alive and side.falling():
                            side.env.y += 1
                show(sides, args.frame_ms, live=True)  # both pieces fall at the same visible speed
            for side in sides:
                if side.alive:
                    side.commit()
            show(sides, args.hold_ms, live=True)
            if first_dead is None and any(not side.alive for side in sides):
                first_dead = next(side.name for side in sides if not side.alive)
                print(json.dumps({"first_dead": first_dead,
                                  "at": {side.name: [side.placed, side.cleared] for side in sides}}), flush=True)
    finally:
        cv2.destroyWindow(TITLE)
        cv2.waitKey(1)
        print(report(sides, args.seed, started, first_dead))


def report(sides, seed, started, first_dead=None):
    return json.dumps({"seed": seed, "seconds": round(time.perf_counter() - started), "first_dead": first_dead,
                       **{side.name: {"pieces": side.placed, "rows": side.cleared, "alive": side.alive,
                                      "seconds": round(side.seconds(), 1),
                                      "pieces_per_second": round(side.placed / max(side.seconds(), 1e-9), 2)}
                          for side in sides}}, indent=1)


if __name__ == "__main__":
    main()
