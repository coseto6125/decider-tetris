"""The board as a window, with the decision that produced it written beside it.

The package's own `render_mode="human"` casts to `np.integer`, which numpy 2 rejects, so the
board is drawn here instead. Drawing it ourselves also puts the model's pick, its probability
and its latency on screen, which is what a run needs to be watchable.
"""

import cv2
import numpy as np

CELL = 26
PANEL = 430
GRID = (60, 60, 60)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
TITLE = "Tetris on Decider-2B"


def board_image(env, board=None):
    """The playfield including the piece in flight, one pixel per cell."""
    colors = np.array([pixel.color_rgb for pixel in env.pixels], dtype=np.uint8)
    if board is None:
        board = env.project_tetromino() if env.active_tetromino is not None else env.board.copy()
    return colors[env.crop_padding(board)][:, :, ::-1]  # OpenCV windows are BGR


def frame(env, notes, title=TITLE, board_owner=None):
    """The board beside a panel: the first note is the headline, the rest are detail lines.

    `env` may be a board array taken as a snapshot, in which case `board_owner` supplies the
    palette and the padding.
    """
    cells = board_image(board_owner, env) if board_owner is not None else board_image(env)
    field = np.repeat(np.repeat(cells, CELL, 0), CELL, 1)
    for y in range(0, field.shape[0] + 1, CELL):
        cv2.line(field, (0, y), (field.shape[1], y), GRID, 1)
    for x in range(0, field.shape[1] + 1, CELL):
        cv2.line(field, (x, 0), (x, field.shape[0]), GRID, 1)
    panel = np.zeros((field.shape[0], PANEL, 3), dtype=np.uint8)
    cv2.putText(panel, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
    y = 90
    for index, note in enumerate(notes):
        if index == 0:  # the headline carries the score, readable from across the room
            cv2.putText(panel, note, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, TEXT, 3, cv2.LINE_AA)
            y += 46
            continue
        while note:  # a long phrase wraps rather than running off the panel
            cut = len(note) if len(note) <= 46 else (note.rfind(" ", 0, 46) + 1 or 46)
            cv2.putText(panel, note[:cut], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        TEXT if index == 1 else DIM, 1, cv2.LINE_AA)
            note, y = note[cut:], y + 22
        y += 6
    return np.hstack((field, panel))


class Window:
    """One OpenCV window; `close` releases it even when the run ends in an exception."""

    def __init__(self, title=TITLE):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, env, notes, hold_ms=1):
        cv2.imshow(self.title, frame(env, notes))
        cv2.waitKey(hold_ms)

    def close(self):
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)
