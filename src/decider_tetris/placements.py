"""Where the active piece can land, and what each landing does to the stack.

All arithmetic lives here, as in brotato/movement.py: the decision model never counts or
compares numbers. Every placement becomes one Choice criterion that carries its
consequences as words. `score` is a published hand-tuned heuristic (Dellacherie's
features), kept as an independent baseline to compare the model against.
"""

from dataclasses import dataclass

import numpy as np

COUNT = ["no", "one", "two", "three", "four"]


@dataclass(frozen=True)
class Outcome:
    """The stack after a placement. `id` is the option name the model answers with."""

    id: str
    rotation: int
    x: int  # left edge of the piece's box on the padded board, the position the player moves to
    lines: int
    holes: int  # empty cells with at least one filled cell above them
    height: int  # highest filled row, counted up from the floor
    bumpiness: int  # summed height difference between neighbouring columns
    landing: int  # height the piece comes to rest at
    row_transitions: int
    column_transitions: int
    wells: int
    eroded: int  # cleared rows times the piece cells that went with them
    where: str = ""  # which part of the stack it lands on, and which wall it touches


def surface(field):
    """Height of every column above the floor; 0 for an empty column."""
    filled = field > 0
    return np.where(filled.any(0), field.shape[0] - filled.argmax(0), 0)


def buried(field):
    """Empty cells that something covers, so they cannot be filled from above."""
    filled = field > 0
    return int((~filled & (np.cumsum(filled, 0) > 0)).sum())


def _transitions(field):
    """Filled/empty flips along each row and each column, walls counted as filled."""
    filled = (field > 0).astype(np.int8)
    rows = np.pad(filled, ((0, 0), (1, 1)), constant_values=1)
    columns = np.pad(filled, ((1, 0), (0, 0)), constant_values=0)
    columns = np.pad(columns, ((0, 1), (0, 0)), constant_values=1)
    return int(np.abs(np.diff(rows, axis=1)).sum()), int(np.abs(np.diff(columns, axis=0)).sum())


def _wells(field):
    """Depth of every one-wide gap, summed as 1+2+...+depth so deep wells weigh more."""
    heights = surface(field)
    walled = np.concatenate(([field.shape[0]], heights, [field.shape[0]]))
    total = 0
    for i in range(1, len(walled) - 1):
        depth = min(walled[i - 1], walled[i + 1]) - walled[i]
        if depth > 0:
            total += depth * (depth + 1) // 2
    return int(total)


def describe(field, identity, rotation, x, lines, landing, eroded, where=""):
    heights = surface(field)
    rows, columns = _transitions(field)
    return Outcome(id=identity, rotation=rotation, x=x, lines=int(lines), holes=buried(field),
                   height=int(heights.max()), bumpiness=int(np.abs(np.diff(heights)).sum()),
                   landing=int(landing), row_transitions=rows, column_transitions=columns,
                   wells=_wells(field), eroded=int(eroded), where=where)


def current(env):
    """The stack as it stands, so a placement's facts can be stated as a change."""
    return describe(env.crop_padding(env.board.copy()), "now", 0, 0, 0, 0, 0)


def _collides(board, matrix, x, y):
    height, width = matrix.shape
    if y + height > board.shape[0]:
        return True
    return bool(np.any(board[y:y + height, x:x + width][matrix > 0]))


def placements(env):
    """Every landing the active piece can reach, one per distinct resulting stack.

    A square box holds pieces whose 0 and 180 degree turns look the same, so two rotations can
    reach the same stack; the model must not see one landing twice.
    """
    board, first = env.board, env.active_tetromino.matrix
    heights = surface(env.crop_padding(board.copy()))
    found = {}
    for rotation in range(4):
        matrix = np.rot90(first, k=rotation)
        for x in range(board.shape[1] - matrix.shape[1] + 1):
            if _collides(board, matrix, x, 0):
                continue
            y = 0
            while not _collides(board, matrix, x, y + 1):
                y += 1
            field, outcome = _outcome(env, matrix, rotation, x, y, heights)
            found.setdefault(field.tobytes(), outcome)
    return list(found.values())


def _outcome(env, matrix, rotation, x, y, heights):
    board = env.board.copy()
    board[y:y + matrix.shape[0], x:x + matrix.shape[1]][matrix > 0] = matrix[matrix > 0]
    filled = ~(board == 0).any(1) & ~(board == 1).all(1)
    cells = int((matrix[filled[y:y + matrix.shape[0]]] > 0).sum())
    rows = np.nonzero(matrix.any(1))[0]
    columns = np.nonzero(matrix.any(0))[0] + x - env.padding
    board, lines = env.clear_filled_rows(board)
    field = env.crop_padding(board)
    return field, describe(field, f"r{rotation}c{columns[0]}", rotation, x, lines,
                         env.height - (y + rows[-1]), lines * cells, _where(columns, heights, env.width))


def _where(columns, heights, width):
    """Which part of the stack the piece lands on, in words the model can compare."""
    notes = []
    if heights.max() > heights.min():
        covered = heights[columns]
        if covered.min() == heights.min():
            notes.append("It fills the lowest part of the stack.")
        elif covered.max() == heights.max():
            notes.append("It piles onto the highest part of the stack.")
    if columns[0] == 0:
        notes.append("It sits against the left wall.")
    elif columns[-1] == width - 1:
        notes.append("It sits against the right wall.")
    return " ".join(notes)


def phrase(outcome, before):
    """The placement's consequences in words, short enough to serve as one option."""
    holes = outcome.holes - before.holes
    # A fixed band saturates: above twelve rows every landing read "The stack is high", exactly
    # where the choice decides the game. The change from the present stack keeps its edge.
    rise = outcome.height - before.height
    height = ("It lowers the top of the stack." if rise < 0 else
              "It keeps the top of the stack where it is." if rise == 0 else
              f"It raises the top of the stack by {COUNT[rise] if rise < len(COUNT) else 'several'} "
              f"row{'' if rise == 1 else 's'}.")
    if outcome.height >= 15:
        height += " The stack is close to the top."
    evenness = ("It flattens the surface." if outcome.bumpiness < before.bumpiness - 1 else
                "It leaves the surface as uneven as it is now." if outcome.bumpiness <= before.bumpiness + 1 else
                "It makes the surface more uneven.")
    wells = ("It opens a deep one-wide gap." if outcome.wells >= before.wells + 3 else
             "It opens a one-wide gap." if outcome.wells > before.wells else "")
    # The trapped cells lead: read last, they lost to the positive facts beside them.
    return " ".join(part for part in [
        "Traps nothing." if holes <= 0 else
        f"Traps {COUNT[holes] if holes < len(COUNT) else 'several'} "
        f"cell{'' if holes == 1 else 's'} that no piece can reach.",
        f"Clears {COUNT[outcome.lines]} row{'' if outcome.lines == 1 else 's'}.",
        outcome.where, height, evenness, wells,
    ] if part)


def score(outcome):
    """Dellacherie's hand-tuned evaluation, the baseline the model is compared against."""
    return (-4.500158825082766 * outcome.landing
            + 3.4181268101392694 * outcome.eroded
            - 3.2178882868487753 * outcome.row_transitions
            - 9.348695305445199 * outcome.column_transitions
            - 7.899265427351652 * outcome.holes
            - 3.3855972247263626 * outcome.wells)
