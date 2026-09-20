"""Where the active piece can land, and what each landing does to the stack.

All arithmetic lives here, as in brotato/movement.py: the decision model never counts or
compares numbers. Every placement becomes one Choice criterion that carries its
consequences as words. `score` is a published hand-tuned heuristic (Dellacherie's
features), kept as an independent baseline to compare the model against.
"""

from dataclasses import dataclass

import numpy as np

COUNT = ["no", "one", "two", "three", "four"]
# "several" covered everything above four, so capping a twelve-deep well with an S piece read the
# same as burying five cells. One piece laid across the mouth of a deep one-wide well buries the
# whole column at once, and that is how the measured games ended: holes went 12 to 25 on one Z.
MANY = COUNT + ["five", "six", "seven", "eight", "nine", "ten"]
# Measured as no better (986 rows against 987 over five seeds), kept because it is the only fact
# that speaks about what a landing gives up rather than what it does.
RESERVED = False


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
    splits: int  # rows of the stack cut into separate pieces, one count per extra piece
    well_depth: int  # rows in the deepest one-wide gap, the one only an I piece fills
    deep_gaps: int  # one-wide gaps three or more rows deep, each needing its own I piece
    eroded: int  # cleared rows times the piece cells that went with them
    where: str = ""  # which part of the stack it lands on, and which wall it touches
    above_lowest: int = 0  # rows between the lowest column it covers and the lowest of all
    wall: str = ""  # the sentence naming the wall it touches, kept apart so `where` can be reworded


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
    """Depth of every one-wide gap, summed as 1+2+...+depth so deep wells weigh more.

    The deepest single gap comes back as well, because one piece laid across its mouth buries the
    whole column, and so does the count of gaps three or more deep: a stack survives one of those,
    waiting for its I piece, and dies around two.
    """
    heights = surface(field)
    walled = np.concatenate(([field.shape[0]], heights, [field.shape[0]]))
    total = deepest = deep = 0
    for i in range(1, len(walled) - 1):
        depth = min(walled[i - 1], walled[i + 1]) - walled[i]
        if depth > 0:
            total += depth * (depth + 1) // 2
            deepest = max(deepest, depth)
            deep += depth >= 3
    return int(total), int(deepest), int(deep)


def splits(field):
    """Valleys: how often a row of the stack is cut into separate pieces.

    A row filled in one run counts nothing, a row filled in two runs counts one. Summed over the
    stack this is the number of towers standing apart, which a sum of neighbour height differences
    cannot see: two towers three columns apart and one long slope give the same bumpiness. Counting
    the runs and not the transitions keeps the number free of the stack's height, because the empty
    rows above the stack would otherwise move it by two per row.
    """
    filled = field > 0
    starts = filled & ~np.pad(filled, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    return int(np.clip(starts.sum(1) - 1, 0, None).sum())


def reserved(heights, floor):
    """The one column being kept for an I piece: three or more lower than both its neighbours.

    Only a single such column counts. Two deep columns cannot both be served by the next I, so
    filling one of them is not the loss this fact is about.
    """
    walled = np.concatenate(([floor], heights, [floor]))
    deep = np.nonzero(np.minimum(walled[:-2], walled[2:]) - heights >= 3)[0]
    return int(deep[0]) if len(deep) == 1 else None


def describe(field, identity, rotation, x, lines, landing, eroded, where="", above_lowest=0, wall=""):
    heights = surface(field)
    rows, columns = _transitions(field)
    total, deepest, deep = _wells(field)
    return Outcome(id=identity, rotation=rotation, x=x, lines=int(lines), holes=buried(field),
                   height=int(heights.max()), bumpiness=int(np.abs(np.diff(heights)).sum()),
                   landing=int(landing), row_transitions=rows, column_transitions=columns,
                   wells=total, well_depth=deepest, deep_gaps=deep, splits=splits(field),
                   eroded=int(eroded), where=where, above_lowest=int(above_lowest), wall=wall)


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


def settled(env, outcome):
    """The stack as this landing leaves it, rows cleared and the padding taken off."""
    matrix = np.rot90(env.active_tetromino.matrix, k=outcome.rotation)
    board = env.board.copy()
    y = 0
    while not _collides(board, matrix, outcome.x, y + 1):
        y += 1
    board[y:y + matrix.shape[0], outcome.x:outcome.x + matrix.shape[1]][matrix > 0] = matrix[matrix > 0]
    board, _ = env.clear_filled_rows(board)
    return env.crop_padding(board)


def doomed(env, field, matrix):
    """True when the next piece cannot spawn on this stack, or every landing it reaches buries a cell.

    The queue shows the next piece, so this is foresight the player already has. Whether a landing
    leaves the next piece a clean move of its own is arithmetic, so it stays here and the sentences
    the model reads do not change. The measured deaths start where the red lines leave one landing
    and the stack ratchets up two rows at a time; a landing that dooms the next piece is the first
    step of that ratchet.
    """
    board = np.ones_like(env.board)
    board[:env.height, env.padding:env.padding + env.width] = field
    # The stack can end the game before the piece is ever moved: the environment spawns it
    # unrotated at the centre and the game is over if it does not fit there. Searching the landings
    # alone misses that, and misses it in the shape this check exists to catch — a tower in the
    # middle with clean room at the sides answered "not doomed" for five of the seven pieces.
    if _collides(board, matrix, board.shape[1] // 2 - matrix.shape[0] // 2, 0):
        return True
    covered = buried(field)
    for rotation in range(4):
        piece = np.rot90(matrix, k=rotation)
        for x in range(board.shape[1] - piece.shape[1] + 1):
            if _collides(board, piece, x, 0):
                continue
            y = 0
            while not _collides(board, piece, x, y + 1):
                y += 1
            after = board.copy()
            after[y:y + piece.shape[0], x:x + piece.shape[1]][piece > 0] = piece[piece > 0]
            after, lines = env.clear_filled_rows(after)
            if lines or buried(env.crop_padding(after)) <= covered:
                return False
    return True


def _outcome(env, matrix, rotation, x, y, heights):
    board = env.board.copy()
    board[y:y + matrix.shape[0], x:x + matrix.shape[1]][matrix > 0] = matrix[matrix > 0]
    filled = ~(board == 0).any(1) & ~(board == 1).all(1)
    cells = int((matrix[filled[y:y + matrix.shape[0]]] > 0).sum())
    rows = np.nonzero(matrix.any(1))[0]
    columns = np.nonzero(matrix.any(0))[0] + x - env.padding
    board, lines = env.clear_filled_rows(board)
    field = env.crop_padding(board)
    where, wall = _where(columns, heights, env.width)
    kept = reserved(heights, env.height) if RESERVED else None
    if kept is not None and lines == 0 and surface(field)[kept] > heights[kept]:
        # "It fills the only column deep enough for an I piece" was meant as the cost of the
        # landing and read as its merit: the sentence appeared on the landing that ended one
        # measured game, beside "It fills the lowest part of the stack" and "It flattens the
        # surface". A fact that states a loss has to say the loss.
        where += " It gives up the only column deep enough for an I piece."
    return field, describe(field, f"r{rotation}c{columns[0]}", rotation, x, lines,
                           env.height - (y + rows[-1]), lines * cells, where.strip(),
                           int(heights[columns].min() - heights.min()), wall)


def _where(columns, heights, width):
    """Which part of the stack the piece lands on, and which wall it touches.

    The two come back apart so the first can be reworded without losing the second: the wall
    sentence is the one the instructions were taught to value, and rewording the other is the
    change under test.
    """
    part = ""
    if heights.max() > heights.min():
        covered = heights[columns]
        if covered.min() == heights.min():
            part = "It fills the lowest part of the stack."
        elif covered.max() == heights.max():
            part = "It piles onto the highest part of the stack."
    wall = ("It sits against the left wall." if columns[0] == 0 else
            "It sits against the right wall." if columns[-1] == width - 1 else "")
    return part, wall


def graded_evenness(outcome, before):
    """How much flatter or rougher, not just which way.

    The three-way wording saturates the same way the fixed height band did: over a long game the
    surface sits at a bumpiness of ten, so a landing that roughens it by two and one that roughens
    it by eight both read "It makes the surface more uneven", and the model cannot separate them.
    """
    change = outcome.bumpiness - before.bumpiness
    return ("It flattens the surface a lot." if change <= -5 else
            "It flattens the surface." if change < -1 else
            "It leaves the surface as uneven as it is now." if change <= 1 else
            "It makes the surface more uneven." if change < 5 else
            "It makes the surface much more uneven.")


def graded_where(outcome, before):
    """How far above the lowest column the landing sits, not whether it is the lowest.

    "It fills the lowest part of the stack" and "It piles onto the highest part" leave every landing
    in between silent: on a stack with a dip at one end and a tower at the other, a landing two rows
    above the floor reads the same as one six rows above it. Both of the wordings that have gained
    rows rewrote a sentence this way instead of adding one.
    """
    return ("It fills the lowest part of the stack." if outcome.above_lowest == 0 else
            "It lands one row above the lowest part of the stack." if outcome.above_lowest == 1 else
            f"It lands {MANY[outcome.above_lowest]} rows above the lowest part of the stack."
            if outcome.above_lowest < len(MANY) else
            "It lands far above the lowest part of the stack.")


def phrase(outcome, before, grade=False, fills=False, count=False, depth=False, blocks=False, rest=False, terse=False, where=False, skip=()):
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
    evenness = (graded_evenness(outcome, before) if grade else
                "It flattens the surface." if outcome.bumpiness < before.bumpiness - 1 else
                "It leaves the surface as uneven as it is now." if outcome.bumpiness <= before.bumpiness + 1 else
                "It makes the surface more uneven.")
    # Only opening a gap was ever reported, so filling one earned nothing. A one-wide gap is what
    # forces the next piece to bury a cell, and filling it is how the stack stops needing an I.
    # `depth` states the gap the landing leaves instead of the change it makes: the games ended on
    # one piece laid across a twelve-deep gap, and "opens a deep one-wide gap" says nothing about
    # a gap that was already there.
    wells = ((f"It leaves a one-wide gap {MANY[outcome.well_depth]} rows deep."
              if outcome.well_depth < len(MANY) else "It leaves a one-wide gap more than ten rows deep.")
             if depth and outcome.well_depth >= 2 else "" if depth else
             "It opens a deep one-wide gap." if outcome.wells >= before.wells + 3 else
             "It opens a one-wide gap." if outcome.wells > before.wells else
             "It fills in a deep one-wide gap." if fills and outcome.wells <= before.wells - 3 else
             "It fills in a one-wide gap." if fills and outcome.wells < before.wells else "")
    # The trapped cells lead: read last, they lost to the positive facts beside them.
    # Of the six features the baseline weighs, this is the last one the options never carried.
    # On a stack with no buried cell the column transitions follow from the buried count, so they
    # add nothing; the split rows do not. `splits` counts them.
    towers = ("It joins the stack together." if outcome.splits < before.splits else
              "It leaves the stack in more separate towers." if outcome.splits > before.splits
              else "") if blocks else ""
    # The baseline's second heaviest feature is where the piece comes to rest, and no option ever
    # said it. "It keeps the top of the stack where it is" is true both of a piece that drops four
    # rows into a valley and of one that lies flat on the top row, and "It fills the lowest part of
    # the stack" is one bit. How far below the top it rests is the magnitude, and it does not
    # saturate as the stack grows.
    under = before.height - outcome.landing
    deep = ("It drops deep into the stack." if under >= 4 else
            "It drops into a dip in the stack." if under >= 2 else
            "It rests on top of the stack.") if rest else ""
    # Every sentence added to an option lowered agreement with the baseline, and the two changes
    # that helped reworded a sentence already there. The option text is the scarce thing, so the
    # parts are named and `skip` can take one out: an ablation says what each one is worth.
    parts = {
        "traps": ("" if terse else "Traps nothing.") if holes <= 0 else
                 f"Traps {(MANY[holes] if holes < len(MANY) else 'more than ten') if count else
                          (COUNT[holes] if holes < len(COUNT) else 'several')} "
                 f"cell{'' if holes == 1 else 's'} that no piece can reach.",
        # "Traps nothing." and "Clears no rows." are true of nearly every option once the red line
        # has run, so they are constant text that every other sentence has to compete with. With
        # `terse` the absence is silent and only the landing that traps or clears says so.
        "clears": "" if terse and outcome.lines == 0 else
                  f"Clears {COUNT[outcome.lines]} row{'' if outcome.lines == 1 else 's'}.",
        "where": graded_where(outcome, before) if where else outcome.where,
        "wall": outcome.wall,
        "rest": deep,
        "height": height,
        "evenness": evenness,
        "wells": wells,
        "towers": towers,
    }
    return " ".join(text for name, text in parts.items() if text and name not in skip)


def score(outcome):
    """Dellacherie's hand-tuned evaluation, the baseline the model is compared against."""
    return (-4.500158825082766 * outcome.landing
            + 3.4181268101392694 * outcome.eroded
            - 3.2178882868487753 * outcome.row_transitions
            - 9.348695305445199 * outcome.column_transitions
            - 7.899265427351652 * outcome.holes
            - 3.3855972247263626 * outcome.wells)
