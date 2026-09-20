# decider-tetris

Tetris played by a small decision model. Python enumerates every landing the piece can reach and
states what each one does to the stack; the model only picks one. It never sees the board, counts
a cell or compares a number.

The model is any endpoint that answers the TypeSafe Jev wire format
(`POST /v1/systemone`, a `choice` question with up to 255 described options). The runs below used
the open [Decider-2B](https://huggingface.co/Mapika/decider-2b) checkpoint on one RTX 3070 Ti,
about 45 ms per question.

The published [Dellacherie evaluation](https://hal.science/hal-00926213/document) scores the same
landings on every piece, so every run carries its own baseline.

## Result

Same pieces, same seeds, 500-piece cap, five games each.

| | rows cleared | games that reached 500 pieces |
|---|---|---|
| Decider-2B | 974 | 5 / 5 |
| Dellacherie heuristic | 987 | 5 / 5 |

The model plays at 98.7% of a hand-tuned weighted evaluation while comparing short English
sentences, without arithmetic, and agrees with the heuristic's pick on only about two thirds of
the landings: most of the disagreements are between moves that are both fine.

Lift the cap and the difference finally shows. On one seed, side by side on the same pieces, the
model died on piece 1759 with 688 rows; the heuristic was on 701 at that moment and was still
playing, on an almost empty board, at piece 1875.

## How it got there

Nothing below is a reworded prompt. Each step changes what the model is told or how it is asked.

| change | rows | baseline |
|---|---|---|
| one sentence of facts per landing | 41 | 293 |
| name the one-wide gap in the instructions, since the options already report it | 57 | 293 |
| never offer a landing that traps a cell while a clean landing exists | 139 | 293 |
| state the height as a change from the current stack, not as a fixed band | 243 | 293 |
| (the four rows above ran to a 150-piece cap; from here the cap is 500) | 728 | 991 |
| put every landing in one question instead of a tournament of tens | 898 | 991 |
| read the list backwards and add the probabilities when the top two are within 0.15 | 974 | 987 |

The two large jumps are both about information, not persuasion:

* **The red line.** A landing that buries a cell while a clean landing is on offer is never right,
  so the code removes it before the model sees it. Trapping fell from a fifth of all placements to
  almost none, and the model still ranks everything that remains.
* **A fact that does not saturate.** "The stack is high" was true of every option above twelve
  rows, exactly where the choice decides the game. "It raises the top of the stack by two rows"
  keeps its edge at any height.

## The pick moves with the option order

Asking one position twelve times under twelve different option orders:

| | decisions | mean gap between the top two probabilities |
|---|---|---|
| same pick every time | 3 | 0.34 |
| pick changed with the order | 11 | 0.12 |

The numbers are not rounded: the model simply separates similar landings by less than the pull of
the leading position. Averaging over orders removes it. Reading the list once forwards and, only
when the top two are within 0.15, once backwards costs about 1.3 questions per piece and keeps the
run reproducible: no random draw is involved, so one board always produces one decision.

## What did not work

Kept here because each one is a measurement, and each one cost a few hundred games to find out.

| attempt | result |
|---|---|
| the same four goals as an ordered list ("First… Second… Third…") | 0 rows in five games, against 41 for one short sentence |
| the board itself in the state, as 20 rows of text | 41 pieces against 86; the model's own card reports the same weakness on positional lookups |
| the same board as TOON instead of JSON | 60 pieces against 41: the per-element `_index` markers hurt, but a grid still hurts more than no grid |
| rating each landing alone with a yes/no question | 11 rows against 18, and four times slower: a landing rated alone has nothing to be better than |
| a next-piece fact ("the next piece can clear a row here") | worse on three of four seeds |
| dropping landings that another landing beats on every fact at once | leaves a median of one option, so the code is playing, not the model |
| sorting the options best-first | 844 rows against 822 for the enumeration order: position bias flips individual picks but does not carry a game |

## Running it

```bash
uv sync
uv run python -m decider_tetris.play --decider http://127.0.0.1:8000 \
    --clean --group 20 --order enum --settle 0.15 --games 5 --max-pieces 500
uv run python -m decider_tetris.play --games 5          # the heuristic alone, no model needed
uv run python -m decider_tetris.race --decider http://127.0.0.1:8000    # both, side by side
```

`--decider` is the base URL of anything that answers `POST /v1/systemone`. For the open checkpoint,
its own `decider.serve` works, though it captures a CUDA graph for 78 shapes before answering the
first request; a server that builds graphs on demand starts in about a minute on an 8 GB card.

`race` puts both players in one window on the same piece sequence. `--free` lets each side run at
its own speed instead of one piece each in turn.

`web.py` plays the published [react-tetris](https://chvin.github.io/react-tetris/) page instead of
the local board: it reads the page's own Redux state out of `localStorage`, sends key events over
CDP, and changes nothing on the page. Its rules mirror that page's source, so a landing is only
offered if the page would accept it; over twelve pieces its predicted board matched the page's
board exactly every time.

## Layout

```
src/decider_tetris/
  placements.py        every landing the piece can reach, its facts in words, the baseline score
  play.py              one game, and the flags every measurement above was made with
  race.py              both players in one window, in step or at their own speed
  view.py              the board drawn with the decision beside it
  web.py               the same player driving the published react-tetris page
  tournament_probe.py  asks one decision two ways to measure what grouping costs
```

## Credits

The board is [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium). The split between
arithmetic in code and judgement in a fast model follows
[jev-game-tools](https://github.com/Eniip/jev-game-tools), which plays Brotato the same way.
