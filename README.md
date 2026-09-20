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

<video src="https://github.com/coseto6125/decider-tetris/raw/main/docs/race.webm" poster="https://github.com/coseto6125/decider-tetris/raw/main/docs/race.png" controls width="100%"></video>

[![Dellacherie and Decider-2B on the same pieces, both at 380 rows](docs/race.png)](docs/race.webm)

`race.py`, 91 seconds: the published heuristic on the left, the model on the right, one piece
sequence between them. Under each board is the sentence that player acted on — the heuristic's
weighted score, and for the model the option text it picked and the probability it gave it. They
finish the recording level, 380 rows each at 954 pieces.

## Result

Same pieces, same seeds, 500-piece cap, five games each.

| | rows cleared | games that reached 500 pieces |
|---|---|---|
| Decider-2B | 987 | 5 / 5 |
| Dellacherie heuristic | 987 | 5 / 5 |

The model matches a hand-tuned weighted evaluation while comparing short English sentences,
without arithmetic, and while agreeing with that evaluation's pick on only 72% of the landings:
most of the disagreements are between moves that are both fine.

Lift the cap and the question changes from "does it play well" to "how long does it last". On one
seed with no cap:

| | pieces | rows | still alive |
|---|---|---|---|
| Decider-2B, every red line below | 35,874 | 14,338 | no |
| Dellacherie heuristic | 75,004 | 30,000 | yes, stopped by the row cap |

The published controllers are measured in millions of rows per game. The heuristic is not at its
own ceiling in that table either; it was stopped, not beaten.

**Rows are a saturated measure here.** Over the 14,338-row game the stack averaged 4.7 rows high
with 0.6 buried cells, and every 2,500 pieces cleared exactly 1,000 rows: 2,500 pieces are 10,000
cells and 1,000 rows are 10,000 cells, so no cell is wasted. Rows are therefore pieces times 0.4
and one game is one sample of the only thing that varies, which is when it dies. The ruler used
below is the climb: every time the stack passes twelve rows is a sample, and the game ends when one
of those climbs does not come back down. The 14,338-row game survived 63 climbs; the 5,893-row
game survived 19.

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
| say in the instructions that a landing against a wall beats one in the middle | 987 | 987 |
| (a tie in `settle` was broken by set iteration order, so one seed gave 3894, 1863 and 322 rows under the same flags; every row above this line predates the fix and every row below it is one uncapped game on seed 6) | | |
| when the red line has to bury, offer only the landings that bury fewest | 3,396 | |
| hide the landings that leave a gap three or more rows deep, while a shallower one exists | 5,893 | |
| above fifteen rows, let a burying landing through when fewer than three clean ones remain | 9,981 | |
| drop the landings that leave the next piece no landing of its own that buries nothing | 14,338 | |

The two large jumps are both about information, not persuasion:

* **The red line.** A landing that buries a cell while a clean landing is on offer is never right,
  so the code removes it before the model sees it. Trapping fell from a fifth of all placements to
  almost none, and the model still ranks everything that remains.
* **A fact nobody asked for.** Cloning the board at every disagreement and letting the heuristic
  play both sides forward showed which differences cost rows: the expensive ones nearly all had
  the heuristic landing against a wall and the model landing in the middle. The option text had
  said `It sits against the left wall` all along; the instructions had never said that was good.
  `regret.py` runs that comparison.
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
| relaxing the buried-cell red line at any height | 288 rows against 5,893; the same relaxation above fifteen rows reaches 9,981, so the height gate carries the whole effect |
| letting a burying landing through when it clears two rows | 2,803 rows against 5,893 |
| rewording which part of the stack the landing sits on | 3,882 rows against 5,893 |
| raising the second-reading threshold from 0.15 to 0.3 | 2,462 rows against 5,893 |
| Decider-2B v10 in place of v8, same flags and seed | 7,731 rows against 14,338. v10 is the more decisive model — mean confidence 0.65 against 0.58, and it asks for the second reading half as often — but it kept no cleaner a board and survived 31 climbs against 63. One game each: two samples of a heavy-tailed variable differ by two times a third of the time, so this says v10 is not better here, not that it is worse |
| predicting a climb from the board | four measures were tried at the moment the stack first passes eight rows — buried cells, unevenness, how many landings survive the red lines, and how many of the seven pieces the stack can take without growing taller. None separates the 36 excursions that ran away from the 375 that did not |

## Running it

```bash
uv sync
uv run python -m decider_tetris.play --decider http://127.0.0.1:8000 \
    --wording walls --group 20 --order enum --settle 0.15 \
    --clean --grade --fills --least --gap any --relax 3 --high 15 --foresee 10 \
    --seed 6 --max-pieces 200000 --postmortem death.json
uv run python -m decider_tetris.play --games 5          # the heuristic alone, no model needed
uv run python -m decider_tetris.race --decider http://127.0.0.1:8000    # both, side by side
```

That line is the 14,338-row run. Every switch on it is one row of the table above, and the result
line repeats all of them, so a number always says which flags produced it. `--seed` must be 1 or
more: the environment reads `if seed and seed > 0`, so seed 0 is taken as no seed and the pieces
come from the operating system, which makes a run look reproducible when it is not.

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
  regret.py            replays each disagreement with the heuristic and says what it cost
tests/                 27 tests, including one per red line and one for the tie-break
```

## Credits

The board is [Tetris Gymnasium](https://github.com/Max-We/Tetris-Gymnasium). The split between
arithmetic in code and judgement in a fast model follows
[jev-game-tools](https://github.com/Eniip/jev-game-tools), which plays Brotato the same way.
