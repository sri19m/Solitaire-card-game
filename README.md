# Solitaire bot

Plays a TriPeaks-style solitaire game on an Android phone over ADB. Everything is local and free.

## Setup (laptop)
1. Install Android platform-tools (`adb`). On the phone: Developer options -> USB debugging on. Plug in, accept the prompt, check `adb devices`.
2. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Keep the game in landscape, with the screen on.

## First run: teach it the cards (once)
    python main.py --learn
Play to a board by hand or let the bot tap through. Whenever it meets a card it can't read, a window shows it and
the terminal asks for the rank (A, 2..10, J, Q, K). Labels are saved to `glyphs/`. About 13 labels cover all ranks;
add a few extra samples if it misreads any.

## Run
    python main.py --dry-run     # read the screen and print the plan, no taps
    python main.py               # play autonomously
    python main.py --levels 50   # stop after 50 wins
Results go to `runs.csv`.

## How it decides
* `bot/vision.py`: finds fully visible face-up cards by colour, reads the big centre glyph by nearest-neighbour against your labelled samples.
* `bot/brain.py`: counts every card seen this level, so it knows the odds that a hidden card is playable.
  It scores every possible chain of taps by chain length, face-down cards likely to be freed, how many cards stay playable, and the odds that the next card continues the run.
* `bot/sim.py` + `tools/simulate.py`: an offline simulator for comparing strategies (`python tools/simulate.py`).

## Safety rules built in
* Never taps undo (250-400 coins), the "+5 cards" shop tile (2,900), or the WILD card (3,600).
* Stops after `max_losses_in_a_row` losses, because each retry costs coins.
* Unknown screens are saved to `unknown/` and the bot stops after repeated ones.
