# Solitaire bot

Plays a TriPeaks-style solitaire game on an Android phone over ADB. Everything runs locally and is free.

## Setup (Windows laptop)
1. Platform-tools (adb): download and unzip once, then add it to PATH in each new terminal:
   ```
   $env:Path += ";$env:USERPROFILE\android\platform-tools"
   adb devices        # must show your phone as "device"
   ```
2. `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`
3. Keep the game in landscape with the screen on.

## Teach it the cards
    python main.py --learn
It asks whenever it isn't sure about a card. A window shows the card; type its rank
(A, 2..10, J, Q, K, or s to skip). **Every rank needs a sample, including 5 and 8**: until all
13 are labelled, learn mode only trusts very close matches. When a tap does nothing it shows
you that card too; type the real rank, or press Enter if the bot's reading was right (the card is covered).

Check your labels at any time:

    python tools/check_glyphs.py       # counts per rank, likely mislabels, writes glyph_sheet.png

A wrong label lives in `glyphs/` as `<RANK>_<n>.png`: delete that file.

## Teach it screens
Reward popups such as "Congratulations! Tap to collect" are learned the same way: in `--learn`
mode an unknown screen opens in a window. Click where the bot should tap and press Enter
(`w` = the screen goes away by itself, `s` = stop the bot there, Esc = skip). Taught screens live in
`screens/`; delete a pair of files there to forget one. Never teach a tap on anything that costs coins.

## Run
    python main.py --dry-run     # read the screen once, print the plan, no taps
    python main.py               # play
    python main.py --debug       # also save annotated screenshots to debug/
    python main.py --levels 50   # stop after 50 wins
Cards it was unsure about during normal play go to `pending/`; label them later with
`python tools/label_pending.py`. Results go to `runs.csv`.

## How it plays
* `bot/vision.py`: finds fully visible face-up cards (also tilted ones, and splits overlapping
  cards so only the top one counts), straightens each card, and reads the big centre rank plus the
  corner index against your labelled samples. A card is only used when the match is clearly better
  than every other rank; otherwise it is "unsure" and never tapped blindly.
* `bot/brain.py`: counts every card seen this level, and scores every possible chain of taps by
  length, face-down cards freed, cards left playable and the odds the next card continues the run.
  Ace and King are adjacent.
* Every tap is verified on the next screenshot. A card that is still there after being tapped
  (misread or covered) is skipped until the board changes.
* `bot/sim.py` + `tools/simulate.py`: offline simulator for comparing strategies.

## Never spends coins during a level
* Never taps undo, the WILD card, or the "+5 cards" tile.
* Draws only while the pile still looks like it did at level start, the END LEVEL button is not
  shown, and the last draw actually changed the bottom card. Otherwise it stops and saves a screenshot.
* When the pile is empty and no move is left, it taps END LEVEL instead of buying cards.
* Only taps the level popup's Play / Try again and the lobby Level button, never other green buttons.
  Starting a level costs the game's entry fee; it stops after `max_losses_in_a_row` levels not won.
* Unknown screens are saved to `unknown/` and nothing is tapped.
