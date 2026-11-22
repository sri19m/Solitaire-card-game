"""Screen coordinates in the 'canonical' frame: screenshots are resized to 1536 px wide
(an S22 Ultra at 2316x1080 maps to 1536x716). Measured from real screenshots of the game.
ROIs are (x, y, w, h).
"""
CANON_W = 1536

BOARD_ROI = (340, 110, 870, 420)        # all board cards live here
ACTIVE_ROI = (800, 540, 120, 165)       # the face-up card at the bottom
ACTIVE_TAP = (858, 622)

DRAW_TAP = (727, 622)                   # top of the face-down draw pile
STOCK_COIN_ROI = (668, 662, 38, 34)     # gold coin of the "+5 cards" shop tile (only when pile is empty)
END_LEVEL_TAP = (370, 648)              # red END LEVEL button (only shown when stuck)

GEAR_ROI = (75, 20, 56, 56)             # blue settings gear, top-left, in-game only
WILD_ROI = (1415, 515, 80, 100)         # golden WILD card, in-game only

# Never tap these: they cost coins.
UNDO_ZONE = (930, 540, 140, 130)        # green undo button (250-400 coins)
SHOP_ZONE = (630, 540, 150, 160)        # "+5" card purchase (2900 coins) appears where the pile was

CARD_W, CARD_H = 89, 138
