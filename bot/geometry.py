"""Screen coordinates in the 'canonical' frame: screenshots are resized to 1536 px wide
(an S22 Ultra at 2316x1080 maps to 1536x716). Measured from real screenshots of the game.
ROIs are (x, y, w, h).
"""
CANON_W = 1536

BOARD_ROI = (180, 95, 1210, 445)        # all board cards live here
ACTIVE_ROI = (790, 535, 140, 175)       # the face-up card at the bottom

DRAW_TAP = (727, 622)                   # top card of the face-down draw pile
PILE_TOP_ROI = (692, 562, 70, 120)      # that card, compared against its look at level start
END_LEVEL_ROI = (285, 622, 190, 52)     # red END LEVEL button, shown once the pile is empty
END_LEVEL_TAP = (377, 648)

GEAR_ROI = (75, 20, 56, 56)             # blue settings gear, top-left, in-game only
WILD_ROI = (1415, 515, 80, 100)         # golden WILD card, in-game only

# Green buttons the bot may tap: name -> (x, y, tolerance, requires the red X of a level popup)
SAFE_BUTTONS = {
    "level popup Play / Try again": (767, 500, 60, True),
    "lobby Level": (1435, 615, 90, False),
}
POPUP_CLOSE_ROI = (1060, 103, 46, 46)   # red X of the level popup

# Never tapped, they cost coins: undo (930-1070, 540-670), the "+5 cards" tile that replaces the
# empty pile (630-780, 540-700) and the WILD card. The bot only taps board cards, the pile while
# it is verified present, END LEVEL, and SAFE_BUTTONS.

CARD_W, CARD_H = 89, 138
CARD_SHORT = (55, 115)                  # accepted card width range in px (layouts can scale cards)
