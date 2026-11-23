"""Screen coordinates in the 'canonical' frame: screenshots are resized to 1536 px wide
(an S22 Ultra at 2316x1080 maps to 1536x716). Measured from real screenshots of the game.
ROIs are (x, y, w, h).
"""
CANON_W = 1536

BOARD_ROI = (180, 45, 1150, 495)        # all board cards live here (top row starts near y=75)
ACTIVE_ROI = (790, 535, 140, 175)       # the face-up card at the bottom
ACTIVE_INNER = (826, 566, 64, 112)      # strictly inside that card (used to spot a WILD there)
ACTIVE_TAP = (858, 622)

DRAW_TAP = (727, 622)                   # top card of the face-down draw pile
PILE_TOP_ROI = (692, 562, 70, 120)      # that card, compared against its look at level start
END_LEVEL_ROI = (285, 622, 190, 52)     # red END LEVEL button, shown once the pile is empty
END_LEVEL_TAP = (377, 648)
PLUS5_TAP = (712, 615)                  # "+N cards" tile (+5, +7, +10 ...) that replaces the empty pile
PLUS5_BADGE_ROI = (744, 538, 30, 30)    # red count badge on that tile = you own it for free
TILE_PRICE_ROI = (600, 640, 260, 76)    # where a coin price shows under/over the tile's bottom when it costs coins
WILD_PRICE_ROI = (1370, 645, 90, 65)    # coin + "4,000" under the WILD card: always shown, used as the coin reference

GEAR_ROI = (75, 20, 56, 56)             # blue settings gear, top-left, in-game only
WILD_ROI = (1415, 515, 80, 100)         # golden WILD card, in-game only

# Green buttons the bot may tap: name -> (x, y, tolerance, requires the red X of a level popup)
SAFE_BUTTONS = {
    "level popup Play / Try again": (767, 500, 60, True),
    "lobby Level": (1435, 615, 90, False),
}
POPUP_CLOSE_ROI = (1060, 103, 46, 46)   # red X of the level popup
CLOSE_SEARCH_ROI = (0, 0, 1536, 260)    # where event popups put their red X close button

# Never tapped: undo (930-1070, 540-670) and the WILD card (both cost coins).
# The "+N cards" tile is tapped when free, or when bought on purpose (config "buy_extra_cards").

CARD_W, CARD_H = 89, 138
CARD_SHORT = (55, 115)                  # accepted card width range in px (layouts can scale cards)
