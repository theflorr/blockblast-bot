# Block Blast Bot

Python script plays the mobile game Block Blast for you. It mirrors your Android phone's screen on your computer, watches the board using basic computer vision (no machine learning model, just color matching and contour detection), figures out the best move, and taps your phone for you to place the piece.

## What it does

1. Connects to your phone through scrcpy and pulls a live video feed of the screen.
2. Looks at the feed and works out two things:
   - Which of the 8x8 board cells are filled or empty, by comparing pixel colors against the known background color.
   - What pieces are sitting in the tray at the bottom of the screen, by finding their outlines and turning each one into a small grid shape.
3. Every few seconds it takes the current board and the available pieces and searches through the possible ways to place them (looking a few pieces ahead), scoring each option by how many lines it clears and how clean the board looks afterward.
4. Once it picks a move, it sends touch and swipe commands back to the phone through scrcpy to actually pick up the piece and drop it in place.
5. Shows a debug window on your computer the whole time with boxes drawn over what it's detecting, plus an FPS counter, so you can see what it's "seeing" and check it's not confused.

## Requirements

- Python 3
- An Android phone with Wireless debugging enabled, connected via ADB
- Packages: `opencv-python`, `numpy`, `scrcpy-client`
- Windows (This is not supported for linux, nor mac fuck you guys)

## Running it

We recommend using wireless debugging to combat physical cable issues. The script was originally made using wireless debugging so use it too.

```
python app.py
```

A window will pop up showing the mirrored screen with detection overlays.

## Important notes

- The board and tray positions (`bx, by, bw, bh` and the tray region) are hardcoded pixel coordinates. They were set up for one specific phone resolution and one specific game layout, so this will very likely need to be recalibrated (different coordinates, maybe a different background color for the board/tray) if you're running it on a different device or if the app updates. This was made on a samsung s22 !!
- The formula that converts a detected piece position into a touch target on screen was arrived at by fitting numbers to what worked, not from any real geometry. Same story here: it may need to be redone if your screen or phone is different.
- This is for personal experimentation. Automating input in a game like this is likely against the game's terms of service, so use it at your own risk.
