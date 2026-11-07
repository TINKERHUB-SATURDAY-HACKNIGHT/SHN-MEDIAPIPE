# FreshPick - Beyond Gesture (MediaPipe)

A voice-first fruit freshness guide for visually impaired users. MediaPipe turns the hand into a
**spatial guide, a region-of-interest cropper and a gesture remote**; a vision-language model narrates
what the fruit looks like; everything is spoken aloud without freezing the camera loop.

## Run
```bash
pip install -r requirements.txt
set GEMINI_API_KEY=your_key        # Windows  (export on macOS/Linux). Optional: no key = offline HSV mode
python main.py                     # add --fullscreen or --offset-x 1920 for a side screen
python main.py --analyze-image apple.jpg   # test the vision pipeline with no camera
```
Use mediapipe 0.10.x (legacy `mp.solutions.hands` API). Beeps use `winsound` (Windows); other OSes get the terminal bell.

## Gestures
| Gesture | Action |
|---|---|
| Pinch thumb+index, hold 0.4 s | **Rotate-to-inspect scan**: scans front, asks you to twist your wrist, scans back, speaks one description |
| Both hands up, hold 2 s | **Duel mode**: compares left vs right fruit, says which to eat first |
| Open palm, hold 1 s | **Pantry recap**, ranked Eat today -> Store for later |
| Thumbs up | **Touch check** tips (cameras cannot see internal spoilage) |
| Fist, hold 1 s | Repeat last answer |

## Features
- **Audio homing**: spoken "move hand left/right/up/down" + confirmation beep when the hand enters the centre box.
- **Audio sonar**: beep pitch rises as the fruit nears the camera; "bring closer / move back" voice prompts.
- **Hand-anchored ROI crop** (from a clean frame, no skeleton overlay) sent to the VLM, which removes background clutter.
- **Gemini VLM** image-to-text with strict assistive prompt, **offline OpenCV HSV fallback** when no key / no network.
- **Threaded TTS** with interruptible sentences, so the 30 FPS loop never stalls.
- **Pantry memory** saved to `pantry_session.json`.
- **High-contrast HUD**: skeleton, pinch meters, hold-progress bar, big subtitles, colour-coded pantry panel.
- Keyboard fallbacks for demos: `S` scan, `D` duel, `R` recap, `T` touch tip, `P` repeat, `C` clear pantry,
  `M` mute, `N` sonar on/off, `H` help, `F` fullscreen, `Q` quit.

## Demo tips (side screen)
1. `python main.py --offset-x 1920 --fullscreen` (or drag the window, then press `F`).
2. Good front lighting, hand about 30-50 cm from the camera, fruit visible beside the palm.
3. Run: homing -> pinch scan -> twist -> second fruit -> both-hands duel -> open-palm recap.

## Ethical note
FreshPick only describes visible surface condition and never claims to detect internal rot.
It always reminds users to check by touch and smell. It is an aid, not a food-safety guarantee.
