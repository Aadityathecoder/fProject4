# Source Code Warehouse and Scrum Prototype Audit

This warehouse consolidates the strongest source-code ideas from the existing Q3 prototypes into one final direction for the rover vision GUI.

## Final Deliverable Files

- `webVersion/api.py`: consolidated FastAPI backend, image-processing pipeline, movement state, user log, simulated stream, uploaded frame support, and source-code warehouse endpoint.
- `webVersion/gui.html`: CEO-facing clickable four-quadrant GUI prototype.
- `SOURCE_CODE_WAREHOUSE.md`: audit record for which prototype ideas were reused.

## Prototype Audit

| Source | Best Reused Idea | Reason It Was Selected |
| --- | --- | --- |
| `webVersion/gui.html` | Browser GUI base | Already had a 2x2 quadrant layout and clickable rover controls. |
| `webVersion/api.py` | FastAPI web entrypoint | Existing web server route structure was the best place to integrate the final prototype. |
| `turn/api.py` | Hough Lines lane tracking with recovery | Strongest lane-border logic because it classifies left/right lines and remembers the last visible side. |
| `straightLineAuto/api.py` | Processed/raw stream endpoints and status fields | Kept compatibility with `/video_feed`, `/video_feed_raw`, `/status`, `/upload_frame`, `/stop`, and movement commands. |
| `junkFile/perspectiveTransform.py` | Perspective Transform | Provided the clearest bird's-eye transform approach for path geometry. |
| `obstacleDetection/main.py` | Motion plus color filtering | Useful future source for obstacle overlays based on movement and white-object filtering. |
| `junkFile/canCode.py` | Hough Circles | Reused as a circular marker/target detection concept in the final video overlay. |
| `facialDetection/api.py` | SIFT/homography object detection | Kept as a future extension pattern for recognizing specific signs or visual targets. |
| `advanced_navigation_framework.py` | Architecture reference | Best long-term design because it includes state-machine navigation, threaded vision, smoothing, and odometry fallback. |
| `MotorDriver.py` | Robot client contract | Preserved the status and frame-upload expectations used by the Raspberry Pi robot bridge. |

## Final Selected Solution

The final prototype uses `webVersion` as the integration point.

The backend now supports:

- Hough Lines for left/right path borders.
- Estimated missing path borders when only one side is visible.
- Centerline calculation and pixel error reporting.
- Hough Circles for circular marker overlays on uploaded camera frames.
- Perspective Transform inset for bird's-eye path preview.
- Clean simulated video frames when no rover camera is connected.
- Uploaded camera frames through the existing `/upload_frame` endpoint.
- Clickable manual state with FWD, BWD, LEFT, RIGHT, GO, and STOP.
- User log events returned by `/status`.
- A `/source_code_warehouse` endpoint for the warehouse data used by the GUI.

The GUI now shows:

- Quadrant 1: realtime rover video stream with path overlay, centerline, path borders, travel vector, and perspective inset.
- Quadrant 2: directional rover controls with GO and STOP.
- Quadrant 3: source-code warehouse audit from the previous prototypes.
- Quadrant 4: user log and live status telemetry.

## Run Command

From the repo's `pwpr` folder:

```bash
cd webVersion
python3 -m uvicorn api:app --host 127.0.0.1 --port 5000 --reload
```

Then open:

```text
http://127.0.0.1:5000
```
