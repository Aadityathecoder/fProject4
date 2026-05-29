from __future__ import annotations

import asyncio
import base64
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse


app = FastAPI(title="PWP Rover CEO Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRAME_W = 640
FRAME_H = 480
LANE_WIDTH_ESTIMATE = 260
CENTER_DEADBAND = 35
BLUE_LOWER = np.array([90, 80, 50])
BLUE_UPPER = np.array([140, 255, 255])


@dataclass
class PathOverlay:
    left_line: Optional[tuple[int, int, int, int]]
    right_line: Optional[tuple[int, int, int, int]]
    center_line: Optional[tuple[int, int, int, int]]
    center_error: int
    command: str
    status: str
    confidence: float


latest_frame_raw: Optional[bytes] = None
latest_frame_processed: Optional[bytes] = None
latest_overlay = PathOverlay(None, None, None, 0, "stop", "Prototype stream ready", 0.0)

controls = {"forward": False, "backward": False, "left": False, "right": False}
selected_direction = "forward"
motion_enabled = False
autonomous_enabled = False
auto_command = "stop"
auto_error = 0
lane_status = "Prototype stream ready"
frame_source = "simulated"
event_log: deque[str] = deque(maxlen=80)


SOURCE_CODE_WAREHOUSE = [
    {"source": "webVersion/gui.html", "selected": "Browser GUI shell", "reason": "Already used a 2x2 quadrant layout and clickable movement controls."},
    {"source": "turn/api.py", "selected": "Best Hough line lane recovery", "reason": "Classifies left/right Hough lines, estimates missing borders, and keeps last-seen lane memory."},
    {"source": "straightLineAuto/api.py", "selected": "Stable stream and status endpoints", "reason": "Provides processed/raw stream endpoints and status fields that match the motor client."},
    {"source": "junkFile/perspectiveTransform.py", "selected": "Perspective transform helper", "reason": "Shows the clearest reusable bird's-eye transformation pattern."},
    {"source": "obstacleDetection/main.py", "selected": "Motion plus color obstacle filtering", "reason": "Combines background subtraction and HSV masks for moving bright obstacles."},
    {"source": "junkFile/canCode.py", "selected": "Hough circle marker detection", "reason": "Detects circular objects that can become cans, signs, or target markers."},
    {"source": "facialDetection/api.py", "selected": "Feature matching and homography concept", "reason": "Useful for future object recognition overlays after the CEO prototype is approved."},
    {"source": "advanced_navigation_framework.py", "selected": "Final architecture reference", "reason": "Adds a state machine, threaded vision, odometry fallback, smoothing, and clear control contracts."},
    {"source": "MotorDriver.py", "selected": "Robot bridge contract", "reason": "Keeps the status fields and upload endpoint compatible with the Raspberry Pi motor client."},
]


def log_event(message: str) -> None:
    event_log.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def reset_controls() -> None:
    for key in controls:
        controls[key] = False


def apply_motion_state() -> None:
    reset_controls()
    if motion_enabled and selected_direction in controls:
        controls[selected_direction] = True


def manual_command() -> str:
    for direction, active in controls.items():
        if active:
            return direction
    return "stop"


def average_line(lines: list[tuple[int, int, int, int]]) -> Optional[tuple[float, float]]:
    if not lines:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    if len(xs) < 4:
        return None
    fit = np.polyfit(np.array(ys, dtype=np.float32), np.array(xs, dtype=np.float32), 1)
    return float(fit[0]), float(fit[1])


def x_at_y(fit: Optional[tuple[float, float]], y: int) -> Optional[int]:
    if fit is None:
        return None
    return int(fit[0] * y + fit[1])


def line_from_fit(fit: Optional[tuple[float, float]], y1: int, y2: int, width: int) -> Optional[tuple[int, int, int, int]]:
    if fit is None:
        return None
    return (
        int(np.clip(fit[0] * y1 + fit[1], 0, width - 1)),
        y1,
        int(np.clip(fit[0] * y2 + fit[1], 0, width - 1)),
        y2,
    )


def perspective_transform(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    src = np.float32([[int(w * 0.20), int(h * 0.45)], [int(w * 0.80), int(h * 0.45)], [int(w * 0.03), h - 1], [int(w * 0.97), h - 1]])
    dst = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h))


def lane_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
    value = cv2.GaussianBlur(hsv[:, :, 2], (7, 7), 0)
    sat_mask = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    val_mask = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 145)
    mask = cv2.bitwise_or(cv2.bitwise_and(sat_mask, val_mask), edges)
    mask[: int(frame.shape[0] * 0.42), :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def classify_hough_lines(lines: Optional[np.ndarray], width: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    left_lines: list[tuple[int, int, int, int]] = []
    right_lines: list[tuple[int, int, int, int]] = []
    if lines is None:
        return left_lines, right_lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        length = math.hypot(x2 - x1, y2 - y1)
        mid_x = 0.5 * (x1 + x2)
        if abs(slope) < 0.35 or length < 30:
            continue
        if slope < 0 and mid_x < width * 0.76:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > width * 0.24:
            right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines


def detect_hough_circles(frame: np.ndarray) -> list[tuple[int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=45, param1=90, param2=24, minRadius=8, maxRadius=75)
    if circles is None:
        return []
    rounded = np.uint16(np.around(circles))
    return [(int(x), int(y), int(r)) for x, y, r in rounded[0, :]]


def detect_blue_stop_line(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700:
            continue
        x, y, line_w, line_h = cv2.boundingRect(contour)
        if line_w > line_h * 2.5 and line_w / float(w) >= 0.45 and y + line_h >= int(h * 0.82):
            return True
    return False


def compute_overlay(frame: np.ndarray) -> PathOverlay:
    global auto_command, auto_error, lane_status
    h, w = frame.shape[:2]
    mask = lane_mask(frame)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=34, minLineLength=30, maxLineGap=35)
    left_lines, right_lines = classify_hough_lines(lines, w)
    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)
    y_bottom = h - 1
    y_top = int(h * 0.46)
    y_look = int(h * 0.82)
    frame_center = w // 2
    x_left = x_at_y(left_fit, y_look)
    x_right = x_at_y(right_fit, y_look)
    if x_left is not None and x_right is not None:
        lane_center = (x_left + x_right) // 2
        status = "Both path borders detected"
        confidence = 1.0
    elif x_left is not None:
        lane_center = x_left + LANE_WIDTH_ESTIMATE // 2
        right_fit = (left_fit[0], left_fit[1] + LANE_WIDTH_ESTIMATE) if left_fit else None
        status = "Left border detected; right border estimated"
        confidence = 0.58
    elif x_right is not None:
        lane_center = x_right - LANE_WIDTH_ESTIMATE // 2
        left_fit = (right_fit[0], right_fit[1] - LANE_WIDTH_ESTIMATE) if right_fit else None
        status = "Right border detected; left border estimated"
        confidence = 0.58
    else:
        lane_center = frame_center
        left_fit = (-0.42, w * 0.48)
        right_fit = (0.42, w * 0.16)
        status = "No strong Hough lines; using prototype guide overlay"
        confidence = 0.25
    lane_center = int(np.clip(lane_center, 0, w - 1))
    error = lane_center - frame_center
    if detect_blue_stop_line(frame):
        command = "stop"
        status = "Blue stop line detected"
    elif abs(error) <= CENTER_DEADBAND:
        command = "forward"
    else:
        command = "left" if error < 0 else "right"
    auto_command = command
    auto_error = int(error)
    lane_status = f"{status} | center error {error}px"
    left_line = line_from_fit(left_fit, y_bottom, y_top, w)
    right_line = line_from_fit(right_fit, y_bottom, y_top, w)
    center_line = None
    if left_line and right_line:
        center_line = ((left_line[0] + right_line[0]) // 2, y_bottom, (left_line[2] + right_line[2]) // 2, y_top)
    return PathOverlay(left_line, right_line, center_line, int(error), command, status, confidence)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.rectangle(image, (x - 8, y - text_h - 8), (x + text_w + 8, y + 8), (8, 13, 19), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, overlay: PathOverlay, detect_markers: bool = True) -> np.ndarray:
    rendered = frame.copy()
    h, w = rendered.shape[:2]
    if overlay.left_line and overlay.right_line and overlay.center_line:
        lx1, ly1, lx2, ly2 = overlay.left_line
        rx1, ry1, rx2, ry2 = overlay.right_line
        path_poly = np.array([[(lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)]], dtype=np.int32)
        fill = rendered.copy()
        cv2.fillPoly(fill, path_poly, (40, 125, 205))
        rendered = cv2.addWeighted(fill, 0.22, rendered, 0.78, 0)
        cv2.line(rendered, (lx1, ly1), (lx2, ly2), (0, 220, 255), 4, cv2.LINE_AA)
        cv2.line(rendered, (rx1, ry1), (rx2, ry2), (0, 220, 255), 4, cv2.LINE_AA)
        cx1, cy1, cx2, cy2 = overlay.center_line
        cv2.line(rendered, (cx1, cy1), (cx2, cy2), (0, 255, 105), 3, cv2.LINE_AA)
        cv2.arrowedLine(rendered, (cx1, cy1 - 30), (cx2, cy2 + 55), (255, 165, 45), 4, tipLength=0.18)
    if detect_markers:
        for x, y, radius in detect_hough_circles(frame)[:2]:
            cv2.circle(rendered, (x, y), radius, (210, 90, 255), 3, cv2.LINE_AA)
            cv2.circle(rendered, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(rendered, (w // 2, h - 1), (w // 2, int(h * 0.42)), (255, 225, 70), 1, cv2.LINE_AA)
    draw_label(rendered, "Path borders", (18, 28), (0, 220, 255))
    draw_label(rendered, "Centerline", (18, 62), (0, 255, 105))
    draw_label(rendered, f"Auto: {overlay.command}   confidence: {overlay.confidence:.2f}", (18, h - 18), (245, 245, 245))
    return rendered


def create_simulated_scene() -> tuple[np.ndarray, PathOverlay]:
    now = time.time()
    phase = math.sin(now * 0.9)
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (32, 42, 50)
    frame[int(FRAME_H * 0.55):, :] = (47, 61, 52)
    horizon = int(FRAME_H * 0.43)
    for offset in range(-80, FRAME_W + 120, 90):
        sway = int(9 * math.sin(now + offset))
        cv2.line(frame, (offset + sway, horizon), (offset + 84 + sway, FRAME_H), (57, 74, 65), 1, cv2.LINE_AA)
    lb = int(FRAME_W * 0.24 + phase * 18)
    rb = int(FRAME_W * 0.76 + phase * 18)
    lt = int(FRAME_W * 0.43 + phase * 8)
    rt = int(FRAME_W * 0.57 + phase * 8)
    top_y = int(FRAME_H * 0.46)
    cv2.fillPoly(frame, np.array([[(lb, FRAME_H - 1), (lt, top_y), (rt, top_y), (rb, FRAME_H - 1)]], dtype=np.int32), (78, 69, 58))
    cv2.line(frame, (lb, FRAME_H - 1), (lt, top_y), (104, 111, 98), 2, cv2.LINE_AA)
    cv2.line(frame, (rb, FRAME_H - 1), (rt, top_y), (104, 111, 98), 2, cv2.LINE_AA)
    overlay = PathOverlay((lb, FRAME_H - 1, lt, top_y), (rb, FRAME_H - 1, rt, top_y), ((lb + rb) // 2, FRAME_H - 1, (lt + rt) // 2, top_y), 0, "forward", "Clean simulated path overlay", 1.0)
    return frame, overlay


def create_simulated_frame() -> np.ndarray:
    frame, _ = create_simulated_scene()
    return frame


def render_processed_frame(frame: np.ndarray, overlay: PathOverlay, detect_markers: bool = True) -> bytes:
    rendered = draw_overlay(frame, overlay, detect_markers=detect_markers)
    bird = perspective_transform(frame)
    bird = cv2.resize(bird, (150, 112))
    cv2.rectangle(rendered, (FRAME_W - 164, 12), (FRAME_W - 8, 132), (10, 18, 24), -1)
    rendered[18:130, FRAME_W - 158:FRAME_W - 8] = bird
    cv2.putText(rendered, "Perspective", (FRAME_W - 154, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("failed to encode processed frame")
    return jpeg.tobytes()


def process_frame(frame: np.ndarray) -> bytes:
    global latest_overlay
    resized = cv2.resize(frame, (FRAME_W, FRAME_H))
    latest_overlay = compute_overlay(resized)
    return render_processed_frame(resized, latest_overlay, detect_markers=True)


def simulated_jpeg(processed: bool) -> bytes:
    global latest_overlay
    frame, overlay = create_simulated_scene()
    if processed:
        latest_overlay = overlay
        return render_processed_frame(frame, overlay, detect_markers=False)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("failed to encode simulated frame")
    return jpeg.tobytes()


async def stream_frames(processed: bool):
    while True:
        if processed:
            frame_bytes = latest_frame_processed or simulated_jpeg(processed=True)
        else:
            frame_bytes = latest_frame_raw or simulated_jpeg(processed=False)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        await asyncio.sleep(0.08)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(Path(__file__).with_name("gui.html").read_text())


@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame_raw, latest_frame_processed, frame_source
    data = await request.json()
    frame_b64 = data.get("frame")
    if not frame_b64:
        return {"status": "no frame"}
    frame_bytes = base64.b64decode(frame_b64)
    latest_frame_raw = frame_bytes
    frame_source = "uploaded"
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "bad frame"}
    latest_frame_processed = await asyncio.get_running_loop().run_in_executor(None, process_frame, frame)
    return {"status": "ok", "auto_command": auto_command, "auto_error": auto_error}


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(stream_frames(processed=True), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_raw")
async def video_feed_raw():
    return StreamingResponse(stream_frames(processed=False), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/source_code_warehouse")
async def source_code_warehouse():
    return {"warehouse": SOURCE_CODE_WAREHOUSE, "final_selection": {"gui": "webVersion/gui.html", "backend": "webVersion/api.py", "overlay": "Hough Lines + estimated path borders + centerline + perspective inset + Hough circle markers"}}


@app.get("/status")
async def status():
    return {"controls": controls, "selected_direction": selected_direction, "manual_command": manual_command(), "motion_enabled": motion_enabled, "go": motion_enabled, "autonomous": autonomous_enabled, "auto_command": auto_command, "auto_error": auto_error, "lane_status": lane_status, "frame_source": frame_source, "overlay_confidence": latest_overlay.confidence, "user_log": list(event_log), "source_modules": SOURCE_CODE_WAREHOUSE}


@app.post("/go")
async def go():
    global motion_enabled, autonomous_enabled
    motion_enabled = True
    autonomous_enabled = False
    apply_motion_state()
    log_event(f"GO pressed. Running {selected_direction.upper()}.")
    return await status()


@app.post("/stop")
async def stop():
    global motion_enabled, autonomous_enabled, auto_command
    motion_enabled = False
    autonomous_enabled = False
    auto_command = "stop"
    reset_controls()
    log_event("STOP pressed. Rover motion halted.")
    return await status()


@app.post("/autonomous/start")
async def autonomous_start():
    global autonomous_enabled, motion_enabled
    motion_enabled = False
    autonomous_enabled = True
    reset_controls()
    log_event("Autonomous GO pressed. Vision overlay command active.")
    return await status()


@app.post("/autonomous/stop")
async def autonomous_stop():
    global autonomous_enabled
    autonomous_enabled = False
    log_event("Autonomous mode stopped.")
    return await status()


@app.post("/{direction}")
async def move(direction: str):
    global selected_direction, autonomous_enabled
    if direction not in controls:
        return {"error": "invalid direction", "valid": list(controls)}
    selected_direction = direction
    autonomous_enabled = False
    apply_motion_state()
    log_event(f"Direction {'changed to' if motion_enabled else 'selected:'} {direction.upper()}{'' if motion_enabled else '. Press GO to run.'}")
    return await status()


log_event("CEO prototype loaded with reusable Q3 vision code.")

if os.getenv("PWP_WARM_SIMULATION", "1") == "1":
    latest_frame_processed = simulated_jpeg(processed=True)
