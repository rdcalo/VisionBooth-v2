import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from gesture_detector import GestureDetector
from speech_handler import SpeechHandler
from filter_processor import FilterProcessor
import datetime
import time
from threading import Lock, Thread
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photobooth_secret'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25
)

gesture_detector = GestureDetector()
speech_handler   = SpeechHandler()
filter_processor = FilterProcessor()

try:
    gesture_detector.hands.min_detection_confidence = 0.6
    gesture_detector.hands.min_tracking_confidence  = 0.5
except Exception:
    pass

if not os.path.exists("sessions"):
    os.mkdir("sessions")
if not os.path.exists("static/templates"):
    os.makedirs("static/templates", exist_ok=True)

SESSION_DIR      = None
PHOTOS_PER_STRIP = 4
CONSECUTIVE_REQUIRED = 5

GESTURE_EVERY_N_FRAMES = 3
GESTURE_DETECT_WIDTH   = 320
STREAM_JPEG_QUALITY    = 60

# ── Global state ──────────────────────────────────────────────────────────────
current_state = {
    'screen': 'TIMER_DETECT',
    'phase':  'CAMERA',
    'timer_value':    None,
    'countdown_end':  None,
    'detected_gesture': None,
    'last_count':     None,
    'count_streak':   0,
    'thumb_up_streak': 0,
    'fist_streak':    0,
    'capture_count':      0,
    'captured_images':    [],
    'captured_filenames': [],
    'selected_template':    None,
    'selected_filter':      None,
    'final_strip_filename': None,
    'listening':         False,
    'expected_commands': [],
}

state_lock  = Lock()
mic_lock    = Lock()   # only one thread may use the microphone at a time

# Monotonic counter — each start_listening_* call increments this.
# The spawned thread captures its own copy; if the counter changes
# before the thread fires an emit, the thread knows it's stale and exits.
_listener_id      = 0
_listener_id_lock = Lock()

_frame_count  = 0
_last_gesture = None


# ── Mic helper ────────────────────────────────────────────────────────────────
def _guarded_listen(timeout=6):
    """Serialise all microphone access through mic_lock."""
    mic_lock.acquire(blocking=True)
    try:
        return speech_handler.listen_for_command(timeout=timeout)
    finally:
        mic_lock.release()


def _next_listener_id():
    global _listener_id
    with _listener_id_lock:
        _listener_id += 1
        return _listener_id


def _is_current_listener(my_id):
    with _listener_id_lock:
        return _listener_id == my_id


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/index')
def photobooth():
    return render_template('index.html')

@app.route('/retake')
def retake():
    return render_template('retake.html')

@app.route('/template_selection')
def template_selection():
    return render_template('template_selection.html')

@app.route('/filter_selection')
def filter_selection():
    return render_template('filter_selection.html')

@app.route('/final_preview')
def final_preview():
    return render_template('final_preview.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sessions/<path:filename>')
def serve_photo(filename):
    return send_from_directory('sessions', filename)

@socketio.on('request_session_info')
def handle_request_session_info():
    emit('session_info', {
        'selected_template': current_state.get('selected_template', 1),
        'selected_filter':   current_state.get('selected_filter', None),
    })
 

@socketio.on('new_session_request')
def handle_new_session():
    global SESSION_DIR
    # Create a fresh session directory
    SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(SESSION_DIR, exist_ok=True)
 
    # Full state reset — wipe everything from the previous session
    with state_lock:
        current_state.update({
            'screen':               'TIMER_DETECT',
            'phase':                'CAMERA',
            'timer_value':          None,
            'countdown_end':        None,
            'detected_gesture':     None,
            'last_count':           None,
            'count_streak':         0,
            'thumb_up_streak':      0,
            'fist_streak':          0,
            'capture_count':        0,
            'captured_images':      [],
            'captured_filenames':   [],
            'selected_template':    None,
            'selected_filter':      None,
            'final_strip_filename': None,
            'listening':            False,
            'expected_commands':    [],
        })
 
    # Invalidate any lingering listener threads
    _next_listener_id()
 
    print(f"New session started: {SESSION_DIR}")
    emit('session_reset', {'session': SESSION_DIR})
# ── Connect / disconnect ──────────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    global SESSION_DIR
    SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"Client connected. Session: {SESSION_DIR}")
    with state_lock:
        current_state['listening'] = False
    # Invalidate any running listener threads from a previous page
    _next_listener_id()
    emit('connected', {'session': SESSION_DIR})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
    with state_lock:
        current_state['listening'] = False
    _next_listener_id()


# ── Video frame ───────────────────────────────────────────────────────────────
@socketio.on('video_frame')
def handle_video_frame(data):
    global _frame_count, _last_gesture
    try:
        img_str  = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr    = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            emit('state_update', _default_state(data['image']))
            return

        _frame_count += 1

        if _frame_count % GESTURE_EVERY_N_FRAMES == 0:
            h, w  = frame.shape[:2]
            scale = GESTURE_DETECT_WIDTH / w
            small = cv2.resize(frame, (GESTURE_DETECT_WIDTH, int(h * scale)))
            try:
                small_annotated, gesture_name = gesture_detector.detect_gesture(small)
                frame = cv2.resize(small_annotated, (w, h))
            except Exception as e:
                print(f"Gesture detection error: {e}")
                gesture_name = None
            _last_gesture = gesture_name
        else:
            gesture_name = _last_gesture

        with state_lock:
            current_state['detected_gesture'] = gesture_name
            _process_camera_state_machine(gesture_name)
            ok, buf = cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
            )
            if not ok:
                return
            emit('state_update', {
                'frame':           f'data:image/jpeg;base64,{base64.b64encode(buf).decode()}',
                'state':           current_state.get('screen', 'CAMERA'),
                'timer_value':     current_state['timer_value'],
                'gesture':         gesture_name,
                'countdown':       _get_countdown(),
                'streak_progress': _get_streak_progress(),
                'trigger_capture': (current_state['screen'] == 'CAPTURE'),
                'capture_count':   current_state['capture_count'],
                'total_captures':  PHOTOS_PER_STRIP,
            })
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('state_update', _default_state(data.get('image', '')))


def _default_state(image):
    return {
        'frame': image,
        'state': current_state.get('screen', 'CAMERA'),
        'timer_value': current_state['timer_value'],
        'gesture': None,
        'countdown': _get_countdown(),
        'streak_progress': _get_streak_progress(),
        'trigger_capture': False,
        'capture_count': current_state['capture_count'],
        'total_captures': PHOTOS_PER_STRIP,
    }


# ── Camera state machine ──────────────────────────────────────────────────────
def _process_camera_state_machine(gesture_name):
    screen = current_state.get('screen', 'TIMER_DETECT')
    finger_count_map = {
        "One Finger": 1, "Peace Sign": 2, "Three Fingers": 3,
        "Four Fingers": 4, "Open Palm": 5,
    }
    detected_count = finger_count_map.get(gesture_name)
    fist_detected  = (gesture_name == "Fist")

    if screen == 'TIMER_DETECT':
        if detected_count and 1 <= detected_count <= 5:
            current_state.update({'screen': 'TIMER_DETECTING', 'last_count': detected_count, 'count_streak': 1})

    elif screen == 'TIMER_DETECTING':
        if detected_count and 1 <= detected_count <= 5:
            if detected_count == current_state['last_count']:
                current_state['count_streak'] += 1
                if current_state['count_streak'] >= CONSECUTIVE_REQUIRED:
                    current_state.update({
                        'timer_value': detected_count, 'screen': 'TIMER_SET',
                        'count_streak': 0, 'thumb_up_streak': 0, 'fist_streak': 0,
                    })
                    print(f"✓ Timer set to: {detected_count}s")
            else:
                current_state.update({'count_streak': 1, 'last_count': detected_count})
        elif gesture_name is None:
            _reset_camera_state()

    elif screen == 'TIMER_SET':
        current_state['screen'] = 'TIMER_READY'

    elif screen == 'TIMER_READY':
        if fist_detected:
            current_state['fist_streak'] += 1
            if current_state['fist_streak'] >= CONSECUTIVE_REQUIRED:
                _reset_camera_state()
        else:
            current_state['fist_streak'] = 0

    elif screen == 'COUNTDOWN':
        if _get_countdown() == 0:
            current_state.update({'screen': 'CAPTURE', 'countdown_end': None})


def _reset_camera_state():
    current_state.update({
        'screen': 'TIMER_DETECT', 'timer_value': None, 'countdown_end': None,
        'detected_gesture': None, 'last_count': None, 'count_streak': 0,
        'thumb_up_streak': 0, 'fist_streak': 0,
    })


def _get_countdown():
    if current_state.get('screen') == 'COUNTDOWN' and current_state['countdown_end']:
        return max(0, int(round(current_state['countdown_end'] - time.time())))
    return None


def _get_streak_progress():
    if current_state.get('screen') == 'TIMER_DETECTING':
        return {'current': current_state['count_streak'], 'required': CONSECUTIVE_REQUIRED}
    return None


# ── Countdown ticker ──────────────────────────────────────────────────────────
def _countdown_ticker():
    last_sent = None
    while True:
        try:
            with state_lock:
                screen = current_state.get('screen')
            if screen == 'COUNTDOWN':
                val = _get_countdown()
                if val != last_sent:
                    socketio.emit('countdown_tick', {'countdown': val})
                    last_sent = val
                if val == 0:
                    last_sent = None
            else:
                last_sent = None
        except Exception as e:
            print(f"Countdown ticker error: {e}")
        time.sleep(0.2)


# ── Photo capture ─────────────────────────────────────────────────────────────
@socketio.on('save_photo')
def handle_save_photo(data):
    global SESSION_DIR
    try:
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            return
        if SESSION_DIR is None or not os.path.exists(SESSION_DIR):
            SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(SESSION_DIR, exist_ok=True)
        img_data = data.get('image')
        if not img_data:
            emit('photo_error', {'error': 'No image data'})
            return
        current_state['captured_images'].append(img_data)
        current_state['capture_count'] += 1
        photo_filename = f"{SESSION_DIR}/photo_{current_state['capture_count']}.png"
        img_bytes = base64.b64decode(img_data.split(',')[1])
        with open(photo_filename, 'wb') as f:
            f.write(img_bytes)
        current_state['captured_filenames'].append(photo_filename)
        emit('photo_received', {'count': current_state['capture_count'], 'total': PHOTOS_PER_STRIP})
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            print("All photos captured.")
            emit('photos_captured', {'count': current_state['capture_count']})
        else:
            time.sleep(1)
            current_state.update({
                'countdown_end': time.time() + current_state['timer_value'],
                'screen': 'COUNTDOWN',
            })
    except Exception as e:
        print(f"Error saving photo: {e}")
        emit('photo_error', {'error': str(e)})


# ── Retake ────────────────────────────────────────────────────────────────────
@socketio.on('request_retake_photos')
def handle_request_retake_photos():
    emit('photos_for_retake', {'photos': current_state['captured_images']})


@socketio.on('start_listening_retake')
def handle_start_listening_retake(data):
    sid      = request.sid
    my_id    = _next_listener_id()
    expected = list(data.get('expected_commands', ['YES', 'NO']))
    with state_lock:
        current_state['listening'] = True

    def listen():
        for attempt in range(4):
            if not _is_current_listener(my_id):
                print(f"[retake] listener {my_id} is stale, exiting")
                break
            recognized_text, _ = _guarded_listen(6)
            if not _is_current_listener(my_id):
                break
            matched_cmd, is_match = speech_handler.validate_command(recognized_text, expected)
            print(f"[retake] attempt {attempt+1}: got '{recognized_text}' → match={is_match} cmd={matched_cmd}")
            socketio.emit('voice_detected_retake', {
                'detected_text': recognized_text, 'is_valid': is_match, 'matched_command': matched_cmd,
            }, room=sid)
            if is_match:
                break
            if attempt < 3:
                time.sleep(0.3)
        with state_lock:
            current_state['listening'] = False

    Thread(target=listen, daemon=True).start()


@socketio.on('user_retake_decision')
def handle_retake_decision(data):
    action = data.get('action')
    if action == 'retake':
        with state_lock:
            current_state['capture_count']      = 0
            current_state['captured_images']    = []
            current_state['captured_filenames'] = []
            _reset_camera_state()
        emit('retake_response', {'action': 'retake'})
    elif action == 'continue':
        emit('retake_response', {'action': 'continue'})


# ── Template selection ────────────────────────────────────────────────────────
@socketio.on('start_listening_template')
def handle_start_listening_template(data):
    sid      = request.sid
    my_id    = _next_listener_id()   # invalidates any previous listener thread
    expected = list(data.get('expected_commands', ['CLASSIC', 'FUN', 'ELEGANT']))
    with state_lock:
        current_state['listening'] = True
        current_state['expected_commands'] = expected

    def listen():
        for attempt in range(4):
            if not _is_current_listener(my_id):
                print(f"[template] listener {my_id} is stale, exiting")
                break
            recognized_text, _ = _guarded_listen(6)
            if not _is_current_listener(my_id):
                break
            matched_cmd, is_match = speech_handler.validate_command(recognized_text, expected)
            print(f"[template] listener={my_id} attempt {attempt+1}: got '{recognized_text}' → match={is_match} cmd={matched_cmd}")
            socketio.emit('voice_detected_template', {
                'detected_text': recognized_text, 'is_valid': is_match, 'matched_command': matched_cmd,
            }, room=sid)
            if is_match:
                break
            if attempt < 3:
                time.sleep(0.3)
        with state_lock:
            current_state['listening'] = False

    Thread(target=listen, daemon=True).start()


@socketio.on('select_template')
def handle_select_template(data):
    template = data.get('template')
    try:
        template = int(template)
    except (TypeError, ValueError):
        pass
    current_state['selected_template'] = template
    print(f"Template selected: {template}")
    speech_handler.speak(f"Template {template} selected")
    emit('template_selected', {'template': template})


# ── Filter selection ──────────────────────────────────────────────────────────
@socketio.on('start_listening_filter')
def handle_start_listening_filter(data):
    sid      = request.sid
    my_id    = _next_listener_id()
    expected = list(data.get('expected_commands', ['NORMAL', 'COLOR', 'BLACK AND WHITE', 'BLACK WHITE']))
    with state_lock:
        current_state['listening'] = True
        current_state['expected_commands'] = expected

    def listen():
        for attempt in range(4):
            if not _is_current_listener(my_id):
                print(f"[filter] listener {my_id} is stale, exiting")
                break
            recognized_text, _ = _guarded_listen(6)
            if not _is_current_listener(my_id):
                break
            matched_cmd, is_match = speech_handler.validate_command(recognized_text, expected)
            print(f"[filter] listener={my_id} attempt {attempt+1}: got '{recognized_text}' → match={is_match} cmd={matched_cmd}")
            socketio.emit('voice_detected_filter', {
                'detected_text': recognized_text, 'is_valid': is_match, 'matched_command': matched_cmd,
            }, room=sid)
            if is_match:
                break
            if attempt < 3:
                time.sleep(0.3)
        with state_lock:
            current_state['listening'] = False

    Thread(target=listen, daemon=True).start()


@socketio.on('select_filter')
def handle_select_filter(data):
    filter_type = data.get('filter')
    current_state['selected_filter'] = filter_type
    print(f"Filter selected: {filter_type}")
    speech_handler.speak(f"{filter_type} filter selected")
    emit('filter_selected', {'filter': filter_type})


# ── Final preview ─────────────────────────────────────────────────────────────
@socketio.on('request_final_preview')
def handle_request_final_preview():
    try:
        template_num  = current_state.get('selected_template', 1)
        filter_type   = current_state.get('selected_filter', 'normal')
        template_path = f"static/images/template_{template_num}.png"
        photo_paths   = current_state.get('captured_filenames', [])

        if not os.path.exists(template_path):
            emit('strip_generation_error', {'error': f'Template not found: {template_path}'})
            return
        if len(photo_paths) < PHOTOS_PER_STRIP:
            emit('strip_generation_error', {'error': 'Not all photos captured'})
            return

        final_strip = filter_processor.composite_strip_with_template(
            template_path, photo_paths, filter_type=filter_type
        )
        if final_strip is None:
            emit('strip_generation_error', {'error': 'Failed to process photos'})
            return

        now        = datetime.datetime.now()
        filename   = f"final_strip_{now.strftime('%Y%m%d_%H%M%S')}.png"
        final_path = os.path.join(SESSION_DIR, filename)
        final_strip.save(final_path, dpi=(300, 300), quality=95)
        current_state['final_strip_filename'] = (
            f"sessions/{SESSION_DIR.split('/')[-1]}/{filename}"
        )
        print(f"Final strip created: {final_path}")
        emit('strip_preview_ready', {'filename': current_state['final_strip_filename']})

    except Exception as e:
        print(f"Error generating preview: {e}")
        import traceback
        traceback.print_exc()
        emit('strip_generation_error', {'error': str(e)})


# ── Voice relay (camera START) ────────────────────────────────────────────────
@socketio.on('voice_detected_camera')
def handle_voice_detected_camera(data):
    detected_text = data.get('detected_text', '').upper()
    if 'START' in detected_text:
        with state_lock:
            if current_state['screen'] == 'TIMER_READY' and current_state['timer_value']:
                current_state.update({
                    'screen': 'COUNTDOWN',
                    'countdown_end': time.time() + current_state['timer_value'],
                    'thumb_up_streak': 0, 'fist_streak': 0,
                })
                speech_handler.speak(f"Starting {current_state['timer_value']} second countdown")
                emit('countdown_started', {'timer_value': current_state['timer_value']})


# ── Background camera voice listener ─────────────────────────────────────────
def _camera_voice_listener():
    while True:
        try:
            with state_lock:
                screen    = current_state.get('screen')
                listening = current_state.get('listening', False)

            if screen != 'TIMER_READY' or listening:
                time.sleep(0.5)
                continue

            if not mic_lock.acquire(blocking=False):
                time.sleep(0.3)
                continue

            try:
                recognized_text, _ = speech_handler.listen_for_command(timeout=2)
            finally:
                mic_lock.release()

            if recognized_text and 'START' in recognized_text.upper():
                with state_lock:
                    if current_state['screen'] == 'TIMER_READY' and current_state['timer_value']:
                        current_state.update({
                            'screen': 'COUNTDOWN',
                            'countdown_end': time.time() + current_state['timer_value'],
                            'thumb_up_streak': 0, 'fist_streak': 0,
                        })
                        print(f"▶ Countdown started via voice: {current_state['timer_value']}s")
                        speech_handler.speak("Starting countdown")

            time.sleep(0.3)

        except Exception as e:
            print(f"Camera voice listener error: {e}")
            time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    Thread(target=_camera_voice_listener, daemon=True).start()
    Thread(target=_countdown_ticker,      daemon=True).start()

    print("=" * 50)
    print("VisionBooth Starting...")
    print("Open browser at: http://localhost:5000")
    print("=" * 50)
    socketio.run(
        app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True
    )