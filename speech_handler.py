import speech_recognition as sr
import pyttsx3
import threading


FUZZY_MAP = {
    'CLASSIC': ['classic', 'classics', 'class', 'classy', 'classical'],
    'FUN':     ['fun', 'funny', 'fund', 'fan'],
    'ELEGANT': ['elegant', 'elegance', 'elegantly', 'element'],

    'NORMAL':          ['normal', 'colour', 'color', 'colored'],
    'COLOR':           ['color', 'colour', 'colored', 'normal', 'cola'],
    'BLACK AND WHITE': ['black and white', 'black & white', 'blackandwhite',
                        'black white', 'black in white', 'black n white'],
    'BLACK WHITE':     ['black white', 'black and white', 'black & white'],

    'YES':   ['yes', 'yeah', 'yep', 'yea', 'ya', 'sure', 'ok', 'okay'],
    'NO':    ['no', 'nope', 'nah', 'na', 'know', 'not'],
    'START': ['start', 'star', 'started', 'sport', 'dart'],
}


class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        # ── Tuned for speed ───────────────────────────────────────────────
        # Fixed threshold — dynamic calibration on Windows sets it too high
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False

        # How long a silence ends a phrase — 0.5s is plenty for single words
        self.recognizer.pause_threshold = 0.5

        # Minimum audio length to attempt recognition — lower = faster
        self.recognizer.phrase_threshold = 0.2

        # Silence before the phrase that's still included in the audio chunk
        self.recognizer.non_speaking_duration = 0.3
        # ─────────────────────────────────────────────────────────────────

        self._speak_lock  = threading.Lock()
        self._calibrated  = False
        self._calib_lock  = threading.Lock()

        self._init_engine()

        # Calibrate once in a background thread so startup isn't blocked
        threading.Thread(target=self._calibrate_once, daemon=True).start()

    def _calibrate_once(self):
        """
        Do a single 1-second ambient noise calibration at startup.
        This sets energy_threshold to a good value for the current room
        without the repeated per-call overhead that caused slowdowns.
        After calibration, dynamic adjustment is locked off so subsequent
        calls are not affected.
        """
        with self._calib_lock:
            if self._calibrated:
                return
            try:
                with sr.Microphone() as source:
                    print("[SpeechHandler] Calibrating microphone...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    # Clamp: never go above 1500 (too high blocks short words)
                    if self.recognizer.energy_threshold > 1500:
                        self.recognizer.energy_threshold = 1500
                    print(f"[SpeechHandler] Calibrated. energy_threshold="
                          f"{self.recognizer.energy_threshold:.0f}")
                self._calibrated = True
            except Exception as e:
                print(f"[SpeechHandler] Calibration failed (using default): {e}")
                self._calibrated = True   # don't retry

    def _init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"[TTS] Engine init error: {e}")
            self.engine = None

    # ── LISTENING ─────────────────────────────────────────────────────────────

    def listen_for_command(self, timeout=4):
        """
        Listen for a single voice command.
        timeout     : max seconds to wait for speech to START  (was 6, now 4)
        phrase_time_limit: max seconds of actual speech to record (2s is enough
                           for any command word used in VisionBooth)
        """
        try:
            with sr.Microphone() as source:
                print(f"Listening for command (timeout: {timeout}s)...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=2      # was 5 — commands are 1-3 words
                )
            try:
                text = self.recognizer.recognize_google(audio).upper().strip()
                print(f"Recognized: {text}")
                return text, True
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None, False
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None, False

        except sr.WaitTimeoutError:
            print("Listen error: timed out waiting for phrase")
            return None, False
        except Exception as e:
            print(f"Listen error: {e}")
            return None, False

    # ── VALIDATION ────────────────────────────────────────────────────────────

    def validate_command(self, recognized_text, expected_commands):
        if not recognized_text:
            return None, False
        text = recognized_text.upper().strip()
        for cmd in expected_commands:
            cmd_upper = cmd.upper()
            if cmd_upper in text or text in cmd_upper:
                return cmd, True
            for variant in FUZZY_MAP.get(cmd_upper, []):
                if variant.upper() in text or text == variant.upper():
                    return cmd, True
        return None, False

    # ── SPEAKING ──────────────────────────────────────────────────────────────

    def speak(self, text):
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        if not self._speak_lock.acquire(blocking=False):
            print(f"[TTS] Skipped (engine busy): '{text}'")
            return
        try:
            self._init_engine()
            if self.engine is None:
                return
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError as e:
            print(f"[TTS] RuntimeError (suppressed): {e}")
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self._speak_lock.release()

    def close(self):
        try:
            self.engine.stop()
        except Exception:
            pass