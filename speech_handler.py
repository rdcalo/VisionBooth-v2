import speech_recognition as sr
import pyttsx3
import threading


class SpeechHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

        # Lock ensures only one speak() runs at a time
        self._speak_lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        """Initialize (or re-initialize) the TTS engine."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"[TTS] Engine init error: {e}")
            self.engine = None

    # ------------------------------------------------------------------
    # LISTENING
    # ------------------------------------------------------------------

    def listen_for_command(self, timeout=5):
        """
        Listen for a voice command with timeout.
        Returns: (detected_text, is_valid)
        """
        try:
            with sr.Microphone() as source:
                print(f"Listening for command (timeout: {timeout}s)...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=5
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

        except sr.RequestError:
            print("Microphone error")
            return None, False
        except Exception as e:
            print(f"Listen error: {e}")
            return None, False

    def validate_command(self, recognized_text, expected_commands):
        """
        Validate recognized text against a list of expected commands.
        Returns: (matched_command, is_match)
        """
        if not recognized_text:
            return None, False

        recognized_text = recognized_text.upper().strip()

        for cmd in expected_commands:
            if cmd.upper() in recognized_text:
                return cmd, True

        return None, False

    # ------------------------------------------------------------------
    # SPEAKING
    # ------------------------------------------------------------------

    def speak(self, text):
        """
        Speak text non-blocking.
        Uses a non-blocking lock so if the engine is already speaking
        the new request is silently dropped instead of crashing.
        """
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def _speak(self, text):
        # Try to acquire the lock without blocking.
        # If another speak() is already running, just skip this one.
        if not self._speak_lock.acquire(blocking=False):
            print(f"[TTS] Skipped (engine busy): '{text}'")
            return

        try:
            # Re-initialize the engine each time to avoid stale loop state.
            # pyttsx3's runAndWait() can leave the engine in a broken state
            # after an exception; re-init is the safest recovery.
            self._init_engine()
            if self.engine is None:
                return

            self.engine.say(text)
            self.engine.runAndWait()

        except RuntimeError as e:
            # 'run loop already started' — should not happen with the lock,
            # but guard defensively just in case.
            print(f"[TTS] RuntimeError (suppressed): {e}")
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            # Always release so the next speak() can proceed.
            self._speak_lock.release()

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def close(self):
        """Clean up TTS engine resources."""
        try:
            self.engine.stop()
        except Exception:
            pass