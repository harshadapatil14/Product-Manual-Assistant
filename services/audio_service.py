"""
Audio service for voice input and text-to-speech functionality
"""
import speech_recognition as sr
import pyttsx3
import tempfile
import os
from typing import Optional
from config.settings import AUDIO_SETTINGS


class AudioService:
    """Handles voice input and text-to-speech operations"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {str(e)}")
            self.tts_engine = None
    
    def _configure_tts(self):
        """Configure text-to-speech engine settings"""
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('rate', AUDIO_SETTINGS['default_rate'])
                self.tts_engine.setProperty('volume', AUDIO_SETTINGS['default_volume'])
                
                # Get available voices and set a good one
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find a female voice, otherwise use the first one
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    else:
                        self.tts_engine.setProperty('voice', voices[0].id)
                
                print(f"✅ TTS configured - Rate: {AUDIO_SETTINGS['default_rate']}, Volume: {AUDIO_SETTINGS['default_volume']}")
            except Exception as e:
                print(f"❌ Error configuring TTS: {str(e)}")
    
    def record_audio(self) -> Optional[sr.AudioData]:
        """
        Record audio from microphone
        
        Returns:
            AudioData object if successful, None otherwise
        """
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... Speak your question now!")
                audio = self.recognizer.listen(
                    source, 
                    timeout=AUDIO_SETTINGS['timeout'], 
                    phrase_time_limit=AUDIO_SETTINGS['phrase_time_limit']
                )
                print("✅ Audio recorded!")
                return audio
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            return None
    
    def speech_to_text(self, audio: sr.AudioData) -> Optional[str]:
        """
        Convert speech to text using Google Speech Recognition
        
        Args:
            audio: AudioData object from microphone
            
        Returns:
            Transcribed text if successful, None otherwise
        """
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
    
    def text_to_speech(self, text: str, rate: int = None) -> Optional[str]:
        """
        Convert text to speech and save as temporary audio file
        
        Args:
            text: Text to convert to speech
            rate: Speech rate (optional, uses default if not provided)
            
        Returns:
            Path to temporary audio file if successful, None otherwise
        """
        if not self.tts_engine:
            print("❌ TTS engine not available")
            return None
            
        if not text or not text.strip():
            print("❌ No text provided for TTS")
            return None
            
        try:
            print(f"🔊 Converting text to speech: {text[:50]}...")
            
            # Set rate if provided
            original_rate = self.tts_engine.getProperty('rate')
            if rate:
                self.tts_engine.setProperty('rate', rate)
            
            # Create temporary file with proper extension
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='w+b')
            temp_file.close()
            
            # Save to file
            self.tts_engine.save_to_file(text, temp_file.name)
            self.tts_engine.runAndWait()
            
            # Reset to original rate
            if rate:
                self.tts_engine.setProperty('rate', original_rate)
            
            # Check if file was created and has content
            if os.path.exists(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                print(f"✅ Audio file created: {temp_file.name} ({os.path.getsize(temp_file.name)} bytes)")
                return temp_file.name
            else:
                print("❌ Audio file was not created or is empty")
                return None
                
        except Exception as e:
            print(f"❌ Error in text-to-speech: {str(e)}")
            return None
    
    def cleanup_audio_file(self, file_path: str):
        """
        Clean up temporary audio file
        
        Args:
            file_path: Path to temporary audio file
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                print(f"✅ Cleaned up audio file: {file_path}")
        except Exception as e:
            print(f"❌ Error cleaning up audio file: {str(e)}")
    
    def get_audio_bytes(self, file_path: str) -> Optional[bytes]:
        """
        Read audio file and return bytes
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio bytes if successful, None otherwise
        """
        try:
            if not file_path or not os.path.exists(file_path):
                print(f"❌ Audio file not found: {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
                print(f"✅ Audio bytes read: {len(audio_bytes)} bytes")
                return audio_bytes
        except Exception as e:
            print(f"❌ Error reading audio file: {str(e)}")
            return None
    
    def test_tts(self) -> bool:
        """
        Test if TTS is working properly
        
        Returns:
            True if TTS is working, False otherwise
        """
        try:
            if not self.tts_engine:
                print("❌ TTS engine not available")
                return False
                
            test_text = "Hello, this is a test of the text to speech system."
            audio_file = self.text_to_speech(test_text)
            
            if audio_file and os.path.exists(audio_file):
                print("✅ TTS test successful")
                self.cleanup_audio_file(audio_file)
                return True
            else:
                print("❌ TTS test failed")
                return False
                
        except Exception as e:
            print(f"❌ TTS test error: {str(e)}")
            return False 