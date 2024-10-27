import cv2
import pytesseract
import speech_recognition as sr
import moviepy.editor as mp
import shutil
import os

def extract_text_from_frames(video_path):
    """Extracts text from frames of a video using Tesseract OCR.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: List of extracted text from each frame.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return []

    text_list = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh_frame)
        print("admahajan "+ text)
        text_list.append(text)

    cap.release()
    cv2.destroyAllWindows()

    return text_list

def extract_audio_transcript(video_path):
    """Extracts audio transcript from a video using SpeechRecognition and MoviePy.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: Transcribed text from the audio.
    """

    recognizer = sr.Recognizer()

    try:
        # Ensure FFmpeg is installed or path is set in the environment variable
        if not os.environ.get("IMAGEIO_FFMPEG_EXE"):
            # Check for FFmpeg installation (replace commands with appropriate ones for your system)
            if not shutil.which("ffmpeg"):
                print("FFmpeg not found. Please install FFmpeg or set the IMAGEIO_FFMPEG_EXE environment variable.")
                return ""

        # Extract audio using MoviePy
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio.write_audiofile("extracted_audio.wav")

        with sr.AudioFile("extracted_audio.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))
    except Exception as e:  # Catching general exceptions for error handling
        print(f"An error occurred during audio extraction or recognition: {e}")
    finally:
        # Cleanup temporary audio file if it exists
        try:
            os.remove("extracted_audio.wav")
        except FileNotFoundError:
            pass  # Ignore if the file doesn't exist

    return ""

if __name__ == "__main__":
    video_path = "/Users/adityamahajan/Downloads/test.mp4"  # Replace with your video path

    text_list = extract_text_from_frames(video_path)
    audio_transcript = extract_audio_transcript(video_path)

    print("Text from frames:")
    for text in text_list:
        print(text)

    print("\nAudio Transcript:")
    print(audio_transcript)