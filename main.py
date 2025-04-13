import cv2
import pytesseract
import speech_recognition as sr
import moviepy.editor as mp
import shutil
import os


def extract_text_from_frames(video_path, sample_rate=30, debug=True):
    """Extracts text from video frames using enhanced preprocessing and Tesseract OCR.

    Args:
        video_path (str): Path to the video file.
        sample_rate (int): Process 1 frame per this many frames (reduces processing).
        debug (bool): If True, saves processed frames for debugging.

    Returns:
        list: List of extracted text from processed frames, with frame numbers.
    """
    import cv2
    import pytesseract
    import numpy as np
    import os

    # Configure pytesseract for better text detection
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()\- "'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {total_frames} frames, {fps} FPS")
    print(f"Processing approximately {total_frames // sample_rate} frames")

    results = []
    frame_count = 0

    # Create debug directory if needed
    if debug and not os.path.exists("debug_frames"):
        os.makedirs("debug_frames")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process only every Nth frame based on sample_rate
        if frame_count % sample_rate != 0:
            continue

        # Image preprocessing pipeline
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # 3. Apply adaptive thresholding instead of global
        thresh = cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 4. Try both normal and inverted images (text could be white-on-black or black-on-white)
        text_normal = pytesseract.image_to_string(thresh, config=custom_config).strip()
        text_inverted = pytesseract.image_to_string(255 - thresh, config=custom_config).strip()

        # Choose the result with more characters as likely better
        text = text_normal if len(text_normal) > len(text_inverted) else text_inverted

        # Only save non-empty results
        if text:
            timestamp = frame_count / fps
            results.append({
                "frame": frame_count,
                "time": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                "text": text
            })
            print(f"Frame {frame_count} ({results[-1]['time']}): Text found")

        # Save debug images if requested
        if debug and text:
            print("wtf")
            cv2.imwrite(f"debug_frames/frame_{frame_count}_original.jpg", frame)
            cv2.imwrite(f"debug_frames/frame_{frame_count}_processed.jpg", thresh)

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, found text in {len(results)} frames")
    return results


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
