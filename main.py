import cv2
import dlib
import time
import datetime
from tqdm import tqdm
from utils.utils import *
from utils.methods import *
from utils.post_processing import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import scipy.signal
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode


def record_video(camera_fps: int, recording_duration: int, output_dir: str):
    """
    Records a video that will be analyzed later.

    Inputs:
        camera_fps (int): fps of the camera being used 
        recording_duration (int): number of seconds to record for
        output_dir (str): directory to save the mp4 video
    """
    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir, fourcc, camera_fps, (frame_width, frame_height))

    record = False

    total_frames = recording_duration * camera_fps
    frame_count = 0

    print("Press r to record, q to quit.")

    while cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            print("Camera not detected")

        if record:
            if (frame_count >= total_frames):
                break
            out.write(frame)
            cv2.imshow(f"Recording...", frame)
            frame_count += 1
        else:
            cv2.imshow("Press r to record", frame)
        
        k = cv2.waitKey(1) & 0xFF 
        if k != -1:
            if k == ord('q'):
                print("Exiting")
                break
            elif k == ord('r'):
                record = True

    cam.release()
    out.release()
    cv2.destroyAllWindows()

def analyze_video(video_source: str, show_frame: bool = False):
    """
    Reads a video file and returns a dict object containing average rgb value for each frame.
    Uses MediaPipe Tasks API (mediapipe >= 0.10) for face landmark detection.

    Requires: face_landmarker.task model file in working directory.
    Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

    Output: tuple(
            video data (tuple): tuple of (total frames, fps)
            undetected frames (np.ndarray): array of length (video length), marked 1 if ROI's undetected for that frame,
            data (np.ndarray): (3, 3, video length) matrix, data = [left cheek, right cheek, forehead], with
                each being a (3, video length) matrix of (R, G, B) data
        )
    """

    def get_rgb_average(frame, x, y, width, height):
        h_frame, w_frame = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + width), min(h_frame, y + height)
        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(roi, axis=(0, 1))[::-1]  # BGR -> RGB

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened")

    # --- MediaPipe Tasks API setup ---
    base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,   # VIDEO mode enables inter-frame tracking
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    undetected_frames = np.zeros(total_frames)
    data = np.zeros((3, 3, total_frames))

    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            print("DID NOT RETURN FRAME: analyze_video")
            break

        h, w = frame.shape[:2]

        # VIDEO mode requires a monotonically increasing timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            undetected_frames[i] = 1
            continue

        lm = result.face_landmarks[0]  # normalised landmarks, x/y in [0, 1]

        def px(idx):
            """Convert normalised landmark to pixel coordinates."""
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # Left cheek (dlib 2→MP 234, dlib 41→MP 110, dlib 48→MP 61)
        x = px(423)[0]  # inner edge (near nose)
        y = px(347)[1]  # top edge (under eye)
        width = px(411)[0] - px(423)[0] # inner to outer
        height = px(426)[1] - px(347)[1]    # top to bottom
        data[0, :, i] = get_rgb_average(frame, x, y, width, height)
        if show_frame:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Right cheek (dlib 54→MP 291, dlib 46→MP 340, dlib 14→MP 454)
        x = px(187)[0] # outer edge (near ear)
        y = px(118)[1] # top edge (under eye)
        width = px(203)[0] - px(187)[0] # outer to inner
        height = px(206)[1] - px(118)[1] # top to bottom
        data[1, :, i] = get_rgb_average(frame, x, y, width, height)
        if show_frame:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Forehead (x: (104, 333), y:(10, 151))
        width  = px(299)[0] - px(69)[0]
        height = px(151)[1] - px(10)[1]
        x = px(69)[0]
        y = px(10)[1]
        data[2, :, i] = get_rgb_average(frame, x, y, width, height)
        if show_frame:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        if show_frame:
            cv2.imshow("Video", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                raise KeyboardInterrupt("Exited")

    cap.release()
    face_landmarker.close()     # release MediaPipe resources
    cv2.destroyAllWindows()

    # Substitute missing frames via linear interpolation across each channel
    frame_indices = np.arange(total_frames)
    detected_mask = undetected_frames == 0

    if np.any(~detected_mask):  # only run if there are gaps to fill
        for roi in range(data.shape[0]):       # left cheek, right cheek, forehead
            for channel in range(data.shape[1]):  # R, G, B
                signal = data[roi, channel, :]
                # interpolate at undetected positions using detected frames
                signal[~detected_mask] = np.interp(
                    frame_indices[~detected_mask],  # x positions to fill
                    frame_indices[detected_mask],   # known x positions
                    signal[detected_mask]           # known values
                )
                data[roi, channel, :] = signal

    return (total_frames, fps), undetected_frames, data

def evaluate(signal:np.ndarray, fps:int, results:dict, method_name:str, post_process_method:str, **kwargs):
    """
    Evaluates the 3d color signal using the chosen method, stores data in results

    The method specified by method_name returns a 1d BVP (Blood Volume Pulse) array
    The method specified by post_process_method returns a tuple (bpm, (graph_data_x, graph_data_y))

    The data is stored in results as a 2d dictionary: post_process_method -> method_name -> data

    Inputs:
        signal (np.ndarray): (3, 3, N) array containing data (ROI, color, frame length)
        fps (int): fps of the recording
        results (dict): dictionary to store the data
        method_name (str): name of the rPPG method to use
        post_process_method (str): name of the method to use to calculate heart rate from the BVP signal
    """
    BVP = None
    show_graph = kwargs["show_graph"] if "show_graph" in kwargs else False
    average_signal = signal.mean(axis=0) # take average across ROI's: (3, N)
    # average_signal = signal[2] # forehead only
    match method_name:
        case "GREEN":
            BVP = green_only(average_signal[1], fps, show_graph=show_graph)
        case "Green/Red":
            BVP = ratio_method(average_signal, 0, fps, show_graph=show_graph)
        case "Green/Blue":
            BVP = ratio_method(average_signal, 2, fps, show_graph=show_graph)
        case "CHROM":
            BVP = CHROM_method(average_signal, fps, show_graph=show_graph)
        case "CHROM Windowed":
            BVP = CHROM_method_windowed(average_signal, fps, show_graph=show_graph)
        case "POS":
            BVP = POS_method(average_signal, fps, show_graph=show_graph)
        case "POS Windowed":
            BVP = POS_method_windowed(average_signal, fps, show_graph=show_graph)
            # bvps = [POS_method_windowed(signal[roi], fps) for roi in range(3)]
            # bvps = [b / (np.std(b) + 1e-8) for b in bvps]
            # BVP = np.mean(bvps, axis=0)
            # BVP = bandpass_filter(scipy.signal.detrend(BVP), fps)
        case _:
            raise ValueError(f"Method name {method_name} not recognized")
    
    if BVP is None:
        raise ValueError("BVP is None after method")
    
    data = None
    match post_process_method:
        case "rFFT":
            data = rFFT(BVP, fps, show_graph=show_graph)
        case "periodogram":
            data = periodogram(BVP, fps, show_graph=show_graph)
        case "peak":
            data = peak_detection(BVP, fps, show_graph=show_graph)
        case _:
            raise ValueError(f"Post Processing method {post_process_method} not recognized")

    if data is None:
        raise ValueError("data is None after processing")
    if post_process_method not in results:
        results[post_process_method] = {}
    results[post_process_method][method_name] = data


if __name__ == "__main__":
    # set some variables
    camera_fps = 30     # needs to be manually checked
    recording_duration = 15
    video_dir = "vid.avi"
    data_dir = "data.pkl"
    data = None

    """
    Start point values:
    0: record video
    1: analyze a recorded video using video_dir
    2: calculate
    """
    start_point = 0

    # Record video
    if (start_point <= 0):
        record_video(camera_fps, recording_duration, video_dir)

    # Analyze and save data
    if (start_point <= 1):
        video_data, undetected_frames, data = analyze_video(video_dir, show_frame=True)
        print(f"# of undetected frames: {np.sum(undetected_frames)}")
        # if more than 5% of video has undetected frames, don't save video
        if np.sum(undetected_frames) >= int(video_data[0] * 0.05):
            print("There was an error while processing video.")
        else:
            with open(data_dir, 'wb') as file:
                pickle.dump((video_data, data), file)
    
    # open data and display
    if (start_point <= 2):
        if data is None:
            with open(data_dir, 'rb') as file:
                (video_data, data) = pickle.load(file)

        camera_fps = video_data[1]
        print(f"{camera_fps=}")

        results = {}
        show_graph = False
        method_names = ["GREEN", "Green/Red", "Green/Blue", "CHROM", "CHROM Windowed", "POS", "POS Windowed"]
        post_processing_methods = ["rFFT", "periodogram", "peak"]
        
        for method_name in method_names:
            for post_processing_method in post_processing_methods:
                evaluate(data, camera_fps, results, method_name, post_processing_method)

        for ppm in results:
            interactive_graph(results[ppm], f"Results: {ppm}")
            print()
    
