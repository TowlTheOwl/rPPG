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

def analyze_video(video_source:str, video_fps:int, video_duration:int, show_frame:bool=False):
    """
    Reads a video file and returns a dict object containing average rgb value for each frame

    Output dict: maps region_name -> list[tuple (r, g, b), ...]
    """
    cap = cv2.VideoCapture(video_source)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    total_frames = video_duration * video_fps

    undetected_frame = False
    data = RGBData()

    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            print("DID NOT RETURN FRAME: analyze_video")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 1)

        if not len(faces) > 0:
            undetected_frame = True
            break # there is a frame where a face cannot be detected.
        else:
            face = faces[0]
            landmarks = predictor(gray, face)

            face_left = face.left()
            face_top = face.top()
            face_right = face.right()
            face_bottom = face.bottom()

            # extract forehead and cheeks
            # left cheek (third between points x: (2, 48), y: (41, 48))
            width = int((landmarks.part(48).x - landmarks.part(2).x)/3)
            height = int((landmarks.part(48).y - landmarks.part(41).y)/3)
            x = landmarks.part(2).x + width
            y = landmarks.part(41).y + height

            rgb_avg = get_rgb_average(frame, x, y, width, height)
            data.data["left"].append(rgb_avg)

            if show_frame: cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # right cheek (third between x: (54, 14), y: (46, 54))
            width = int((landmarks.part(14).x - landmarks.part(54).x)/3)
            height = int((landmarks.part(54).y - landmarks.part(46).y)/3)
            x = landmarks.part(54).x + width
            y = landmarks.part(46).y + height

            rgb_avg = get_rgb_average(frame, x, y, width, height)
            data.data["right"].append(rgb_avg)

            if show_frame: cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # forehead (x: (21, 22), third between y: (22, 27))
            width = int((landmarks.part(22).x - landmarks.part(21).x))
            height = int((landmarks.part(27).y - landmarks.part(22).y))
            x = landmarks.part(21).x
            y = landmarks.part(22).y - int(1.5 * height)

            rgb_avg = get_rgb_average(frame, x, y, width, height)
            data.data["forehead"].append(rgb_avg)

            if show_frame: cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)


            # cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 2)

        if show_frame:
            cv2.imshow("Video", frame)
        k = cv2.waitKey(1) & 0xFF 
        if k != -1:
            if k == ord('q'):
                print("Exiting")
                break

    cap.release()
    cv2.destroyAllWindows()

    roi_values = data.get_data()
    roi_signals = np.array([
        roi_values["left"],
        roi_values["right"],
        roi_values["forehead"]
    ])  # (3, N, 3)
    return (not undetected_frame), roi_signals

def evaluate(signal_3d:np.ndarray, fps:int, results:dict, method_name:str, post_process_method:str, **kwargs):
    """
    Evaluates the 3d color signal using the chosen method, stores data in results

    The method specified by method_name returns a 1d BVP (Blood Volume Pulse) array
    The method specified by post_process_method returns a tuple (bpm, (graph_data_x, graph_data_y))

    The data is stored in results as a 2d dictionary: post_process_method -> method_name -> data

    Inputs:
        signal_3d (np.ndarray): (3, N) array containing data
        fps (int): fps of the recording
        results (dict): dictionary to store the data
        method_name (str): name of the rPPG method to use
        post_process_method (str): name of the method to use to calculate heart rate from the BVP signal
    """
    BVP = None
    show_graph = kwargs["show_graph"] if "show_graph" in kwargs else False
    match method_name:
        case "GREEN":
            BVP = green_only(signal_3d[1], fps)
        case "Green/Red":
            BVP = ratio_method(signal_3d, 0, fps, show_graph)
        case "Green/Blue":
            BVP = ratio_method(signal_3d, 2, fps, show_graph)
        case "CHROM":
            BVP = CHROM_method(signal_3d, fps, show_graph)
        case "CHROM Windowed":
            BVP = CHROM_method_windowed(signal_3d, fps, show_graph)
        case "POS":
            BVP = POS_method(signal_3d, fps, show_graph)
        case "POS Windowed":
            BVP = POS_method_windowed(signal_3d, fps, show_graph)
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
    recording_duration = 10
    video_dir = "output.mp4"
    data_dir = "data.pkl"
    data = None

    """
    Start point values:
    0: record video
    1: analyze a recorded video using video_dir
    2: calculate
    """
    start_point = 2

    # Record video
    if (start_point <= 0):
        record_video(camera_fps, recording_duration, video_dir)

    # Analyze and save data
    if (start_point <= 1):
        ret, data = analyze_video(video_dir, 30, recording_duration, False)
        if not ret:
            print("There was an error while processing video.")
        else:
            with open(data_dir, 'wb') as file:
                pickle.dump(data, file)
    
    # open data and display
    if (start_point <= 2):
        if data is None:
            with open(data_dir, 'rb') as file:
                data:np.ndarray = pickle.load(file)

        roi_signals = data.mean(axis=0)  # (N, 3)
        color_values = roi_signals.T # now we have (3, N) matrix of color values

        results = {}
        show_graph = False
        method_names = ["GREEN", "Green/Red", "Green/Blue", "CHROM", "CHROM Windowed", "POS", "POS Windowed"]
        post_processing_methods = ["rFFT", "periodogram", "peak"]
        
        for method_name in method_names:
            for post_processing_method in post_processing_methods:
                evaluate(color_values, camera_fps, results, method_name, post_processing_method)

        for ppm in results:
            interactive_graph(results[ppm], f"Results: {ppm}")
            print()
    
