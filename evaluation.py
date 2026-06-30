from main import *
from PPG import *
import os
import pathlib

# store data as 
# [subject #, gt hr, gt hr using peak detection, '' rfft, method 0 (rFFT, periodogram, peak detection), ...]

def generate_results(vid_folder, save_folder=None):
    folder_name = pathlib.Path(vid_folder).name
    subject_num = int(folder_name[7:])
    gt_dir = os.path.join(vid_folder, 'ground_truth.txt')
    video_dir = os.path.join(vid_folder, 'vid.avi')
    (num_frames, fps), _, data = analyze_video(video_dir, True)

    average_signal = data.mean(axis=0) # take average across ROI's: (3, N)
    # get BVPs that are num_frames long.
    BVPs = [
        green_only(average_signal[1], fps, show_graph=False),
        ratio_method(average_signal, 0, fps, show_graph=False),
        ratio_method(average_signal, 2, fps, show_graph=False),
        CHROM_method(average_signal, fps, show_graph=False),
        CHROM_method_windowed(average_signal, fps, show_graph=False),
        POS_method(average_signal, fps, show_graph=False),
        POS_method_windowed(average_signal, fps, show_graph=False)
    ]

    post_processings = [
        lambda bvp: rFFT(bvp, fps)[0],
        lambda bvp: periodogram(bvp, fps)[0],
        lambda bvp: peak_detection(bvp, fps)[0]
    ]

    # segment so that each segment is 10 seconds with 1 second step

    # get ground truth arrays
    try:
        # MATLAB's dlmread is similar to numpy.loadtxt or pandas.read_csv with delimiter
        gt_data = np.loadtxt(gt_dir)
        gt_trace = gt_data[0, :].T # 1st row, transpose PPG Signal
        gt_time = gt_data[2, :].T # 3rd row, transpose Time data
        gt_hr = gt_data[1, :].T # 2nd row, transpose Heart rate Data
        print(f"\nLoaded ground truth from: {gt_dir} (DATASET_2 format)")
        print(gt_data.shape)
    except Exception as e:
        print(f"Error reading {gt_dir}: {e}")

    window_len = 10 * fps # 10 seconds
    step = fps  # 1 second
    start_frame = 0

    windows = []
    if num_frames < window_len: 
        windows.append((0, num_frames))
    while start_frame + window_len <= num_frames:
        if start_frame + step + window_len > num_frames:
            end_frame = num_frames
        else:
            end_frame = start_frame + window_len
            
        # window = arr[start_frame:end_frame]
        windows.append((start_frame, end_frame))
        start_frame += step
    
    # len(windows) data points
    results = np.zeros((len(windows), 4+len(BVPs)*len(post_processings)))
    results[:, 0] = subject_num # set the first column to subject_num

    for i, window in enumerate(windows):
        # for each segment, get the corresponding ground truth array of pairs (time, hr)
        gt_hr_avg = get_avg_hr(gt_hr, gt_time, window[0]/fps, window[1]/fps)
        gt_hr_peak = gt_heartrate_peak(gt_trace, gt_time, window[0]/fps, window[1]/fps)
        gt_hr_fft = gt_heartrate_fft(gt_trace, gt_time, window[0]/fps, window[1]/fps)

        # calculate estimated heart rate using different methods
        est_hr = [post_processing(bvp[window[0]:window[1]]) for bvp in BVPs for post_processing in post_processings]

        # store data
        results[i, 1] = gt_hr_avg
        results[i, 2] = gt_hr_peak
        results[i, 3] = gt_hr_fft
        results[i, 4:] = est_hr
    
    # print(f"{num_frames=}")
    # print(f"{len(windows)=}")
    # print(f"{results.shape=}")

    if save_folder is not None:
        np.savetxt(pathlib.Path(save_folder) / f"{folder_name}.csv", results, delimiter=",", fmt="%.2f")

    return results

if __name__ == "__main__":
    # results = generate_results("subject1")
    # np.savetxt("output.csv", results, delimiter=",", fmt="%.2f")
    data = np.loadtxt('output.csv', delimiter=',')
    error = data[:, 3]-data[:, -2]
    print(np.std(error, ddof=1))
    plt.hist(error)
    plt.show()