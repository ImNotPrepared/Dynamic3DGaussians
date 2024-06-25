import cv2
import os
import re

def to_videos(image_folder, video_name, frame_rate=30, index=None):
    # Get all image files regardless of their extensions
    valid_extensions = [".jpg", ".jpeg", ".png"]
    images = [img for img in os.listdir(image_folder) if os.path.splitext(img)[1].lower() in valid_extensions]

    if not images:
        raise ValueError("No images found in the folder.")
    print(f'creating a video w/ {len(images)} frames!')
    # Sort images (customize this part if you have specific sorting needs)
    def extract_number(filename):
        numbers = re.findall('\d+', filename)
        return int(numbers[0]) if numbers else None

    # Sorting the list using the numerical part of the filenames
    images = sorted(images, key=extract_number)


    # Read the first image to get the video dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        raise IOError(f"Could not read image {images[0]}")

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    # Write images to video
    for image in images[index[0]:index[1]]:
        frame = cv2.imread(os.path.join(image_folder, image))
        print('loaded', os.path.join(image_folder, image))
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Could not read image {image}. It's being skipped.")

    # Release the video writer
    video.release()

#to_videos('/data3/zihanwa3/Capstone-DSR/dynibar/preprocessing/dataset/dense/images_128x128', './groundtruth.avi')
#to_videos('/data3/zihanwa3/Capstone-DSR/dynibar/monocular/test1_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/0/_270000', './training_view.avi')
#to_videos('/data3/zihanwa3/Capstone-DSR/dynibar/monocular/test1_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/0/_610000/videos/0/rgb_out', './novel_view_0.avi', frame_rate=5)
#to_videos('/data3/zihanwa3/Capstone-DSR/dynibar/monocular/test1_mr-42_w-disp-0.100_w-flow-0.010_anneal_cycle-0.1-0.1-w_mode-0/700/_610000/videos/700/rgb_out', './novel_view_700.avi', frame_rate=5)

#to_videos('/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/ims/0', './novel_view_700.avi', frame_rate=30)

to_videos('/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam01', './gt_1.avi', frame_rate=5, index=[183,  294])
to_videos('/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam02', './gt_2.avi', frame_rate=5, index=[183,  294])
to_videos('/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam03', './gt_3.avi', frame_rate=5, index=[183,  294])
to_videos('/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam04', './gt_4.avi', frame_rate=5, index=[183,  294])

