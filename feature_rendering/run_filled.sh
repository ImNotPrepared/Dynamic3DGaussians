python get_features_filled.py --device cuda:0 --seq 1 &
python get_features_filled.py --device cuda:0 --seq 2 &
python get_features_filled.py --device cuda:4 --seq 3 &
python get_features_filled.py --device cuda:7 --seq 4 &


cp /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/1/filled_box_image.npy /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_registered/undist_cam01
cp /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/2/filled_box_image.npy /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_registered/undist_cam02
cp /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/3/filled_box_image.npy /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_registered/undist_cam03
cp /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/4/filled_box_image.npy /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_registered/undist_cam04