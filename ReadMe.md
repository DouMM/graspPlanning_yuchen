## in progress ...
## Use deep learning to realize grasp pose recognition.

### step１: training graspNet network
ssh tams144

source /opt/ros/indigo/setup.bash

cd catkin_ws

source devel/setup.bash

cd /informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/graspPlanning_yuchen

CUDA_VISIBLE_DEVICES=0 python training_yuchen.py

CUDA_VISIBLE_DEVICES=1 python training_yuchen.py

