# Dual Codebook- and Scene Change-Based Loop Closure Detection Using SVM Outlier Removal in Verification

Loop closure detection (LCD) is an important issue in simultaneous localization and mapping (SLAM) systems for correcting accumulated errors and maintaining map consistency. In this code, a novel dual codebook- and scene change-based LCD algorithm is proposed via our fast support vector machine (SVM)-based outlier removal method used in the LCD verification phase. 

## Requirements

- CUDA
- RAFT
- cmake == 3.28.4
- eigen >= 3.4.0
- opencv-contrib >= 4.10.0
- libyaml-cpp >= 0.7.0
- boost >= 1.74

## Usage 
### Build the execution file
```
bash ./build.sh
```
### Run loop closure detection
```
./build/LCDEngine --parameters /root/LCD/LCDEngine/params.yml\
    --img_list /root/LCD/LCD_data/kitti00/image_list.txt\
    --des-path /root/LCD/LCD_data/des9_bin/kitti00\
    -o /root/LCD/LCD_data/experimental_result/kitti00/original_with_dual_with_scch_no_svm_topk2.txt\
    --topk 2\
```

