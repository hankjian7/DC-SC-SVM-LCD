# Dual Codebook- and Scene Change-Based Loop Closure Detection Using SVM Outlier Removal in Verification

Loop closure detection (LCD) is an important issue in simultaneous localization and mapping (SLAM) systems for correcting accumulated errors and maintaining map consistency. In this code, a novel dual codebook- and scene change-based LCD algorithm is proposed via our fast support vector machine (SVM)-based outlier removal method used in the LCD verification phase. 

## Environments

- **CUDA:** 12.5
- **CMake:** 3.28.4
- **Opencv-Contrib:** 4.10.0
- **Eigen:** 3.4.0
- **libyaml-cpp:** 0.7.0
- **Boost:** 1.74

## Environment

### 1. CUDA
This project utilizes the RAFT library for codeword mapping and requires CUDA support. We **highly recommend** using Docker to set up a CUDA environment. You can download a Docker image matching your CUDA version here:

[Docker Hub - NVIDIA CUDA Tags](https://hub.docker.com/r/nvidia/cuda/tags)

### 2. CMake
CMake version 3.28.4 has been tested and confirmed to work with CUDA 12.5. Download this specific version from the following link:

[Kitware CMake Releases](https://github.com/Kitware/CMake/releases)

### 3. OpenCV-Contrib
The project uses the SURF descriptor, so ensure you enable non-free modules by setting OPENCV_ENABLE_NONFREE=ON when building OpenCV-Contrib. You can find the required repositories here:

- [OpenCV Repository](https://github.com/opencv/opencv)
- [OpenCV-Contrib Repository](https://github.com/opencv/opencv_contrib)

### 4. Other Dependencies
Install these dependencies:

```bash
$ apt install git libeigen-dev libyaml-cpp-dev libboost-dev
```
## Building the Execution File
```bash
$ git clone https://github.com/hankjian7/DC-SC-SVM-LCD.git
$ cd DC-SC-SVM-LCD
$ bash ./build.sh
```

## Preparing Testing Data
### Cite Centre
The images captured by the left camera are used to test. Download link:

[CC Dataset](https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/data.html)

### Creating an Image List File
To use the dataset, create a text file that specifies the image folder path on the first line, followed by all image file names. Below is an example:
```
/root/LCD/LCD_data/CC/left_image
0000.jpg
0001.jpg
0002.jpg
0003.jpg
0004.jpg
0005.jpg
0006.jpg
0007.jpg
...
```
### Super Features, Codebook, and SVM Model
We extract super features from all dataset and use their codebook as our primary codebook based on the following repository: 

[Learning Super-Features for Image Retrieval](https://github.com/naver/FIRe?tab=readme-ov-file)

Additionally, we provide the **SVM model**, along with the **codebook** and **test super features** from the City Centre dataset. You can access them here:

[DC-SC-SVM-LCD Data](https://drive.google.com/drive/folders/1KjUwPw_jBxBGoRN3eaXtgFEekJWRxH5J?usp=drive_link)

## Run Loop Closure Detection
```bash
$ bash ./run_LCD.sh
```

