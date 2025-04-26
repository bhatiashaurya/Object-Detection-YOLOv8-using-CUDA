# Object-Detection-YOLOv8-using-CUDA
An optimized C++ implementation of **YOLOv8** with **CUDA** acceleration and **OpenCV** integration.  
This project aims to achieve fast and accurate object detection on the GPU using lightweight C++ and CUDA code.

## 📂 Project Structure
```
YOLOv8-CUDA-CPP/
├── CMakeLists.txt
├── include/
├── src/
├── build/ (generated after compilation)
└── README.md
```

## ⚙️ Requirements
- Visual Studio 2019 (with C++ Desktop Development tools installed)
- CMake 3.20+
- MSYS2 (for `mingw32-make` if building outside Visual Studio)
- OpenCV 4.x (prebuilt with VC16)
- CUDA Toolkit 11.x or newer
- GPU with CUDA compute capability (e.g., NVIDIA GTX 1050 Ti, RTX 3060, etc.)

## 🔥 Setup Instructions

### 1. Install OpenCV
Download prebuilt OpenCV for Windows:  
[https://github.com/opencv/opencv/releases](https://github.com/opencv/opencv/releases)  
Extract it to `C:\opencv\`  
Make sure `OpenCVConfig.cmake` is located at:  
```
C:/opencv/opencv/build/x64/vc16/lib/OpenCVConfig.cmake
```

### 2. Clone This Repository
```bash
git clone https://github.com/bhatiashaurya/Object-Detection-YOLOv8-using-CUDA.git
cd Object-Detection-YOLOv8-using-CUDA
```

### 3. Build Using Visual Studio 2019
- Open `Object-Detection-YOLOv8-using-CUDA` folder in Visual Studio 2019.
- Go to **CMake → Change CMake Settings → YOLOv8_CUDA_CPP**.
- Set `OpenCV_DIR` to:
  ```
  C:/opencv/opencv/build/x64/vc16/lib
  ```
- Save settings.
- Click **Build → Build Solution**.
- Run the executable from the `build/` folder.

### 4. Alternative Build Using Terminal (MinGW)
```bash
cd YOLOv8-CUDA-CPP
mkdir build
cd build
cmake -G "MinGW Makefiles" -D OpenCV_DIR="C:/opencv/opencv/build/x64/vc16/lib" ..
mingw32-make
```

## 🚀 Run
After successful build:
```bash
./Object-Detection-YOLOv8-using-CUDA.exe
```
Pass an image or video as input to test object detection.

## ✨ Features
- YOLOv8 object detection
- CUDA-accelerated inference
- Minimal C++ code
- OpenCV real-time image display
- Support for batch processing

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributions
Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.
