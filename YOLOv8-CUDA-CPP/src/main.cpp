lear#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

namespace yolodetector {

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXYOLOv8");

vector<string> loadClassNames(const string& filename) {
    vector<string> classNames;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

__global__ void normalizeKernel(uchar* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i] / 255.0f;
    }
}

vector<float> preprocess(Mat& frame, const Size& inputSize) {
    Mat resized;
    resize(frame, resized, inputSize);

    int imgSize = resized.total() * resized.channels();
    uchar* d_input;
    float* d_output;
    vector<float> output(imgSize);

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize * sizeof(float));

    cudaMemcpy(d_input, resized.data, imgSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    normalizeKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, imgSize);

    cudaMemcpy(output.data(), d_output, imgSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

void drawPredictions(Mat& frame, const vector<float>& output, const vector<string>& classNames) {
    for (size_t i = 0; i < output.size(); i += 6) {
        float conf = output[i + 4];
        if (conf > 0.5f) {
            int classId = static_cast<int>(output[i + 5]);
            int x = static_cast<int>(output[i]);
            int y = static_cast<int>(output[i + 1]);
            int w = static_cast<int>(output[i + 2]);
            int h = static_cast<int>(output[i + 3]);
            rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
            putText(frame, classNames[classId], Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }
    }
}

void runYOLO(const string& modelPath, const string& classFile) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    OrtCUDAProviderOptions cuda_options;
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);
    vector<string> classNames = loadClassNames(classFile);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam." << endl;
        return;
    }

    const int inputHeight = 640;
    const int inputWidth = 640;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        vector<float> inputTensorValues = preprocess(frame, Size(inputWidth, inputHeight));
        array<int64_t, 4> inputShape{1, 3, inputHeight, inputWidth};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

        vector<const char*> inputNames = {session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
        vector<const char*> outputNames = {session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};

        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);
        float* outputData = outputTensors[0].GetTensorMutableData<float>();

        vector<float> result(outputData, outputData + outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
        drawPredictions(frame, result, classNames);

        imshow("YOLOv8 Object Detection - CUDA", frame);
        if (waitKey(1) == 27) break; // ESC to exit
    }
    cap.release();
    destroyAllWindows();
}

} // namespace yolodetector

int main() {
    yolodetector::runYOLO("yolov8.onnx", "coco.names");
    return 0;
}
