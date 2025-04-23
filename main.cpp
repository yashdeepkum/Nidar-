#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

std::vector<std::string> load_class_list(const std::string& path) {
    std::vector<std::string> class_list;
    std::ifstream ifs(path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& class_list) {
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float CONFIDENCE_THRESHOLD = 0.5;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.5;

    cv::Mat blob;
    auto input_image = format_yolov5(image);
    input_image.convertTo(input_image, CV_32F);
    cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD && class_id.x == 0) { // only 'person'
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (int i : nms_result) {
        Detection det;
        det.class_id = class_ids[i];
        det.confidence = confidences[i];
        det.box = boxes[i];
        output.push_back(det);

        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Person: " + cv::format("%.2f", det.confidence);
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

int main() {
    std::string model_path = "/home/kumrawat/Documents/yolov5s_fixed.onnx";  // your patched ONNX model
    std::string label_path = "/home/kumrawat/Documents/coco-labels-2014_2017.txt";    // COCO labels
    std::vector<std::string> class_list = load_class_list(label_path);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap(0); // default webcam
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open webcam\n";
        return -1;
    }

    std::cout << "Press ESC to exit\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        std::vector<Detection> results;
        detect(frame, net, results, class_list);

        cv::imshow("Person Detection", frame);

        if (cv::waitKey(1) == 27) { // ESC key
            std::cout << "Exiting...\n";
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
