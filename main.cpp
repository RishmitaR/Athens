#include <opencv2/opencv.hpp>
#include <fstream>

const float confThreshold = 0.2;
const float nmsThreshold = 0.4;
const int inpWidth = 416;
const int inpHeight = 416;

std::vector<std::string> classes;
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame) {
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
     
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
     
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    std::cout << label <<std::endl;
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
     
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
     
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<std::string> names;
    if (names.empty()){
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


cv::dnn::Net loadDarknet(cv::Mat& img, cv::Mat& blob) {
     double scale=1; 

    // Load names of classes
    std::string classesFile = "coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) { classes.push_back(line); }
    
    // Give the configuration and weight files for the model
    std::string modelConfiguration = "yolov2-tiny.cfg";
    std::string modelWeights = "yolov2-tiny.weights";
    
    // Load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::dnn::blobFromImage(img, blob, (float) 1/255, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    return net;
}

void effeciencyInfo(cv::Mat& img, cv::dnn::Net& net) {
    // Put efficiency information. The function getPerfProfile returns the 
    // overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    cv::putText(img, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
}



int main(int argc, char *argv[]) {
    cv::Mat img;
    
    if(argc < 1) {
        cv::VideoCapture capture(0);
        if (!capture.isOpened()) { //check if video device has been initialised
            std::cout << "cannot open camera";
        }
        
        capture.read(img);
    } else {
        img = cv::imread(argv[1]);
    }    


    cv::Mat blob;
    cv::dnn::Net net = loadDarknet(img, blob);
   
    
    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    
    // Remove the bounding boxes with low confidence
    postprocess(img, outs);
    
    effeciencyInfo(img, net);

    cv::imshow("ahh", img);

    if(cv::waitKey(0) >= 0) {
        return 0;
    }
}
