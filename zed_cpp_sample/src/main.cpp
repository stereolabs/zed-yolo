#include <iostream>
#include <queue>
#include <string>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <mutex>              
#include <condition_variable>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include "yolo_v2_class.hpp" 

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/opencv.hpp>

#include <sl/Camera.hpp>

#include "GLViewer.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0.6;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
constexpr int INFERENCE_SIZE = 416;

const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof (colors) / sizeof (colors[0]);

std::mutex data_lock;
cv::Mat cur_frame;
std::vector<cv::Mat> result_vect;
std::atomic<bool> exit_flag, new_data;

class bbox_t_3d {
public:
    bbox_t bbox;
    sl::float3 coord;

    bbox_t_3d(bbox_t bbox_, sl::float3 coord_) {
        bbox = bbox_;
        coord = coord_;
    }
};

float getMedian(std::vector<float> &v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

std::vector<bbox_t_3d> getObjectDepth(std::vector<bbox_t> &bbox_vect, sl::Mat &xyzrgba) {
    sl::float4 out(NAN, NAN, NAN, NAN);
    bool valid_measure;
    int i, j;
    const int R_max = 4;

    std::vector<bbox_t_3d> bbox3d_vect;

    for (auto &it : bbox_vect) {

        int center_i = it.x + it.w * 0.5f, center_j = it.y + it.h * 0.5f;

        std::vector<float> x_vect, y_vect, z_vect;
        for (int R = 0; R < R_max; R++) {
            for (int y = -R; y <= R; y++) {
                for (int x = -R; x <= R; x++) {
                    i = center_i + x;
                    j = center_j + y;
                    xyzrgba.getValue<sl::float4>(i, j, &out, sl::MEM::GPU);
                    valid_measure = std::isfinite(out.z);
                    if (valid_measure) {
                        x_vect.push_back(out.x);
                        y_vect.push_back(out.y);
                        z_vect.push_back(out.z);
                    }
                }
            }
        }

        if (x_vect.size() * y_vect.size() * z_vect.size() > 0) {
            float x_med = getMedian(x_vect);
            float y_med = getMedian(y_vect);
            float z_med = getMedian(z_vect);

            bbox3d_vect.emplace_back(it, sl::float3(x_med, y_med, z_med));
        }
    }

    return bbox3d_vect;
}


void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

// Main function
int main(int argc, char** argv) {
    std::string names_file = "coco.names";
    std::string cfg_file = "yolov4.cfg";
    std::string weights_file = "yolov4.weights";
    std::string filename;
    
    if (argc > 3) { //voc.names yolo-voc.cfg yolo-voc.weights svo_file.svo
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
        if (argc > 4)
            filename = argv[4];
    } else if (argc > 1) filename = argv[1];

    std::vector<std::string> class_names;
    {
        std::ifstream class_file(names_file);
        if (!class_file) {
            for (int i = 0; i < NUM_CLASSES; i++)
                class_names.push_back(std::to_string(i));
        } else {
            std::string line;
            while (std::getline(class_file, line))
                class_names.push_back(line);
        }
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD1080;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; 
    
    if (!filename.empty()) init_parameters.svo_input_filename.set(filename.c_str());

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    // Let's define the model as custom box object to specify that the inference is done externally
    detection_parameters.detection_model = sl::DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);

    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;
    // sl::Resolution image_size = zed.getCameraInformation().camera_resolution;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;

    sl::Mat left_sl, depth_image, point_cloud;

    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();

    auto net = cv::dnn::readNetFromDarknet(cfg_file, weights_file);

    //To run the code with CUDA GPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // To run the code with CPU
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;

    exit_flag = false;
    int frame_count = 0;

    while (!exit_flag) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

            zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            zed.retrieveImage(depth_image, sl::VIEW::DEPTH);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

            // Preparing inference
            cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv_rgba, frame, cv::COLOR_BGRA2BGR);

            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(INFERENCE_SIZE, INFERENCE_SIZE), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            net.forward(detections, output_names);
            
            std::cout << "******Frame No : " << frame_count << std::endl;

            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<cv::Point> circs[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];

            for (auto& output : detections) {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++) {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);
                    cv::Point circ(x, y);

                    for (int c = 0; c < NUM_CLASSES; c++) {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD) {
                            boxes[c].push_back(rect);
                            circs[c].push_back(circ);
                            scores[c].push_back(confidence);
                        }
                    }
                }
            }

            for (int c = 0; c < NUM_CLASSES; c++)
                cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

            std::vector<sl::CustomBoxObjectData> objects_in;
            for (int c = 0; c < NUM_CLASSES; c++) {
                for (size_t i = 0; i < indices[c].size(); ++i) {
                    const auto color = colors[c % NUM_COLORS];

                    auto idx = indices[c][i];
                    const auto& rect = boxes[c][idx];
                    const auto& circ = circs[c][idx];
                    auto& rect_score = scores[c][idx];

                    // Fill the detections into the correct format
                    sl::CustomBoxObjectData tmp;
                    tmp.unique_object_id = sl::generate_unique_id();
                    tmp.probability = rect_score;
                    tmp.label = c;
                    tmp.bounding_box_2d = cvt(rect);
                    tmp.is_grounded = (c == 0); 
                    objects_in.push_back(tmp);
                    //--

                    cv::rectangle(frame, rect, color, 3);
                    cv::circle(frame,circ,5,color,3);

                    sl::float4 point_cloud_value;

                    point_cloud.getValue(circ.x, circ.y, &point_cloud_value);
                    float distance = sqrt(point_cloud_value.x*point_cloud_value.x + point_cloud_value.y*point_cloud_value.y + point_cloud_value.z*point_cloud_value.z);

                    std::ostringstream label_ss, dist_ss;
                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                    auto label = label_ss.str();

                    int baseline;
                    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x  + label_bg_sz.width+ 25, rect.y+20), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));

                    std::string dist = std::to_string(distance/1000);

                    dist_ss << "Distance: " << std::fixed << std::setprecision(2) << distance/1000;
                    auto dist_label = dist_ss.str();

                    std::cout << label.c_str() << "  " << dist_label.c_str() << " m" << std::endl;

                    cv::putText(frame, dist_label, cv::Point(rect.x, rect.y + baseline + 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            cv::imshow("Objects", frame);

            frame_count++;

            int key = cv::waitKey(10); 
            if (key == 'q') exit_flag = true;

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
        }
    }
    return 0;
}
