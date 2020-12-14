#ifndef _STEREO_DETECTOR_H_
#define _STEREO_DETECTOR_H_

// Opencv
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
// SIFT descriptor // extractor
#include <opencv2/xfeatures2d.hpp>

#include "Detector.h" // <- the other opencv includes are here
#include "cone.h"

// ZED
#include <sl/Camera.hpp>

using namespace cv;

class Stereo_Detector{
    private:
    // ============================================================================
    // Create window
    const std::string iWName = "Left";
    const std::string oWName = "Right";
    const std::string dWName = "Depth";
    const std::string sWName = "crops";

    // ZED camera properties
    // doubled focal length and pixle size since we are using
    // half of the resolution
    const float dist_right = 120.0; // In milimeters
    const float focal_lent = 2.8;   
    const float pixle_size = 0.004; // 0.002 if we use full scale

    // SIFT setting
    const int minHessian = 400;
    const float ratio_thresh  = 0.7f;

    // Show window or not
    int show_window = 0;

    cv::Ptr<cv::xfeatures2d::SIFT> sift_detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    Detector yolov = Detector(0, 0);

    void avg_list_point(sl::Mat&, sl::float4&, std::vector<Point>&, 
                        size_t, int, int);
    void sift_triangulate(cv::Mat, cv::Mat, cone&, Point, Point, int);

    public:
    // Default constructor;
    Stereo_Detector(int, int, int);

    void detection( cv::Mat&, cv::Mat&, sl::Mat&, std::vector<cone>&);

};

#endif//_STEREO_DETECTOR_H_H
// ============================================================================