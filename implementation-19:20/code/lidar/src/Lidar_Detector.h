#ifndef _LIDAR_DETECTOR_H_
#define _LIDAR_DETECTOR_H_

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>

// For downsampling
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

// For elapsed time
#include <chrono>

// For ground removal
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>

// For Euclidean Clustering
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// For image extraction
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Classification
#include <tiny_dnn/tiny_dnn.h>
#include "cone.h"

#include <fstream>

using namespace cv;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

class Lidar_Detector{

    private:
        // int nGridX = 6;
        // int nGridZ = 6;
        // int nImageSw = 32;
        // int nImageSh = 32;
        // float fThresh = 0.05f;
        int nGridX;
        int nGridZ;
        int nImageSw;
        int nImageSh;
        float fThresh;

        // Create a network 
        network<sequential> net;

        inline void print_point(pcl::PointXYZRGB p ){
            std::cout << " X: " << p.x
                      << " Y: " << p.y 
                      << " Z:" << p.z << std::endl;
        };
        
        void view_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

        void downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr);
        
        void ground_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, 
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr);

        void euclidean_cluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
                    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>&);

        void project_to_image(pcl::PointCloud<pcl::PointXYZRGB>::Ptr, 
                                            cv::Mat&);

        void get_cone_list(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>&,
                   std::vector<cone>&);
    public:
        // Default constructor
        Lidar_Detector();

        // Parametrized Constructor with default values
        Lidar_Detector( const std::string& weight,
                        int imageSw, 
                        int imageSh, 
                        int gridx  ,
                        int gridz  , 
                        float thresh);

        void detection( pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud,
                        std::vector<cone>& cone_collection );
};

#endif  //_LIDAR_DETECTOR_H_
