#include "Lidar_Detector.h"

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>

// For elapsed time
#include <chrono>

// A very quick method of printing out some results
void print_result(std::vector<cone>& cone_collection){

    // print list of detectijon
  for (size_t i = 0; i < cone_collection.size(); i++){
    std::cout << " Class: " << cone_collection[i].cone_class
              << " Accuy: " << cone_collection[i].cone_accuy
              << " X: " << cone_collection[i].x
              << " Y: " << cone_collection[i].y
              << " Z: " << cone_collection[i].z
              << std::endl;
  }
  
    // Print all yellow cones
  std::cout << "Yellow Cones:" << std::endl;
  for (size_t i = 0; i < cone_collection.size(); i++){
    if (cone_collection[i].cone_class == 0 ) 
      std::cout << "(" << cone_collection[i].x << "," << -cone_collection[i].z << "),";
  }
  std::cout << std::endl;

  std::cout << "Blue Cones:" << std::endl;
    // Print all blue cones
  for (size_t i = 0; i < cone_collection.size(); i++){
    if (cone_collection[i].cone_class == 1 ) 
      std::cout << "(" << cone_collection[i].x << "," << -cone_collection[i].z << "),";
  }
  std::cout << std::endl;

  std::cout << "Orange Cones:" << std::endl;
    // Print all blue cones
  for (size_t i = 0; i < cone_collection.size(); i++){
    if (cone_collection[i].cone_class == 2 ) 
      std::cout << "(" << cone_collection[i].x << "," << -cone_collection[i].z << "),";
  }
  std::cout << std::endl;


  std::cout << "Unknows:" << std::endl;
    // Print all blue cones
  for (size_t i = 0; i < cone_collection.size(); i++){
    if (cone_collection[i].cone_class == -1 ) 
      std::cout << "(" << cone_collection[i].x << "," << -cone_collection[i].z << "),";
  }
  std::cout << std::endl;
}

// display a point cloud to window
void view_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::visualization::CloudViewer viewer ("Cloud Viewer");
  viewer.showCloud(cloud);
  while(!viewer.wasStopped()){};
}

int main(int argc, char** argv){

  // READ PCD FILE
   // The original input/file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Begin the timer for pcd reading time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();

  // Read the input pcd file if there is one
  if (argc > 1){
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *pcloud) == -1){
      // load the file
      PCL_ERROR( "Couldn't read file. \n");
    }
  }else{
    std::cout << "No input .pcd file!" << std::endl;
    std::cout << "Usage: ./perception < pcd file >" << std::endl;
    return (0);
  }

  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  std::cout << "PCD LOADING TIME = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  std::cout << "Loaded "
            << pcloud->width * pcloud ->height
            << " data points from the test.pcd"
            << std::endl;

  // define the detector with the necessary arguments
  Lidar_Detector ld("res/net_weights", 32, 32, 6, 6, 0.05f);
  // List of result

  std::vector<cone> cone_collection;
  // Run detector
  ld.detection(pcloud, cone_collection);

  print_result(cone_collection);
  view_point_cloud(pcloud);
  return 0;
}