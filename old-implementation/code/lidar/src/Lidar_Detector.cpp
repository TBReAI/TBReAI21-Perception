#include "Lidar_Detector.h"

// For data collection
int record_index = -1;

// Misc function
float vectorAngle(float x, float y) {
  // add minimal noise to eliminate special cases
  // Would be fine, given that there is a minimum range on LiDAR
  if (x == 0) x = 0.001;
  if (y == 0) y = 0.001;

  float ret = atanf(fabs((float)y/x));
  if ( x > 0 && y < 0) return (M_PI - ret);
  if ( x < 0 && y > 0) return -ret;
  if ( x < 0 && y < 0) return (ret - M_PI);

  return atanf((float)y/x);
}

// This function will record the data
void record_image(cv::Mat& image){
  cv::namedWindow( "Sample", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Sample", image);

  cv::namedWindow( "Gray", cv::WINDOW_AUTOSIZE );
  cv::Mat g;
  cv::cvtColor(image, g, cv::COLOR_RGB2GRAY);
  cv::imshow("Gray", g);

  waitKey(100);

  if (record_index == -1){// ask for starting index to record 
    std::cin.clear();
    std::cout << "Enter the starting index: ";
    std::cin >> record_index;
  }

  char ctype = 0;
  std::cin.clear();
  std::cin.ignore();
  std::cout << "Class of this cone: ";
  std::cin >> ctype;

  if (ctype != '9'){
    // save the image and append the label;
    // Using default locations. (Ain't no time to make it fancy)
    imwrite("res/images/img" + std::to_string(record_index) + ".png", image);
    std::ofstream outfile;
    outfile.open("res/labels.txt", std::ios_base::app);
    outfile << ctype << "\n";
    outfile.close();

    record_index++;
    std::cout << "Saved! " << record_index << std::endl;
  }
}

Lidar_Detector::Lidar_Detector (const std::string& weight,
                                int imageSw,
                                int imageSh,
                                int gridx,
                                int gridz,
                                float thresh)
: nImageSw(imageSw), nImageSh(imageSh), nGridX(gridx), nGridZ(gridz), fThresh(thresh)
{
    // Initialise the neural network
     net << conv(32, 32, 7, 1, 32, padding::same) << activation::tanh()
     << max_pool(32, 32, 32, 2) << activation::tanh()
     << conv(32, 16, 5, 16, 32, padding::same) << activation::tanh()
     << max_pool(64, 16, 16, 2) << activation::tanh()
     << conv(64, 8, 3, 8, 16, padding::same) << activation::tanh()
     << max_pool(128, 8, 8, 2) << activation::tanh()
     << conv(128, 4, 3, 4, 2, padding::same)
     << fc(8*8*16, 128) << activation::tanh()
     << fc(128, 64) << activation::tanh()
     << fc(64, 16) << activation::tanh()
     << fc(16, 3) << softmax();

    std::cout << "loading weights from: " << weight << std::endl;
    net.load(weight);

    std::cout << "NGX: " << nGridX << "\n"
              << "NGZ: " << nGridZ << "\n"
              << "IMW: " << nImageSw << "\n"
              << "IMH: " << nImageSh << "\n"
              << "FTH: " << fThresh << "\n" << std::endl;
}

// display a point cloud to window
void Lidar_Detector::view_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::visualization::CloudViewer viewer ("Cloud Viewer");
  viewer.showCloud(cloud);
  while(!viewer.wasStopped()){};
}

// down samples the point cloud and remove the back view
void Lidar_Detector::downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample){

    // Perform downsampling with voxel
  float fleaf = 0.03f; // Determine the filter size
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud(pcloud);
  vg.setLeafSize(fleaf, fleaf, fleaf);
  vg.filter(*sample);

  // filter the +z axis values
  // For some reason, the Virtual lidar produces negative depth for forward distance
  // Might have used the point the lidar the wrong way
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(sample);
  pass.setFilterFieldName("z");
  pass.setFilterLimitsNegative (true);
  pass.setFilterLimits(0.0, INT_MAX);
  pass.filter(*sample);

  std::cerr << "PointCloud after filtering: " << sample->points.size()
       << " data points \n";
}

void Lidar_Detector::ground_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pOut){

  // View pcd
  //view_point_cloud(pcloud);

  // Gather 3D bounds
  pcl::PointXYZRGB minPoint, maxPoint;
  pcl::getMinMax3D(*pcloud, minPoint, maxPoint);

  // Calc the size of a grid
  float fGridX = (maxPoint.x - minPoint.x) / nGridX;
  float fGridZ = (maxPoint.z - minPoint.z) / nGridZ;

  // Get absolute values
  fGridX = (fGridX > 0)? fGridX : (-1 * fGridX);
  fGridZ = (fGridZ > 0)? fGridZ : (-1 * fGridZ);

  pcl::PointXYZ minBox, maxBox; // the 3D point of a box
  // Assign the values for the first grid
  minBox.x = minPoint.x; 
  minBox.y = minPoint.y; 
  minBox.z = minPoint.z;
  maxBox.y = maxPoint.y; 
  maxBox.x = minPoint.x + fGridX;
  maxBox.z = minPoint.z + fGridZ;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr gCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB localMin, localMax;

  // create filter object
  pcl::CropBox<pcl::PointXYZRGB> grid;
  grid.setInputCloud(pcloud);

  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(gCloud);
  pass.setFilterFieldName("y");

  // Start from min point construct each grid
  for (int xc = 0; xc < nGridX; ++xc){
    for (int zc = 0; zc < nGridZ; ++zc){  
      // For each sector, filter the lower layer
      
      grid.setMin(minBox.getVector4fMap());
      grid.setMax(maxBox.getVector4fMap());
      grid.filter(*gCloud);

      // move the grid point along z
      minBox.z += fGridZ; 
      maxBox.z += fGridZ;

      // Skip if no points
      if (gCloud->points.size() == 0){ continue; }

      // Find minimum point
      pcl::getMinMax3D(*gCloud, localMin, localMax);
      // SKip if this box is less than thresh hold
      if (localMax.y - localMin.y < fThresh) { continue;}
      // Keep everything between lower thresh hold and 5m from zero plane
      pass.setFilterLimits(localMin.y + fThresh, 1); 
      pass.filter(*gCloud);

      //view_point_cloud(gCloud);
      *pOut += *gCloud;
    }
    // move the grid point along x
    minBox.x += fGridX;
    maxBox.x += fGridX;

    // rest z coord
    minBox.z = minPoint.z;
    maxBox.z = minPoint.z + fGridZ;
  }
  // view result
  //view_point_cloud(pOut);
}

// Find the clusters
void Lidar_Detector::euclidean_cluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr original,
                      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& collection){
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);

  // All the things GOD only knows
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.1); // with in 10 cm?
  ec.setMinClusterSize(15);
  ec.setMaxClusterSize(1000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  // create filter object
  pcl::CropBox<pcl::PointXYZRGB> grid;
  grid.setInputCloud(original);
  pcl::PointXYZRGB minBox, maxBox;

  // Make point cloud for each cluster
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
      it != cluster_indices.end(); ++it){
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin(); 
        pit != it->indices.end(); ++pit){
      cloud_cluster->points.push_back (cloud->points[*pit]);
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height= 1;
    cloud_cluster->is_dense = true;

    // Cone retrival from original
    // With the bounding box on the downsampled point cloud
    pcl::getMinMax3D(*cloud_cluster, minBox, maxBox);
    grid.setMin(minBox.getVector4fMap());
    grid.setMax(maxBox.getVector4fMap());
    grid.filter(*cloud_cluster);

    collection.push_back(cloud_cluster);
  }

}

// Given a cluster, project the point cloud to a given image
void Lidar_Detector::project_to_image(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& image){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr projection (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Calculate the plane perpendicualr to the centre of bounding box
  pcl::PointXYZRGB minBox, maxBox, centrePoint;
  pcl::getMinMax3D(*cloud, minBox, maxBox);

  // view cloud for testing purpose
  // view_point_cloud(cloud);

  // Only need x and z
  centrePoint.x = ( minBox.x + maxBox.x ) / 2;
  centrePoint.z = ( minBox.z + maxBox.z ) / 2;
  // Calc rad
  float direction = vectorAngle(centrePoint.x, centrePoint.z);

  // Create a set of planar coeff 
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = cos(direction); // x
  coefficients->values[1] = 0.0;            // y
  coefficients->values[2] = sin(direction); // z
  coefficients->values[3] = 0;

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZRGB> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(cloud);
  proj.setModelCoefficients(coefficients);
  proj.filter(*projection);
  
  // Rotate the plane such that its parallel to the XY plane
  // so all z value in the cluster is all 0.
  float rotation = direction + M_PI/2; // Some how it just works with this value.

  Eigen::Affine3f transform_plane = Eigen::Affine3f::Identity();
  transform_plane.rotate(Eigen::AngleAxisf (rotation, Eigen::Vector3f::UnitY()));
  // Execute rotation
  pcl::transformPointCloud(*projection, *projection, transform_plane);

  // Calculate the image bound on this flat cloud
  pcl::getMinMax3D(*projection, minBox, maxBox);

  // View flattened for testing purposes
  // view_point_cloud(projection);

  // Create black image of size nImageS * nImageS
  image = cv::Mat(nImageSh, nImageSw, CV_8UC3, cv::Scalar(12,12,12)); 

  float ratio_x = nImageSw / (maxBox.x - minBox.x);
  float ratio_y = nImageSh / (maxBox.y - minBox.y); 

  for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it = projection->begin();
      it != projection->end(); ++it){
      cv::circle(image, Point(fabs(it->x - minBox.x) * ratio_x,
                                 fabs(it->y - maxBox.y) * ratio_y), 2, 
                              Scalar(it->b, it->g, it->r), 2);
  } 
}

// Run the cnn on each cluster to provide classification
void Lidar_Detector::get_cone_list(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cluster_collection,
                   std::vector<cone>& cone_collection){
  cv::Mat image; // Cast each cluster to a plane and store as image
  cv::Mat_<uint8_t> gray_scale;
  pcl::PointXYZRGB minp, maxp; // for the cluster's centroid
  cone thisCone;
  vec_t pred;

  // Define iterator
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>::iterator cluster_iterator;

  // Start time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();
  
  for (cluster_iterator =  cluster_collection.begin(); 
       cluster_iterator != cluster_collection.end(); 
       cluster_iterator++)
  {
    project_to_image(*cluster_iterator, image); 
    pcl::getMinMax3D(*(*cluster_iterator), minp, maxp);
    // Convert the colour image to gray scale
    // Painful lesson, OPENCV uses BGR not RGB, and the impage created above is considered as BGR
    cv::cvtColor(image, gray_scale, cv::COLOR_BGR2GRAY);
    vec_t  vimage;
    std::transform(gray_scale.begin(), gray_scale.end(), std::back_inserter(vimage),
                  [=](uint8_t c) {return c; });
    pred = net.predict(vimage);
    thisCone.cone_class = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
    thisCone.cone_accuy = (int)(pred[thisCone.cone_class] * 100);

    // if the accuracy of the prediction is less than 70 then declare as unknown.
    if (thisCone.cone_accuy < 70) thisCone.cone_class = -1 ;

    thisCone.x = (maxp.x + minp.x) / 2; // Horizontal 
    thisCone.y = (maxp.y + minp.y) / 2; // Vertical
    thisCone.z = (maxp.z + maxp.z) / 2; // Sideways 

    cone_collection.push_back(thisCone);
  }

  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  int elasped_time = std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count();
  std::cout << "Tiny-DNN TIME = " 
            << elasped_time
            << "[ms]" << "\n"
            << "Average Infen = " << elasped_time / cone_collection.size() << "[ms]" << std::endl; 
}

// call this function with a point cloud for detection
// cone_collection pointer to where the result is stored
void Lidar_Detector::detection( pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud,
                                std::vector<cone>& cone_collection)
{
  // Start time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();

  // Check if point cloud empty
  if (pcloud->points.size() == 0) return;

  // Downsampled point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Ground removed & clustering
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr kcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Cluster List
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cluster_collection;

  // Downsampling
  downsampling(pcloud, sample);

  // Ground removal
  ground_removal(sample,  kcloud);

  // Euclidean Clustering
  // Clusters will be append into cluster_colletion
  euclidean_cluster(kcloud, pcloud, cluster_collection);

  // Classify each cluster
  get_cone_list(cluster_collection, cone_collection);
  
  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  std::cout << "DETECTION TIME = " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  //std::cout << "Cone Collecction Size: "<< cone_collection.size() << std::endl;
  std::cout << "Number of clusters:: " << cluster_collection.size() << std::endl;

  // View result
  //view_point_cloud(pcloud);
  //view_point_cloud(kcloud);
}