#include "Stereo_Detector.h"

#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand

#include <opencv2/highgui.hpp>

using namespace cv;


/**
 * Yellow HSV

LH 015 LS 100 LV 100
HH 030 HS 255 HV 255
 
 * Blue HSV
LH 100 LS 110 LV 000
HH 135 HS 255 HV 255
 
 * Orange HSV
LH 005 LS 050 LV 050
HH 015 LS 255 LV 255

**/
Scalar minYellow = Scalar( 15, 100, 100);
Scalar maxYellow = Scalar( 30, 255, 255);
Scalar minBlue   = Scalar(100, 110,   0);
Scalar maxBlue   = Scalar(135, 255, 255);
Scalar minOrange = Scalar(  5,  50,  50);
Scalar maxOrange = Scalar( 15, 255, 255);

// randomly sample upto "count" number of points in a list
// find the average of the 3d position and store in pt
// ofx and ofy are the top left of the bounding box
// sample list is the white pixels in the bounding box after filtering
void Stereo_Detector::avg_list_point(
    sl::Mat& point_cloud,
    sl::float4& pt, 
    std::vector<Point>& sample,
    size_t count, 
    int ofx, 
    int ofy
){
    float x, y, z;
    x = 0; y = 0; z = 0;
    int valid_count = 0;

    // if there are more sample data than requested
    // shuffle it to pick random number of them
    if (sample.size() > count){

        std::random_shuffle(sample.begin(), sample.end());
    }else{
        // if there are not enough samples, 
        // dont randomise and use all the samples
        count = sample.size();
    }

    // sum the sample distances
    for(size_t i = 0; i < count; i++){
        point_cloud.getValue(sample[i].x + ofx, sample[i].y + ofy, &pt);
        if (!std::isnan(pt.z) && !std::isnan(pt.x) && !std::isnan(pt.y)){
            x += pt.x;
            y += pt.y;
            z += pt.z;
            valid_count ++;
        }
    }

    x = x / valid_count;
    y = y / valid_count;
    z = z / valid_count;
    //std::cout << "valid counts: " << valid_count << std::endl;
    pt.x = x; pt.y = y; pt.z = z;
}

/** SIFT MATCHING REMOVED
void Stereo_Detector::sift_triangulate(
    cv::Mat imgl, cv::Mat imgr,
    cone&   c,
    Point   lxy,  Point   rxy,
    int     h
){
    //convert image to gray scale
    cvtColor(imgl, imgl, COLOR_BGRA2GRAY);
    cvtColor(imgr, imgr, COLOR_BGRA2GRAY);

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;

    sift_detector->detectAndCompute( imgl, noArray(), keypoints1, descriptor1);
    sift_detector->detectAndCompute( imgr, noArray(), keypoints2, descriptor2);

    if (keypoints1.size() < 2) return;
    if (keypoints2.size() < 2) return;
    std::vector<std::vector<DMatch>> knn_matches;
    
    matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2);

    if (knn_matches.size() == 0) return;

    // Filter with Lowe's ratio test
    std::vector<DMatch> good_matches;
    
    
    float xl, xr, dx;
    float z_axis, x_axis;
    float sum_xaxis, sum_zaxis;
    int dxl, dxr;
    dxl = lxy.x - h;
    dxr = rxy.x - h;

    c.cone_trig_dist = 0;

    // for each matching feature, calculate the approximate distance
    // and sum to the con_trig_dist
    for(size_t i = 0; i < knn_matches.size(); i++){
        if(knn_matches[i][0].distance < \
        ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);    

            xl = keypoints1[knn_matches[i][0].queryIdx].pt.x;
            xr = keypoints2[knn_matches[i][0].trainIdx].pt.x;
            xl += dxl; xr +=dxr;

            xl *= pixle_size;
            xr *= pixle_size;
            xr += dist_right;

            x_axis = -(xl * dist_right)/(xr - dist_right -xl);
            
            if (x_axis == 0)
                z_axis = focal_lent * (- dist_right) / (xr - dist_right);
            else
                z_axis = abs(focal_lent * x_axis/xl);
            // adjust centre to between the cam

            // also sum the coordinates
            c.tx += x_axis;
            c.tz += z_axis;
            x_axis -= dist_right / 2;
            c.cone_trig_dist += sqrt(z_axis * z_axis + x_axis * x_axis);
        }
    }
    
    
    if (good_matches.size() > 0){
        // find the average between the accumulated distances
        int count = static_cast<int>(good_matches.size());
        c.cone_trig_dist = c.cone_trig_dist / count;
        c.tx = c.tx / count;
        c.tz = c.tz / count;

         if (show_window){
            // Draw matches
            Mat img_matches;
            drawMatches(imgl, keypoints1, imgr, keypoints2, good_matches,
                        img_matches, Scalar::all(-1), Scalar::all(-1),
                        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
       
            imshow("MATCHES", img_matches);
         }
    }    
}
**/

void Stereo_Detector::detection(
    cv::Mat& leftframe, 
    cv::Mat& rightframe,
    sl::Mat& point_cloud,
    std::vector<cone>& clist
){
    sl::float4 point3D;
    static size_t sample_size = 50;

    // Find cones in the left frame
    yolov.frame_process(leftframe, clist);
    
    // crop out the cone in the left image
    Mat leftcrop, rightcrop, crophsv;
    std::vector<Point> whitePix;
    
    // get dept information on each cone
    for (size_t i = 0; i < clist.size(); i++){
        //crop = Mat(leftframe, clist[i].cone_box);
        leftcrop = leftframe( Rect(0, 0, leftframe.cols, leftframe.rows)
                          & clist[i].cone_box);
                          
        // convert to HSV channel
        cvtColor(leftcrop, crophsv, COLOR_BGRA2RGB);
        cvtColor(crophsv , crophsv, COLOR_RGB2HSV);
        // produce black and white pic based on class

        int cclass = clist[i].cone_class;

        if      (cclass == 0) inRange( crophsv, minYellow, maxYellow, crophsv);
        else if (cclass == 1) inRange( crophsv, minBlue,   maxBlue,   crophsv);
        else if (cclass == 2 || cclass == 3) inRange( crophsv, minOrange, maxOrange, crophsv);
        else    inRange( crophsv, Scalar(14, 125, 0), Scalar(255, 255, 255), crophsv);

        // count the number of white pixles
        cv::findNonZero(crophsv, whitePix);

        // find avg of the points in this bounding box
        avg_list_point(point_cloud, point3D, whitePix, sample_size,
                       clist[i].cone_box.x, clist[i].cone_box.y);
        clist[i].cone_point_dist = \
        sqrt(point3D.x*point3D.x + point3D.y*point3D.y + point3D.z*point3D.z);
        clist[i].x = point3D.x;
        clist[i].z = point3D.z;
        clist[i].y = point3D.y; 

        if(show_window){
            imshow(sWName, crophsv); 
        }
            
        /** SIFT MATCHING REMOVED
        if(!std::isnan(clist[i].cone_point_dist)
           && clist[i].cone_point_dist < INFINITY){
            // approx this x in the other frame
            int rx; // same y
            float m = (point3D.z) / (point3D.x - dist_right);
            rx = ((focal_lent / m) / pixle_size ) + (rightframe.cols / 2); 
            // need to add half the screen width

            // define a rectangle that is 2x as wide
            Rect right_box = Rect(rx - (clist[i].cone_box.width), 
                    clist[i].cone_box.y, clist[i].cone_box.width * 2, 
                    clist[i].cone_box.height);

            rightcrop = rightframe(Rect( 0, 0, rightframe.cols, rightframe.rows) 
                                    &right_box );

            sift_triangulate(leftcrop, rightcrop, clist[i], 
                             clist[i].cone_box.tl(),
                             right_box.tl(), leftframe.cols / 2);


            // draw a rectangle that is 2x as wide
            rectangle(rightframe, right_box, Scalar(0, 255, 0), 2);
            //draw a circle in the right frame of radius 10
            circle(rightframe, Point(rx, clist[i].cone_centre.y), 10, 
                                     Scalar(0, 255, 0), 2);  
        }
        **/
    
    }


    if(show_window){
        yolov.drawDetections(leftframe, clist);
        // display image
        cv::imshow(iWName, leftframe);
        cv::imshow(oWName, rightframe);
    }
}

Stereo_Detector::Stereo_Detector(int width, int height, int window){
    // initialise detector
    yolov.~Detector();
    new(&yolov) Detector(width, height);
    
    /** SIFT MATCHING REMOVED
    // init sift detector
    sift_detector = \
    xfeatures2d::SIFT::create(minHessian);

    // init matcher
    // Match the descriptor with FLANN based matcher
    matcher = \
    DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    **/

    show_window = window;
    if(show_window){
        namedWindow(iWName, WINDOW_AUTOSIZE);
        namedWindow(oWName, WINDOW_AUTOSIZE);
        namedWindow(sWName, WINDOW_AUTOSIZE);
    }
}