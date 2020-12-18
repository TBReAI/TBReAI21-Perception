// ============================================================================
// Definition of the Cone class
// Contains the following informations
//      -   cone class. (int) e.g 0 (Yellow)
//      -   accuracy of detection (int)
//      -   bounding box (Rect)
// ============================================================================
#ifndef _CONE_H_
#define _CONE_H_

#include <opencv2/imgproc.hpp>

class cone{
    public:
        int cone_class;
        int cone_accuy;
        // this is the bounding box
        cv::Point cone_centre;
        cv::Rect cone_box;
        float cone_point_dist;
        float x, y, z;

        // SIFT Removed
        //float cone_trig_dist;
        //float tx, tz;
};

#endif  // _CONE_H_
// ============================================================================