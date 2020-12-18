// For training a convolutional neural network.
// kind of generic.

// Once the weights are trained here, will be used 
// In the lidar code.

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include "tiny_dnn/tiny_dnn.h"
#include <algorithm>

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

// Convert single image to vect
void convert_image(const std::string& imagefilename,
                   double scale, int w, int h,
                   std::vector<vec_t>& data){
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);

    if (img.data == nullptr) return; // Cannot open

    // Can resize, but output from the LiDAR scan is already sized properly
    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));
    vec_t d;
    
    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) {return c * scale; });

    data.push_back(d);
}

// Convert all images in directory to vec_t
void convert_images(const std::string& directory,
                    double scale, int w, int h,
                    std::vector<vec_t>& data){
    using namespace boost::filesystem;
    
    path dpath(directory);
    typedef std::vector<path> vec;
    vec v; 

    copy(directory_iterator(dpath), directory_iterator(), back_inserter(v));
    std::sort(v.begin(), v.end());

    for (vec::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it)
    {
        std::cout << "   " << *it << '\n';
        convert_image((*it).string(), scale, w, h, data);
    }

}

// Read the image label
// each label is on a new line for simplicity
void read_labels(const std::string& filename, std::vector<label_t>& label){
    // Set default delimiter as , 
    std::ifstream file(filename);
    std::string line = "";

    while (getline(file, line)){
        label.push_back(size_t((int)line[0]-48));
    }
}

// find the percentage of correct predictions given a list of data and a network
void get_percentage(std::vector<vec_t>& train_images, std::vector<label_t>& train_labels, network<sequential>& net){
    // get percentage
    size_t count = 0;
    int  confidence = 0;
    int  a = 0;
    for (size_t k = 0; k < train_labels.size(); k++){
        a = net.predict_label(train_images[k]);
        count += (train_labels[k] == a);
        std::cout << train_labels[k] << " :: " << a << std::endl;
        confidence += (int)(net.predict_max_value(train_images[k]) * 100);
        //std::cout << train_labels[k] << " : " << net.predict_label(train_images[k]) << std::endl;
    }
    std::cout << " - Percentage:" << ((float)count / train_labels.size());
    std::cout << "\n - Avg Confidence: " << (float) confidence / train_labels.size();
    std::cout << std::endl;
}

int main(int argc, char** argv){

    if (argc < 3){
        std::cout << "Add the training image path as first parameter.\n" 
                  << "Add the labels of images as the second parameter.\n"
                  << "Add the weight if you have one.\n"
                  << "Usage:  ./ConvNet <../Images> <../Labels>\n";
        return 0;
    } 

    // Create a network 
    network<sequential> net;
    // Define the layers

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
          
    // if theres a 4th argument (the saved weights)
    if (argc == 4){
        net.load(argv[3]);
        std::cout << "Loaded!" << std::endl;
    }
    // Do some assertions in case im drunk
    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 3); // Yellow, Blue, Orange.

    // Load the dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;
    // Reading images
    convert_images(argv[1], 1, 32, 32, train_images);
    // Reading labels
    read_labels(argv[2], train_labels);

    assert(train_images.size() > 0);
    std::cout << train_images.size() << std::endl;
    std::cout << train_labels.size() << std::endl;
    assert(train_images.size() == train_labels.size());

    std::cout << "Before training: \n";
    get_percentage(train_images, train_labels, net);

    // Training Setting
    size_t batch_size = 16;
    size_t epochs = 50;

    // Train? 
    // Declare Optimiser first

    adagrad opt;
    net.train<cross_entropy>(opt, train_images, train_labels, batch_size, epochs);

    std::cout << "After training " << epochs << " epochs, the result is: \n";
    get_percentage(train_images, train_labels, net);

    std::cout << "\nこんばんは!\n" << std::endl;

    // if given a weight file, save under the same name
    if (argc == 4){
        net.save(argv[3]);
    }else{
        // else save with a generic name
        net.save("res/net_weights");
    }
   

    return 0;
}
