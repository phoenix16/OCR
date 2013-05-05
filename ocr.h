#ifndef OCR_H
#define OCR_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <boost/lexical_cast.hpp>


using namespace std;
using namespace cv;

class OCR
{
private:
    int TrainClassSize;
    int TestClassSize;
    int numClasses;
    CvMLData DataTrain1, DataTrain2, DataTest1, DataTest2;
    float classLabel1, classLabel2;
    Mat responses;

public:

    Mat trainLabels;
    Mat trainData;
    Mat testLabels;
    Mat testData;
    OCR(int TrainClassSize, int TestClassSize, int numClasses);
    Mat svm_classify();
    void read_from_csv(const char* trainFile1, const char*  trainFile2, const char*  testFile1, const char*  testFile2);
};

#endif // OCR_H