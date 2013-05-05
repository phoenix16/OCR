#include "ocr.h"

OCR::OCR(int TrainClassSize, int TestClassSize, int numClasses)
    : TrainClassSize(TrainClassSize), TestClassSize(TestClassSize), numClasses(numClasses)
{
    responses = Mat::zeros(numClasses*TestClassSize, 1, CV_32FC1);
}

// Public function to parse input csvs and load the data into Mats
void OCR::read_from_csv(const char* trainFile1, const char*  trainFile2, const char*  testFile1, const char*  testFile2)
{
    DataTrain1.read_csv(trainFile1);	DataTrain2.read_csv(trainFile2);
    DataTest1.read_csv(testFile1);	DataTest2.read_csv(testFile2);
    if (!DataTrain1.get_values() || !DataTrain2.get_values() || !DataTest1.get_values() || !DataTest2.get_values())
    {
        cout << "Error: Could not open csv file\n";
        exit(0);
    }

    trainData = DataTrain1.get_values();		Mat dataTrain2 = DataTrain2.get_values();
    testData = DataTest1.get_values();			Mat dataTest2 = DataTest2.get_values();

    // Set up Train Data
    // Throw away rows in excess of TrainClassSize in each class, and combine classes to create train set
    trainData.pop_back(trainData.rows - TrainClassSize);
    dataTrain2.pop_back(dataTrain2.rows - TrainClassSize);
    trainData.push_back(dataTrain2);

    // Set up Test Data
    testData.pop_back(testData.rows - TestClassSize);
    dataTest2.pop_back(dataTest2.rows - TestClassSize);
    testData.push_back(dataTest2);

    // Parse last digit of csv filename to get train/test label, Eg: grab 0 from "test0.jpg" or "train0.jpg"
    // Assumption : Class Labels are single digit only!
    std::string t1(trainFile1), t2(trainFile2);
    classLabel1 = boost::lexical_cast<float>(t1.substr(t1.size()-5, 1));
    classLabel2 = boost::lexical_cast<float>(t2.substr(t2.size()-5, 1));

    // Set up Train Labels
    trainLabels = Mat::ones(TrainClassSize, 1, CV_32FC1) * classLabel1;
    Mat trainLabels2 = Mat::ones(TrainClassSize, 1, CV_32FC1) * classLabel2;
    trainLabels.push_back(trainLabels2);

    // Set up Test Labels
    testLabels = Mat::ones(TestClassSize, 1, CV_32FC1) * classLabel1;
    Mat testLabels2 = Mat::ones(TestClassSize, 1, CV_32FC1) * classLabel2;
    testLabels.push_back(testLabels2);
}

Mat OCR::svm_classify()
{
    // Set up SVM parameters
    CvSVMParams SVM_params;
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.kernel_type = CvSVM::LINEAR;
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    // If non-linear kernel:
    //SVM_params.degree = 0;
    //SVM_params.gamma = 1;
    //SVM_params.coef0 = 0;
    //SVM_params.C = 1;
    //SVM_params.nu = 0;
    //SVM_params.p = 0;

    // Train the SVM
    CvSVM SVM;
    SVM.train_auto(trainData, trainLabels, Mat(), Mat(), SVM_params);

    // Test the SVM
    int testSize = testData.rows;
    for (int i = 0; i < testSize; i++)
    {
        if (SVM.predict(testData.row(i), testData.cols) >= 0)
        {
            responses.at<float>(i,0) =  classLabel1;
        }
        else
        {
            responses.at<float>(i,0) = classLabel2;
        }
    }
    return responses;
}

// To view single image
//Mat digit(1, data2.cols, data2.type());
//digit = data2.row(1);
//digit = digit.reshape(1, 28);	// Each image is 28x28, 1 channel
//imshow("digit", digit);
