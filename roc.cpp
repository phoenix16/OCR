#include "roc.h"

ROC::ROC(Mat testlabels, Mat responses, float truelabel)
    : testlabels(testlabels), responses(responses)
{
    Mat trueerror, false_error;
    compare(testlabels, responses, trueerror, CMP_EQ);    false_error = (255 - trueerror);
    int num_true = countNonZero(trueerror);               int num_false = testlabels.rows - num_true;

    Mat true_locs, false_locs;
    cv::findNonZero(trueerror, true_locs);
    cv::findNonZero(false_error, false_locs);
    // Now isolate the true positives and true negatives
    TP = 0;
    for (int i = 0; i < num_true; i++)
    {
        int x = true_locs.at<int>(i,1); // index of true response
        if (testlabels.at<float>(x,0) == truelabel)
        {
            TP++;
        }
    }
    TN = num_true - TP;

    // Isolate the false positives and false negatives
    FP = 0;
    for (int i = 0; i < num_false; i++)
    {
        int x = false_locs.at<int>(i,1); // index of false response
        if (testlabels.at<float>(x,0) != truelabel)
        {
            FP++;
        }
    }
    FN = num_false - FP;
}

// Public function
float ROC::precision()
{
    return (float)TP / (TP + FP);
}

// Public function
float ROC::recall()
{
    return (float)TP / (TP + FN);
}

// Public function to find F-Score = weighted harmonic mean of Precision and Recall
float ROC::FScore()
{
    return (2 * precision() * recall()) / (precision() + recall());
}




//ROC Curve : Receiver Operating Characteristic Curve :  Plot of True Positive Rate vs. False Positive Rate
//    Ideally, knee of the curve should be as close to the top-left as possible (true positives)
//void ROC::roc_curve()
//{

//}
