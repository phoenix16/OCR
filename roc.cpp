#include "roc.h"

// ROC = Receiver Operating Characteristic of the Classifier

ROC::ROC(const Mat& testlabels, Mat& responses, float truelabel)
    : testlabels(testlabels), responses(responses)
{

    Mat correct, incorrect;
    compare(testlabels, responses, correct, CMP_EQ);  // correct = 255 when (testlabels = responses), 0 otherwise
    incorrect = (255 - correct);                      // incorrect = (max_value = 255) - correct
    int num_correct = countNonZero(correct);
    int num_incorrect = testlabels.rows - num_correct;

    Mat correct_locs, incorrect_locs;
    cv::findNonZero(correct, correct_locs);
    cv::findNonZero(incorrect, incorrect_locs);

    Mat trueerror, false_error;
    compare(testlabels, responses, trueerror, CMP_EQ);  // trueerror = 255 when (testlabels = responses), 0 otherwise  
    false_error = (255 - trueerror);            // false_error
    int num_true = countNonZero(trueerror);               
    int num_false = testlabels.rows - num_true;

    Mat true_locs, false_locs;
    cv::findNonZero(trueerror, true_locs);
    cv::findNonZero(false_error, false_locs);

    // Now isolate the true positives and true negatives
    TP = 0;
    for (int i = 0; i < num_correct; i++)
    {
        int x = correct_locs.at<int>(i,1); // index of true response
        if (testlabels.at<float>(x,0) == truelabel)
        {
            TP++;
        }
    }
    TN = num_correct - TP;

    // Isolate the false positives and false negatives
    FP = 0;
    for (int i = 0; i < num_incorrect; i++)
    {
        int x = incorrect_locs.at<int>(i,1); // index of false response
        if (testlabels.at<float>(x,0) != truelabel)
        {
            FP++;
        }
    }
    FN = num_incorrect - FP;
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
