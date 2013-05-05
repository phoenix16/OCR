#include "ocr.h"
#include "roc.h"

int main(int argc, const char* argv[])
{
    if (argc < 4) {
		cout << "OCR between  2 digits (for now)" << endl;
        cout << "Usage: " << argv[0] << " <train csv file1> <train csv file2> <test csv file1> <test csv file2>" << endl;
        exit(1);
    }

    int TrainClassSize = 100, TestClassSize = 50, numClasses = 2;

    OCR ocr_obj(TrainClassSize, TestClassSize, numClasses);

    ocr_obj.read_from_csv(argv[1], argv[2], argv[3], argv[4]);
    Mat responses = ocr_obj.svm_classify();

    int testSize = ocr_obj.testData.rows;
    for (int i = 0; i < testSize; i++)
    {
        cout <<  ocr_obj.testLabels.at<float>(i,0) << " predicted as " << responses.at<float>(i,0) << endl;
    }

    ROC roc_obj(ocr_obj.testLabels, responses, 1);

    cout << "Precision = " << roc_obj.precision() * 100 << "%" << endl;
    cout << "Recall = " << roc_obj.recall() * 100 << "%" << endl;
    cout << "F-Score = " << roc_obj.FScore() * 100 << "%" << endl;

    waitKey(0);
    return 0;
}
