#include "precomp.hpp"
#include <iostream>
#include "descriptor.hpp"
#include "bmAsembled.hpp"
#include "matching.hpp"
#include "bmAsembled.hpp"
#include "opencv2/stereo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "evaluation.hpp"

using namespace cv;
using namespace stereo;
using namespace std;

int main(int, char*)
{
	cout << " Running Main function \n";

	Mat image1, image2;
	uint8_t *map;
	//image1 = imread("D:\\rezult0l.bmp", CV_8UC1);
	//image2 = imread("D:\\rezult0.bmp", CV_8UC1);
	image1 = imread("D:\\imL2l.bmp", CV_8UC1);
	image2 = imread("D:\\imL2.bmp", CV_8UC1);

	int rh = image1.rows;
	int w = image1.cols;
	int pt = 0;//20;
	int pb = 0;//110;
	int h = (image1.rows - pt - pb);

	map = new uint8_t[h * w];
	Mat rezult = Mat(h, w, CV_8UC1);

	BmAsembled *computeFunctions = new BmAsembled();

	computeFunctions->initialize(9, 16, w, rh, pt, pb);
	uint8_t *img1, *img2;
	img1 = image1.data;
	img2 = image2.data;

    computeFunctions->computeStereoLBP(img1, img2, map);

	rezult.data = map;
	//medianBlur(rezult,rezult,7);

	double eval = Evaluation::PerformDetailedTest(rezult);

	cout << "\nEvaluation rezult : " << eval << "\n";
	imshow("rez ",image1);
	imshow("Disparity", rezult);
	imwrite("D:\\Rezult.bmp",rezult);
	cout << " Finished processing Stereo LBP \n";

	imshow("test", image1);
	imshow("test2", image2);

	Mat imgLeft = image1;
	Mat imgRight = image2;

	Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

	if (imgLeft.empty() || imgRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- 2. Call the constructor for StereoBM
	int ndisparities = 16;   /**< Range of disparity */
	int SADWindowSize = 9; /**< Size of the block window. Must be odd */

	Ptr<StereoBinaryBM> sbm = StereoBinaryBM::create(ndisparities, SADWindowSize);

    sbm->setPreFilterCap(31);
    sbm->setMinDisparity(0);
    sbm->setTextureThreshold(10);
    sbm->setUniquenessRatio(15);
    sbm->setSpeckleWindowSize(100);
    sbm->setSpeckleRange(32);
    sbm->setDisp12MaxDiff(1);

	//-- 3. Calculate the disparity image
	sbm->compute(imgLeft, imgRight, imgDisparity16S);

	//-- Check its extreme values
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);

	printf("Min disp: %f Max value: %f \n", minVal, maxVal);
	
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal),16);//255 / (maxVal - minVal)

	imshow("Window", imgDisparity8U);
	eval = Evaluation::PerformDetailedTest(imgDisparity8U);

	cout << "\nEvaluation rezult Current : " << eval << "\n";
	
	waitKey(0);
	return 0;
}

