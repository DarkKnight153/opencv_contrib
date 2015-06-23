#include <stdint.h>
#include "evaluation.hpp"

using namespace std;
using namespace cv;
using namespace stereo;

Evaluation::Evaluation()
{
}
double Evaluation::PerformDetailedTest(Mat rezult)
{
	Mat tsukuba = imread("D:\\groundtruth.bmp", CV_8UC1);
	Mat rezultatulFinal = Mat(tsukuba.rows, tsukuba.cols, CV_8UC3);

	imshow("tsukuba", tsukuba);
	uint8_t *date , *harta;
	harta = rezult.data;
	date = tsukuba.data;
	int w, h;
	w = tsukuba.cols;
	h = tsukuba.rows;
	int eroare = 0;
	for (int i = 0; i < tsukuba.rows ; i++)
	{
		for (int j = 0; j < tsukuba.cols; j++)
		{
			if (date[i * w + j] != 0)
			if (abs(date[i * w + j] - harta[i * w + j]) > 2 * 16)
			{
				eroare += 1;
				rezultatulFinal.at<Vec3b>(i, j)[2] = 255;
				rezultatulFinal.at<Vec3b>(i, j)[0] = rezultatulFinal.at<Vec3b>(i, j)[1] = 0;
			}
			else
			{
				rezultatulFinal.at<Vec3b>(i, j)[0] = rezultatulFinal.at<Vec3b>(i, j)[1] = rezultatulFinal.at<Vec3b>(i, j)[2] = harta[i * w + j];
			}
		}
	}
	imshow("imagineTest", rezultatulFinal);
	imwrite("D:\\eval.bmp",rezultatulFinal);
	double fin = (double)((eroare * 100) * 1.0) / (w * h);
	
	return fin;
}

Evaluation::~Evaluation(void)
{
}
