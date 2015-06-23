//By downloading, copying, installing or using the software you agree to this license.
//If you do not agree to this license, do not download, install,
//copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//                       (3-clause BSD License)
//
//Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
//Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
//Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
//Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
//Copyright (C) 2015, OpenCV Foundation, all rights reserved.
//Copyright (C) 2015, Itseez Inc., all rights reserved.
//Third party copyrights are property of their respective owners.
//
//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall copyright holders or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.

#include "matching.hpp"

using namespace cv;
using namespace stereo;

Matching::Matching(void)
{
}


Matching::~Matching(void)
{
}
//constructor for the matching class
//maxDisp - represents the maximum disparity
Matching::Matching(int maxDisp)
{
	this->maxDisparity = maxDisp;
}

//method for setting the maximum disparity
void Matching::setMaxDisparity(int val)
{
	this->maxDisparity = val;
}
//method for getting the disparity
int Matching::getMaxDisparity()
{
	return this->maxDisparity;
}
//method for computing the hamming difference
void Matching::computeHammingDistance(uint32_t *left, uint32_t *right, int width, int height, uint32_t * &c)
{
	int v = this->maxDisparity;
	memset(c, 0, sizeof(c[0]) * width * height * (v + 1));

#pragma omp parallel for
	for (int i = 0; i < height - 2; i++)
	{
		int iw = i * width;
		for (int j = 0; j < width - 2; j++)
		{
			int j2;
			uint32_t xorul;
			int iwj;
			iwj = iw + j;
			for (int d = 0; d <= v; d++)
			{
				j2 = MAXIM(0, j - d);
				xorul = left[iwj] ^ right[iw + j2];
				c[(iwj)* (v + 1) + d] = __popcnt(xorul);
			}
		}
	}
}
//the partial sums done on the cost volume 
void Matching::computePartialSums3(uint32_t *ham, int width, int height, int maxDisp, uint32_t * &c)
{
	memset(c, 0, sizeof(c[0]) * (width + 1) * (height + 1) * (maxDisp + 1));

#pragma omp parallel for 
	for (int i = 1; i <= height; i++)
	{
		int iw = i * width;
		int iwi = (i - 1) * width;
		for (int j = 1; j <= width; j++)
		{
			int iwj = (iw + j) * (maxDisp + 1);
			int iwjmu = (iw + j - 1) * (maxDisp + 1);
			int iwijmu = (iwi + j - 1) * (maxDisp + 1);
			for (int d = 0; d <= maxDisp; d++)
			{
				c[iwj + d] = ham[iwijmu + d] + c[iwjmu + d];

			}
		}
	}

#pragma omp parallel for
	for (int j = 1; j <= width; j++)
	{
		for (int i = 1; i <= height; i++)
		{
			int iwj = (i * width + j) * (maxDisp + 1);
			int iwjmu = ((i - 1)  * width + j) * (maxDisp + 1);
			for (int d = 0; d <= maxDisp; d++)
			{
				c[iwj + d] += c[iwjmu + d];
			}
		}
	}

}
//the aggregation on the cost volume
void Matching::computeHammingImage(uint32_t *parSum, int width, int height, int maxDisp, int windowSize, uint32_t * &c)
{
	int win = windowSize / 2;
	memset(c, 0, sizeof(c[0]) * width * height * (maxDisp + 1));
#pragma omp parallel for
	for (int i = win + 1; i <= height - win - 1; i++)
	{
		int iw = i * width;
		int iwi = (i - 1) * width;
		for (int j = win + 1; j <= width - win - 1; j++)
		{
			int w1 = ((i + win + 1) * width + j + win) * (maxDisp + 1);
			int w2 = ((i - win) * width + j - win - 1) * (maxDisp + 1);
			int w3 = ((i + win + 1) * width + j - win - 1) * (maxDisp + 1);
			int w4 = ((i - win) * width + j + win) * (maxDisp + 1);
			int w = (iwi + j - 1) * (maxDisp + 1);
			for (int d = 0; d <= maxDisp; d++)
			{
				c[w + d] = parSum[w1 + d] + parSum[w2 + d]
					- parSum[w3 + d] - parSum[w4 + d];
			}
		}
	}
}
uint32_t Matching::MinimDiagonala(uint32_t *c, int iwpj, int widthDisp, double confidenceCheck)
{
	double mini, mini2, mini3;
	mini = mini2 = mini3 = UINT64_MAX;
	double index = 0, index2 = 0, index3 = 0;
	int iw = iwpj;
	int widthDisp2;
	widthDisp2 = widthDisp;
	widthDisp -= 1;

	for (int i = 0; i <= widthDisp; i++)
	{
		if (c[(iw + i) * widthDisp2 + i] < mini)
		{
			mini3 = mini2;
			mini2 = mini;
			index3 = index2;
			index2 = index;
			mini = c[(iw + i) * widthDisp2 + i];
			index = i;
		}
		else if (c[(iw + i) * widthDisp2 + i] < mini2)
		{
			mini3 = mini2;
			index3 = index2;
			mini2 = c[(iw + i) * widthDisp2 + i];
			index2 = i;
		}
		else if (c[(iw + i) * widthDisp2 + i] < mini3)
		{
			mini3 = c[(iw + i) * widthDisp2 + i];
			index3 = i;
		}
	}

	if (mini3 / mini <= confidenceCheck)
		return index;

	return -1;
}
uint32_t Matching::Minim(uint32_t *c, int iwpj, int widthDisp, double confidenceCheck)
{
	double mini, mini2, mini3;
	mini = mini2 = mini3 = UINT64_MAX;
	double index = 0, index2 = 0, index3 = 0;
	int iw = iwpj * widthDisp;
	widthDisp -= 1;
	for (int i = 0; i <= widthDisp; i++)
		if (c[iw + i] < mini)
		{
			mini3 = mini2;
			mini2 = mini;
			index3 = index2;
			index2 = index;
			mini = c[iw + i];
			index = i;
		}
		else if (c[iw + i] < mini2)
		{
			mini3 = mini2;
			index3 = index2;
			mini2 = c[iw + i];
			index2 = i;
		}
		else if (c[iw + i] < mini3)
		{
			mini3 = c[iw + i];
			index3 = i;
		}

		if (mini3 / mini <= confidenceCheck)
			return index;

		return -1;
}
//sub pixel interpolation
double Matching::SimetricVInterpolation(uint32_t *c, int iwjp, int widthDisp, int winDisp)
{
	if (winDisp == 0 || winDisp == widthDisp - 1)
		return winDisp;
	double m2m1, m3m1, m3, m2, m1;
	m2 = c[iwjp * widthDisp + winDisp - 1];
	m3 = c[iwjp * widthDisp + winDisp + 1];
	m1 = c[iwjp * widthDisp + winDisp];
	m2m1 = m2 - m1;
	m3m1 = m3 - m1;

	if (m2m1 == 0 || m3m1 == 0) return winDisp;
	double p;
	p = 0;
	if (m2 > m3)
	{
		p = (0.5 - 0.25 * ((m3m1 * m3m1) / (m2m1 * m2m1) + (m3m1 / m2m1)));
	}
	else
	{
		p = -1 * (0.5 - 0.25 * ((m2m1 * m2m1) / (m3m1 * m3m1) + (m2m1 / m3m1)));
	}
	if (p >= -0.5 && p <= 0.5)
		p = winDisp + p;

	return p;
}
//sub pixel interpolation
double Matching::SimetricVInterpolationDiagonal(uint32_t *c, int iwjp, int widthDisp, int winDisp)
{
	if (winDisp == 0 || winDisp == widthDisp - 1)
		return winDisp;
	double m2m1, m3m1, m3, m2, m1;
	m2 = c[(iwjp + winDisp - 1) * widthDisp + winDisp - 1];
	m3 = c[(iwjp + winDisp + 1)* widthDisp + winDisp + 1];
	m1 = c[(iwjp + winDisp) * widthDisp + winDisp];
	m2m1 = m2 - m1;
	m3m1 = m3 - m1;

	if (m2m1 == 0 || m3m1 == 0) return winDisp;
	double p;
	p = 0;
	if (m2 > m3)
	{
		p = (0.5 - 0.25 * ((m3m1 * m3m1) / (m2m1 * m2m1) + (m3m1 / m2m1)));
	}
	else
	{
		p = -1 * (0.5 - 0.25 * ((m2m1 * m2m1) / (m3m1 * m3m1) + (m2m1 / m3m1)));
	}
	if (p >= -0.5 && p <= 0.5)
		p = winDisp + p;

	return p;
}

//function for generating the disparity map
void  Matching::GenerateDisparityMapImprovedDiagonalSubPixel(uint32_t *c, int width, int height, int disparity, uint8_t * &map, int th)
{
#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		int lr, rl;
		int v = -1;
		double p1, p2;
		int iw = i * width;
		for (int j = 0; j < width; j++)
		{
			//p2 = QuadraticInterpolation(c, iw + j, disparity, p);
			lr = Minim(c, iw + j, disparity + 1, 6);
			//rl = MinimDiagonala(c, iw + j, disparity + 1, 1);
			if (lr != -1)
			{
				if (width - j <= disparity)
				{
					lr = Minim(c, iw + j, disparity + 1, 2.1);
					if (lr != -1)
					{
						p2 = SimetricVInterpolation(c, iw + j, disparity + 1, lr);
						map[iw + j] = (p2) * 16;
					}
					else
						map[iw + j] = 0;
				}
				else
				{
					v = MinimDiagonala(c, iw + j - lr, disparity + 1, 6);
					if (v != -1)
					{
						p1 = SimetricVInterpolationDiagonal(c, iw + j - lr, disparity + 1, v);
						p2 = SimetricVInterpolation(c, iw + j, disparity + 1, lr);

						if (abs(p1 - p2) <= th)
							map[iw + j] = (p2) * 16;
						else
						{
							//lr = -1;
							map[iw + j] = 0;
						}
					}

					else
					{
						map[iw + j] = 0;
					}
				}
			}
			else
				map[iw + j] = 0;
		}
	}

}
//a better version of median filtering
void Matching::MedianFilter(uint8_t *harta, int height, int width, uint8_t * &map)
{
	//uint8_t *map = new uint8_t[width * height];
#pragma omp parallel for
	for (int m = 1; m < height - 1; ++m)
		for (int n = 1; n < width - 1; ++n)
		{
			int k = 0;
			uint8_t window[9];
			for (int j = m - 1; j < m + 2; ++j)
				for (int i = n - 1; i < n + 2; ++i)
					window[k++] = harta[j * width + i];
			for (int j = 0; j < 5; ++j)
			{
				int min = j;
				for (int l = j + 1; l < 9; ++l)
					if (window[l] < window[min])
						min = l;
				//   Put found minimum element in its place
				const uint8_t temp = window[j];
				window[j] = window[min];
				window[min] = temp;
			}
			//   Get result - the middle element
			map[m  * width + n] = window[4];
		}
}
//computation of the partial sums done on the intensity image
void Matching::computePartialSumsIntensityImage(uint8_t *img, uint8_t *img2, int width, int height, uint32_t * &cL, uint32_t * &cR)
{

}
//integral image computation used in the Mean Variation Census Transform
void Matching::computeIntegralImage(uint32_t *img, uint32_t *img2, int width, int height, int windowSize, uint32_t * &c, uint32_t * &c2)
{

}


