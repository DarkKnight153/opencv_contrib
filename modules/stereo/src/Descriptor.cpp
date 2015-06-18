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

/*****************************************************************************************************************\
*   The interface contains the main descriptors that will be implemented in the descriptor class			      *
*																												  *
\******************************************************************************************************************/


#include "Descriptor.hpp"

using namespace cv;
using namespace stereo;

Descriptor::Descriptor(int k)
{
	this->kernelSize = k;
}
//SSE function for computing the census transform on 1 image
void Descriptor::applyCensusOnImage_SSE(uint8_t * image, int height, int width, uint32_t * &dist)
{
	//TO DO
	UNUSED( image );
	UNUSED( height );
	UNUSED( width );
	UNUSED( dist );
	
}
//Implementation for computing the Census transform on the given image
void Descriptor::applyCensusOnImage(uint8_t * image, int height, int width, uint32_t * &dist)
{
	int n2 = (this->kernelSize - 1) / 2;

//#pragma omp parallel for
	for (int i = n2; i <= height - n2; i++)
	{
		int rWidth = i * width;
		for (int j = n2; j <= width - n2; j++)
		{
			uint32_t c = 0;
			for (int ii = i - n2; ii <= i + n2; ii++)
			{
				int rrWidth = ii * width;
				for (int jj = j - n2; jj <= j + n2; jj++)
				{
					if (ii != i || jj != j)
					{
						if (image[rrWidth + jj] > image[rWidth + j])
						{
							c = c + 1;
						}
						c = c * 2;
					}
				}
			}
			dist[rWidth + j] = c;
		}
	}
}
//Implementation of the census transform using a top and bottom padding
void Descriptor::applyCensusOnImageWithPadding(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	int n2 = (this->kernelSize - 1) / 2;

//#pragma omp parallel for
	for (int i = padding_top + n2; i <= height - padding_bottom - n2; i++)
	{
		int rWidth = i * width;
		int distV = (i - padding_top) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			uint32_t c = 0;
			uint32_t c2 = 0;
			for (int ii = i - n2; ii <= i + n2; ii++)
			{
				int rrWidth = ii * width;
				for (int jj = j - n2; jj <= j + n2; jj++)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + jj] > image1[rWidth + j])
						{
							c = c + 1;
						}
						c = c * 2;
					}
					if (ii != i || jj != j)
					{
						if (image2[rrWidth + jj] > image2[rWidth + j])
						{
							c2 = c2 + 1;
						}
						c2 = c2 * 2;
					}
				}
			}
			dist[distV + j] = c;
			dist2[distV + j] = c2;
		}
	}
}
//Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
void Descriptor::applyMCTWithMeanVariation(uint32_t *integralL, uint32_t *integralR, uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	//TO DO
	//marked the variables in order to avoid warnings
	UNUSED( integralL );
	UNUSED( integralR );
	UNUSED( image1 );
	UNUSED( image2 );
	UNUSED( heightTrimmed );
	UNUSED( height );
	UNUSED( width );
	UNUSED( padding_top );
	UNUSED( padding_bottom );
	UNUSED( dist );
	UNUSED( dist2 );

}
//Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
void Descriptor::applyMCT(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, int t, uint32_t * &dist, uint32_t * &dist2)
{
	int n2 = (this->kernelSize - 1) >> 1;
	memset(dist, 0, width * heightTrimmed);

//#pragma omp parallel for
	for (int i = padding_top + n2 + 2; i <= height - padding_bottom - n2 - 2; i++)//sterg + 1 pt ca in c
	{
		int rWidth = i * width;
		int distV = (i - padding_top) * width;
		for (int j = n2 + 2; j <= width - n2 - 2; j++)
		{
			uint32_t c = 0;
			uint32_t c2 = 0;
			for (int ii = i - n2; ii <= i + n2; ii += 2)
			{
				int rrWidth = ii * width;
				for (int jj = j - n2; jj <= j + n2; jj += 2)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + jj] > image1[rWidth + j] - t)
						{
							c = c << 1;
							c = c + 1;
							c = c << 1;
							c = c + 1;
						}
						else if (image1[rWidth + j] - t < image1[rrWidth + jj] && image1[rWidth + j] + t >= image1[rrWidth + jj])
						{
							c = c << 2;
							c = c + 1;
						}
						else
						{
							c = c << 2;
						}

					}
					if (ii != i || jj != j)
					{

						if (image2[rrWidth + jj]  > image2[rWidth + j] - t)
						{
							c2 = c2 << 1;
							c2 = c2 + 1;
							c2 = c2 << 1;
							c2 = c2 + 1;
						}
						else if (image2[rWidth + j] - t < image2[rrWidth + jj] && image2[rWidth + j] + t >= image2[rrWidth + jj])
						{
							c2 = c2 << 2;
							c2 = c2 + 1;
						}
						else
						{
							c2 = c2 << 2;
						}
					}
				}

			}
			for (int ii = i - n2; ii <= i + n2; ii += 4)
			{
				int rrWidth = ii * width;
				for (int jj = j - n2; jj <= j + n2; jj += 4)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + jj]  > image1[rWidth + j] - t)
						{
							c = c << 1;
							c = c + 1;
							c = c << 1;
							c = c + 1;
						}
						else if (image1[rWidth + j] - t < image1[rrWidth + jj] && image1[rWidth + j] + t >= image1[rrWidth + jj])
						{
							c = c << 2;
							c = c + 1;
						}
						else
						{
							c = c << 2;
						}

					}
					if (ii != i || jj != j)
					{

						if (image2[rrWidth + jj]  > image2[rWidth + j] - t)
						{
							c2 = c2 << 1;
							c2 = c2 + 1;
							c2 = c2 << 1;
							c2 = c2 + 1;
						}
						else if (image2[rWidth + j] - t < image2[rrWidth + jj] && image2[rWidth + j] + t >= image2[rrWidth + jj])
						{
							c2 = c2 << 2;
							c2 = c2 + 1;
						}
						else
						{
							c2 = c2 << 2;
						}
					}
				}
			}
			dist[distV + j] = c;
			dist2[distV + j] = c2;
		}
	}
}
//the modified cs census
void Descriptor::applyModifiedCenterSimetricCensus(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	int n2 = (this->kernelSize - 1) / 2;

//#pragma omp parallel for
	for (int i = padding_top + n2; i <= height - padding_bottom - n2; i++)//sterg + 1 pt ca in c
	{
		int distV = (i - padding_top) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			uint32_t c = 0;
			uint32_t c2 = 0;
			for (int ii = i - n2; ii <= i + 1; ii++)
			{
				int rrWidth = ii * width;
				int rrWidthC = (ii + n2) * width;
				for (int jj = j - n2; jj <= j + n2; jj += 2)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + jj] > image1[rrWidthC + (jj + n2)])
						{
							c = c + 1;
						}
						c = c * 2;
					}
					if (ii != i || jj != j)
					{
						if (image2[rrWidth + jj] > image2[rrWidthC + (jj + n2)])
						{
							c2 = c2 + 1;
						}
						c2 = c2 * 2;
					}
				}
			}
			dist[distV + j] = c;
			dist2[distV + j] = c2;
		}
	}
}

//The classical center symetric census
void Descriptor::applyCenterSimetricCensus(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	int n2 = (this->kernelSize - 1) / 2;

//#pragma omp parallel for
	for (int i = padding_top + n2; i <= height - padding_bottom - n2; i++)//sterg + 1 pt ca in c
	{
		int distV = (i - padding_top) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			uint32_t c = 0;
			uint32_t c2 = 0;

			for (int ii = -n2; ii < 0; ii++)
			{
				int rrWidth = (ii + i) * width;
				for (int jj = -n2; jj <= +n2; jj++)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + (jj + j)] > image1[(ii * (-1) + i) * width + (-1 * jj) + j])
						{
							c = c + 1;
						}
						c = c * 2;
					}
					if (ii != i || jj != j)
					{
						if (image2[rrWidth + (jj + j)] > image2[(ii * (-1) + i) * width + (-1 * jj) + j])
						{
							c2 = c2 + 1;
						}
						c2 = c2 * 2;
					}
				}
			}
			for (int jj = -n2; jj < 0; jj++)
			{
				if (image1[i * width + (jj + j)] > image1[i * width + (-1 * jj) + j])
				{
					c = c + 1;
				}
				c = c * 2;
				if (image2[i * width + (jj + j)] > image2[i * width + (-1 * jj) + j])
				{
					c2 = c2 + 1;
				}
				c2 = c2 * 2;
			}
			dist[distV + j] = c;
			dist2[distV + j] = c2;
		}
	}
}
//The brief binary descriptor
void Descriptor::applyBrifeDescriptor(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	//TO DO
	//marked the variables in order to avoid warnings
	UNUSED( image1 );
	UNUSED( image2 );
	UNUSED( heightTrimmed );
	UNUSED( height );
	UNUSED( width );
	UNUSED( padding_top );
	UNUSED( padding_bottom );
	UNUSED( dist );
	UNUSED( dist2 );
}
//The classical Rank Transform
void  Descriptor::applyRTDescriptor(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	//TO DO
	//marked the variables in order to avoid warnings
	UNUSED( image1 );
	UNUSED( image2 );
	UNUSED( heightTrimmed );
	UNUSED( height );
	UNUSED( width );
	UNUSED( padding_top );
	UNUSED( padding_bottom );
	UNUSED( dist );
	UNUSED( dist2 );
}
//The census descriptor that allows block sizes larger than clasical
void Descriptor::applySparseCensusOnImageWithPadding(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2)
{
	int n2 = (this->kernelSize - 1) / 2;

//#pragma omp parallel for
	for (int i = padding_top + n2; i <= height - padding_bottom - n2; i++)//sterg + 1 pt ca in c
	{
		int rWidth = i * width;
		int distV = (i - padding_top) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			uint32_t c = 0;
			uint32_t c2 = 0;
			for (int ii = i - n2; ii <= i + n2; ii += 2)
			{
				int rrWidth = ii * width;
				for (int jj = j - n2; jj <= j + n2; jj += 2)
				{
					if (ii != i || jj != j)
					{
						if (image1[rrWidth + jj] > image1[rWidth + j])
						{
							c = c + 1;
						}
						c = c * 2;
					}
					if (ii != i || jj != j)
					{
						if (image2[rrWidth + jj] > image2[rWidth + j])
						{
							c2 = c2 + 1;
						}
						c2 = c2 * 2;
					}
				}
			}
			dist[distV + j] = c;
			dist2[distV + j] = c2;
		}
	}
}


Descriptor::~Descriptor(void)
{
}
