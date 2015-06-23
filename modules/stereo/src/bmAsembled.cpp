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
*   The interface contains the main methods which will be called by the stereo bm class								      *
*																												  *
\******************************************************************************************************************/

#include "bmAsembled.hpp"

using namespace cv;
using namespace stereo;

BmAsembled::BmAsembled(void)
{
}

void BmAsembled::computeStereoLBP(uint8_t *img1, uint8_t *img2, uint8_t * &map)
{
	//better for 5x5
	//descript->applyCensusOnImageWithPadding(img1, img2, newHeightTrimmed, height, width, padding_top, padding_bottom, cl, cr);
	//better
	//descript->applySparseCensusOnImageWithPadding(img1, img2, newHeightTrimmed, height, width, padding_top, padding_bottom, cl, cr);
	//better
	//descript->applyCenterSimetricCensus(img1, img2, newHeightTrimmed, height, width, padding_top, padding_bottom, cl, cr);
	//better - the best so far
	descript->applyMCT(img1, img2, newHeightTrimmed, height, width, padding_top, padding_bottom,0, cl, cr);
	//not good
	//descript->applyModifiedCenterSimetricCensus(img1, img2, newHeightTrimmed, height, width, padding_top, padding_bottom, cl, cr);
	
	stereoMatching->computeHammingDistance(cl, cr, width, newHeightTrimmed, hamLR);
	stereoMatching->computePartialSums3(hamLR, width, newHeightTrimmed, disparity, parSumsLR);
	stereoMatching->computeHammingImage(parSumsLR, width, newHeightTrimmed, disparity, 7, agregatedHammingLR);
	stereoMatching->GenerateDisparityMapImprovedDiagonalSubPixel(agregatedHammingLR, width, newHeightTrimmed, disparity, map, 3);
	stereoMatching->MedianFilter(map, newHeightTrimmed, width, map);
}
void BmAsembled::initialize(int window_size, int set_disparity, int width, int height, int padding_top, int padding_bottom)
{
	this->window_size = window_size;
	this->disparity = set_disparity;
	this->width = width;
	this->height = height;
	this-> padding_top = padding_top;
	this->padding_bottom = padding_bottom;
	descript = new Descriptor(this->window_size);
	stereoMatching = new Matching(this->disparity);


	newHeightTrimmed = this->height - padding_top - padding_bottom;
	cl = new uint32_t[width * height];
	cr = new uint32_t[width * height];
	hamLR = (uint32_t *)calloc(width * height * (disparity + 1), sizeof(uint32_t));
	parSumsLR = (uint32_t *)calloc((width + 1) * (newHeightTrimmed + 1) * (disparity + 1), sizeof(uint32_t));
	agregatedHammingLR = (uint32_t *)calloc((width + 1) * (newHeightTrimmed + 1) * (disparity + 1), sizeof(uint32_t));
	parSumIntensityImageL = (uint32_t *)calloc((width + 1) * (height + 1), sizeof(uint32_t));
	parSumIntensityImageR = (uint32_t *)calloc((width + 1) * (height + 1), sizeof(uint32_t));
	IntegralImageL = (uint32_t *)calloc((width + 1) * (height + 1), sizeof(uint32_t));
	IntegralImageR = (uint32_t *)calloc((width + 1) * (height + 1), sizeof(uint32_t));

	
}

BmAsembled::~BmAsembled(void)
{
}
