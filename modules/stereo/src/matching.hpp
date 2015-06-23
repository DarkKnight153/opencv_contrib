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
*   The interface contains the main methods for computing the matching between the left and right images	      *
*																												  *
\******************************************************************************************************************/
#pragma once
#include "precomp.hpp"
#include <stdint.h>
#ifndef _OPENCV_MATCHING_HPP_
#define _OPENCV_MATCHING_HPP_ 
#define MAXIM(a,b)  (a > b) ? (a) : (b)
#ifdef __cplusplus
namespace cv
{
	namespace stereo
	{
		class Matching
		{
		private:
			int maxDisparity;
			//function used for getting the minimum disparity from one "collumn of the disparity map"
			uint32_t Minim(uint32_t *c, int iwpj, int widthDisp, double confidenceCheck);
			//function used for getting the minimum disparity from the diagonal - used for LR check
			uint32_t MinimDiagonala(uint32_t *c, int iwpj, int widthDisp, double confidenceCheck);
			//function for refining the disparity at sub pixel using simetric v the diagonal version
			double SimetricVInterpolationDiagonal(uint32_t *c, int iwjp, int widthDisp, int winDisp);
			//function for refining the disparity at sub pixel using simetric v the normal version
			double SimetricVInterpolation(uint32_t *c, int iwjp, int widthDisp, int winDisp);

		public:
			//method for setting the maximum disparity
			void setMaxDisparity(int val);
			//method for getting the disparity
			int getMaxDisparity();
			//method for computing the hamming difference
			void computeHammingDistance(uint32_t *left, uint32_t *right, int width, int height, uint32_t * &c);
			//the partial sums done on the cost volume 
			void computePartialSums3(uint32_t *ham, int width, int height, int maxDisp, uint32_t * &c);
			//the aggregation on the cost volume
			void computeHammingImage(uint32_t *parSum, int width, int height, int maxDisp, int windowSize, uint32_t * &c);
			//computation of the partial sums done on the intensity image
			void computePartialSumsIntensityImage(uint8_t *img, uint8_t *img2, int width, int height, uint32_t * &cL, uint32_t * &cR);
			//integral image computation used in the Mean Variation Census Transform
			void computeIntegralImage(uint32_t *img, uint32_t *img2, int width, int height, int windowSize, uint32_t * &c, uint32_t * &c2);
			//function for generating disparity maps at sub pixel level
			/* costVolume - represents the cost volume
			* width, height - represent the width and height of the iage
			*disparity - represents the maximum disparity
			*map - is the disparity map that will result
			*th - is the LR threshold
			*/
			void GenerateDisparityMapImprovedDiagonalSubPixel(uint32_t *costVlume, int width, int height, int disparity, uint8_t * &map, int th);
			//constructor for the matching class
			//maxDisp - represents the maximum disparity
			//a median filter that has proven to work a bit better especially when applied on disparity maps
			void MedianFilter(uint8_t *harta, int height, int width, uint8_t * &map);
			Matching(int maxDisp);
			Matching(void);
			~Matching(void);
		};
	}
}
#endif
#endif
/*End of file*/