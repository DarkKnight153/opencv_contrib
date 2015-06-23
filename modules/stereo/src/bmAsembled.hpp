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
#pragma once
#include "precomp.hpp"
#include <stdint.h>
#include "descriptor.hpp"
#include "matching.hpp"

#ifndef _OPENCV_BM_ASSEMBLED_HPP_
#define _OPENCV_BM_ASSEMBLED_HPP_ 
#ifdef __cplusplus
namespace cv
{
	namespace stereo
	{
		class BmAsembled
		{
		private:
			int window_size, disparity, width, height, padding_top, padding_bottom, newHeightTrimmed;
			uint32_t *cl, *cr, *hamLR, *parSumsLR, *agregatedHammingLR, *parSumIntensityImageL, *parSumIntensityImageR, *IntegralImageL, *IntegralImageR;
			Descriptor *descript;
			Matching *stereoMatching;
		public:
			void computeStereoLBP(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoCS_LBP(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoMCT(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoMeanVariation(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoModifiedMCT(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoCensus5X5SSE(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoClasicRT(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void computeStereoBRIEF(uint8_t *img1, uint8_t *img2, uint8_t * &map);
			void initialize(int window_size, int set_disparity, int width, int height, int padding_top, int padding_bottom);

			BmAsembled(void);
			~BmAsembled(void);
		};
	}
}
#endif
#endif
/*End of file*/