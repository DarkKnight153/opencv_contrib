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

#include "precomp.hpp"
#include <stdint.h>
#define UNUSED( x ) ( &reinterpret_cast< const int& >( x ) )
#ifndef _OPENCV_DESCRIPTOR_HPP_
#define _OPENCV_DESCRIPTOR_HPP_ 
#ifdef __cplusplus
namespace cv
{
	namespace stereo
	{
		class Descriptor
		{
		private:
			//this is the block size
			int kernelSize;
		public:
			//SSE function for computing the census transform on 1 image
			void applyCensusOnImage_SSE(uint8_t * image, int height, int width, uint32_t * &dist);
			//Implementation for computing the Census transform on the given image
			void applyCensusOnImage(uint8_t * image, int height, int width,uint32_t * &dist);
			//Implementation of the census transform using a top and bottom padding
			void applyCensusOnImageWithPadding(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//Implementation of a census transform which is taking into account just the some pixels from the census kernel thus allowing for larger block sizes
			void applySparseCensusOnImageWithPadding(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
			void applyMCTWithMeanVariation(uint32_t *integralL, uint32_t *integralR , uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
			void applyMCT(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom,int t, uint32_t * &dist, uint32_t * &dist2);
			//The classical center symetric census
			void applyCenterSimetricCensus(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//A modified version of cs census which is comparing the a pixel with its correspondent from the after the center
			void applyModifiedCenterSimetricCensus(uint8_t * image1, uint8_t *image2, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//The brief binary descriptor
			void applyBrifeDescriptor(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);
			//The classical Rank Transform
			void applyRTDescriptor(uint8_t * image1, uint8_t *image2, int heightTrimmed, int height, int width, int padding_top, int padding_bottom, uint32_t * &dist, uint32_t * &dist2);

			Descriptor(int size);
			~Descriptor(void);
		};
	}
}
#endif
#endif
/*End of file*/