#pragma once

#include <cv.h>
#include <vector>
#include "pa_sfdetect_inlib.h"

#define sfFace paSfDetecter::Face

class paSfDetecter{
public:
	struct Face {
		cv::Rect bbox;
		double roll;
		double pitch;
		double yaw;
		double score; /**< Larger score should mean higher confidence. */

		Face(){ memset(this, 0, sizeof(*this)); }
	};

	paSfDetecter(
		int minFace = 40, 
		float scoreThreshold = 2.f,
		float imagePyramidScaleFactor = 0.8f, 
		cv::Size wndStep = cv::Size(4, 4));

	virtual ~paSfDetecter();
	virtual std::vector<sfFace> detect(const cv::Mat& img);

private:
	void* _native;
};