#pragma once

#include <cv.h>
#include <vector>
#include "pa_sfid_inlib.h"

class paSfID{
public:
	paSfID();
	virtual ~paSfID();

	bool loadModel(const char* modelFile);
	std::vector<float> extractFeature(const cv::Mat& faceImage);
	std::vector<float> extractFeature(const cv::Mat& faceImage, const std::vector<cv::Point2d>& landmark);
	cv::Mat cropFace(const cv::Mat& image, const std::vector<cv::Point2d>& landmark);
	double similarity(const std::vector<float>& feat1, const std::vector<float>& feat2);
	double similarity(const float* feat1, const float* feat2);
	static double similarity2(const std::vector<float>& feat1, const std::vector<float>& feat2);
	static double similarity2(const float* feat1, const float* feat2);
	int lengthDesc();

private:
	void* _native;
};