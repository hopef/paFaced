#pragma once

#include <cv.h>
#include <vector>
#include "pa_sfkey_inlib.h"

class paSfKey{
public:
	paSfKey();
	virtual ~paSfKey();

	//������δ����ľ���
	std::vector<cv::Point2d> detect(const cv::Mat& img, const cv::Rect& face);
private:
	void* _native;
};