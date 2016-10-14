/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is an example of how to use SeetaFace engine for face alignment, the
* face alignment method described in the following paper:
*
*
*   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment,
*   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
*   European Conference on Computer Vision (ECCV), 2014
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/

#include "pa_sfkey.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"
#include "common.h"
#include "face_alignment.h"
#include <vector>
#include "seeta_fa_v1.0.h"

using namespace std;
using namespace cv;
using namespace seeta;


paSfKey::paSfKey(){
	FaceAlignment* point_detector = new FaceAlignment(g_data_facealign, sizeof(g_data_facealign));
	this->_native = point_detector;
}

paSfKey::~paSfKey(){
	FaceAlignment* point_detector = (FaceAlignment*)this->_native;
	if (point_detector) delete point_detector;
	this->_native = 0;
}

std::vector<cv::Point2d> paSfKey::detect(const cv::Mat& img, const cv::Rect& face){
	vector<Point2d> pts;

	if (this->_native == 0 || img.empty()) return pts;

	Mat _img;
	if (img.channels() != 1)
		cvtColor(img, _img, CV_BGR2GRAY);
	else
		_img = img;

	seeta::FacialLandmark points[5];
	seeta::ImageData image_data;
	image_data.data = _img.data;
	image_data.width = _img.cols;
	image_data.height = _img.rows;
	image_data.num_channels = _img.channels();

	FaceInfo fi;
	fi.bbox.x = face.x;
	fi.bbox.y = face.y;
	fi.bbox.width = face.width;
	fi.bbox.height = face.height;

	FaceAlignment* point_detector = (FaceAlignment*)this->_native;
	point_detector->PointDetectLandmarks(image_data, fi, points);

	for (int i = 0; i < 5; ++i)
		pts.push_back(Point2d(points[i].x, points[i].y));

	return pts;
}


#if 0
#include <pa_sfdetect/pa_sfdetect.h>
int main(int argc, char** argv)
{
	// Initialize face detection model
	//seeta::FaceDetection detector("C:/Users/Administrator/Desktop/SeetaFaceEngine-master/FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	paSfDetecter detecter;
	paSfKey keys;

	//load image
	Mat img_grayscale = imread("abm.jpg", 0);

	IplImage *img_color = cvLoadImage("abm.jpg", 1);
	int pts_num = 5;

	// Detect faces
	std::vector<paSfDetecter::Face> faces = detecter.detect(img_grayscale);
	int32_t face_num = static_cast<int32_t>(faces.size());

	vector<Point2d> pts = keys.detect(img_grayscale, faces[0].bbox);

	// Visualize the results
	cvRectangle(img_color, cvPoint(faces[0].bbox.x, faces[0].bbox.y), cvPoint(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
	for (int i = 0; i<pts.size(); i++)
	{
		cvCircle(img_color, cvPoint(pts[i].x, pts[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
	}
	//cvSaveImage("result.jpg", img_color);
	imshow("img", Mat(img_color));
	waitKey();

	// Release memory
	cvReleaseImage(&img_color);
	return 0;
}
#endif