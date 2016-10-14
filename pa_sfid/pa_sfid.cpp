/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developped by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
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

#include "pa_sfid.h"
#include "common.h"
#include "face_identification.h"
using namespace seeta;
using namespace std;
using namespace cv;

#define MatIsAlign(m)			((m).step.p[1] * (m).size[1] != (m).step.p[0])

bool paSfID::loadModel(const char* modelFile){
	FaceIdentification* fid = new FaceIdentification();
	if (!fid->LoadModel(modelFile)){
		delete fid;
		fid = 0;
	}

	return (this->_native = fid) != 0;
}

paSfID::paSfID(){
	this->_native = 0;
}

paSfID::~paSfID(){
	if (this->_native)
		delete (FaceIdentification*)this->_native;

	this->_native = 0;
}

std::vector<float> paSfID::extractFeature(const cv::Mat& _faceImage){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	vector<float> feat;
	Mat faceImage = _faceImage;
	if (fr == 0 || faceImage.empty() || fr->feature_size()<1)
		return feat;
	
	//如果gray存在对齐，就让他没有对齐
	if (MatIsAlign(faceImage))
		faceImage = faceImage.clone();

	feat.resize(fr->feature_size());
	ImageData src_img_data(faceImage.cols, faceImage.rows, faceImage.channels());
	src_img_data.data = faceImage.data;

	fr->ExtractFeature(src_img_data, &feat[0]);
	return feat;
}

double paSfID::similarity(const float* feat1, const float* feat2){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	if (fr == 0 || feat1 == 0 || feat2 == 0) return -1;
	return fr->CalcSimilarity((const FaceFeatures)feat1, (const FaceFeatures)feat2);
}

double paSfID::similarity(const std::vector<float>& feat1, const std::vector<float>& feat2){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	if (fr == 0 || feat1.size() != feat2.size() || feat1.size() == 0) return -1;
	return fr->CalcSimilarity((float*)&feat1[0], (float*)&feat2[0], feat1.size());
}

double paSfID::similarity2(const std::vector<float>& feat1, const std::vector<float>& feat2){
	if (feat1.size() != feat2.size() || feat1.size() == 0) return -1;
	return FaceIdentification::CalcSimilarity2((float*)&feat1[0], (float*)&feat2[0], feat1.size());
}

double paSfID::similarity2(const float* feat1, const float* feat2){
	if (feat1 == 0 || feat2 == 0) return -1;
	return FaceIdentification::CalcSimilarity2((const FaceFeatures)feat1, (const FaceFeatures)feat2);
}

std::vector<float> paSfID::extractFeature(const cv::Mat& _faceImage, const std::vector<cv::Point2d>& landmark){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	vector<float> feat;
	cv::Mat faceImage = _faceImage;
	if (fr == 0 || faceImage.empty() || fr->feature_size()<1 || landmark.size() != 5)
		return feat;

	//如果gray存在对齐，就让他没有对齐
	if (MatIsAlign(faceImage))
		faceImage = faceImage.clone();

	feat.resize(fr->feature_size());
	ImageData src_img_data(faceImage.cols, faceImage.rows, faceImage.channels());
	src_img_data.data = faceImage.data;

	fr->ExtractFeatureWithCrop(src_img_data, (seeta::FacialLandmark*)&landmark[0], &feat[0]);
	return feat;
}

cv::Mat paSfID::cropFace(const cv::Mat& _image, const std::vector<cv::Point2d>& landmark){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	Mat image = _image;
	if (fr == 0 || image.empty() || landmark.size() != 5)
		return cv::Mat();

	//如果gray存在对齐，就让他没有对齐
	if (MatIsAlign(image))
		image = image.clone();

	ImageData src_img_data(image.cols, image.rows, image.channels());
	src_img_data.data = image.data;

	cv::Mat dst(
		fr->crop_height(),
		fr->crop_width(),
		CV_8UC(image.channels()));
	ImageData dst_img_data(dst.cols, dst.rows, dst.channels());
	dst_img_data.data = dst.data;

	fr->CropFace(src_img_data, (seeta::FacialLandmark*)&landmark[0], dst_img_data);
	return dst;
}

int paSfID::lengthDesc(){
	FaceIdentification* fr = (FaceIdentification*)this->_native;
	if (fr == 0)
		return 0;

	return fr->feature_size();
}


#if 0
#include <pa_lbf\pa_lbf.h>
#include <pa_sfdetect\pa_sfdetect.h>
#include <pa_sfkey\pa_sfkey.h>
#include <highgui.h>
using namespace cv;

void main(){
	paSfID fr("seeta_fr_v1.0.bin");
	paSfDetecter detect;
	paSfKey key;

	Mat im = imread("por.png");
	vector<sfFace> fs = detect.detect(im);
	vector<Point2d> keys = key.detect(im, fs[0].bbox);
	Mat crop = fr.cropFace(im, keys);
	
	Mat im2 = imread("DSC_0462.JPG");
	fs = detect.detect(im2);
	keys = key.detect(im2, fs[0].bbox);
	Mat crop2 = fr.cropFace(im2, keys);

	vector<float> feat1 = fr.extractFeature(crop);
	vector<float> feat2 = fr.extractFeature(crop2);

	printf("sm = %f\n", fr.similarity(feat1, feat2));
	imshow("c1", crop);
	imshow("c2", crop2);
	waitKey();
}
#endif