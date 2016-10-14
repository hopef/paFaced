


#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <highgui.h>
#include "face_detection.h"

#include "pa_sfdetect.h"

using namespace seeta;
using namespace std;
using namespace cv;

#define MatIsAlign(m)			((m).step.p[1] * (m).size[1] != (m).step.p[0])

paSfDetecter::paSfDetecter(
	int minFace /* = 40 */, 
	float scoreThreshold /* = 2.f */, 
	float imagePyramidScaleFactor /* = 0.8f */, 
	cv::Size wndStep /* = cv::Size(4, 4) */){

	FaceDetection* detector = new FaceDetection();
	detector->SetMinFaceSize(minFace);
	detector->SetScoreThresh(scoreThreshold);
	detector->SetImagePyramidScaleFactor(imagePyramidScaleFactor);
	detector->SetWindowStep(wndStep.width, wndStep.height);

	this->_native = detector;
}

paSfDetecter::~paSfDetecter(){
	if (this->_native)
		delete (FaceDetection*)this->_native;

	this->_native = 0;
}

paSfDetecter::Face toFace(const FaceInfo& fi){
	paSfDetecter::Face f;
	f.bbox = cv::Rect(fi.bbox.x, fi.bbox.y, fi.bbox.width, fi.bbox.height);
	f.pitch = fi.pitch;
	f.roll = fi.roll;
	f.yaw = fi.yaw;
	f.score = fi.score;
	return f;
}

vector<paSfDetecter::Face> paSfDetecter::detect(const cv::Mat& img){
	vector<Face> result;

	if (this->_native == 0 || img.empty())
		return result;

	cv::Mat gray;
	if (img.channels() != 1)
		cvtColor(img, gray, CV_BGR2GRAY);
	else
		gray = img;

	//如果gray存在对齐，就让他没有对齐
	if (MatIsAlign(gray))
		gray = gray.clone();

	ImageData img_data;
	img_data.data = gray.data;
	img_data.width = gray.cols;
	img_data.height = gray.rows;
	img_data.num_channels = 1;

	FaceDetection* detector = (FaceDetection*)this->_native;
	vector<FaceInfo> faces = detector->Detect(img_data);
	result.resize(faces.size());

	for (int i = 0; i < faces.size(); ++i)
		result[i] = toFace(faces[i]);
	
	return result;
}

#if 0
int main(int argc, char** argv) {

	const char* img_path = "abm.jpg";
	paSfDetecter detector;

	Mat img = imread(img_path);
	vector<paSfDetecter::Face> faces = detector.detect(img);

	for (int32_t i = 0; i < faces.size(); i++) 
		cv::rectangle(img, faces[i].bbox, CV_RGB(0, 0, 255), 4, 8, 0);

	cv::imshow("Test", img);
	cv::waitKey(0);
}
#endif

#if 0

using namespace cv;
using namespace std;
extern void initFacealignmentModel();
extern vector<Point> detectKeyPoint(const Mat& img, cv::Rect face);
extern void destroyFacealigmentModel();
void main(){
	initFacealignmentModel();
	FaceDetection detector(
		"seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::VideoCapture cap(0);
	cv::Mat frame, gray;
	cap >> frame;

	while (!frame.empty()){
		cvtColor(frame, gray, CV_BGR2GRAY);
		ImageData img_data;
		img_data.data = gray.data;
		img_data.width = gray.cols;
		img_data.height = gray.rows;
		img_data.num_channels = 1;

		vector<FaceInfo> faces = detector.Detect(img_data);

		cv::Rect face_rect;
		int32_t num_face = static_cast<int32_t>(faces.size());

		for (int32_t i = 0; i < num_face; i++) {
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;

			vector<Point> ps = detectKeyPoint(gray, face_rect);
			cv::rectangle(frame, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);

			for (int j = 0; j < ps.size(); ++j)
				circle(frame, ps[j], 5, Scalar(0, 255), 2);
		}

		cv::imshow("video", frame);
		if (cv::waitKey(30) == 'q') break;
		cap >> frame;
	}

	destroyFacealigmentModel();
}
#endif