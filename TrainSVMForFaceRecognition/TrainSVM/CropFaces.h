#pragma once
//Application Name : FaceRecognition
//Programmer : Nahar Singh

#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

class CropFaces
{
public:
	CascadeClassifier face_cascade, eyes_cascade;
	void DetectAndCropFaces(Mat, string);
	void PlayVideoForCropFaces(string);

};