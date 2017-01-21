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
#include <direct.h>

#include "Classification.h"
#include "CropFaces.h"

using namespace std;
using namespace cv;

static void read_csv(const string& filename, Mat& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	Mat m;
	cout << "start" << endl;

	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		//        cout<<path<<" "<<endl;
		if (!path.empty() && !classlabel.empty()) {
			m = imread(path);
			//            cout<<"m size"<<m.rows<<" "<<m.cols<<" "<<m.channels()<<endl;
			resize(m, m, Size(85, 85));

			Rect rect(10, 10, m.cols - 10, m.rows - 10);
			Mat croppedRef(m, rect);

			cv::Mat cropped;
			croppedRef.copyTo(cropped);
			m = cropped;

			Mat hsv, h, s, v, me;
			if (m.channels() == 3)
			{
				hsv = Mat(m.rows, m.cols, CV_64FC3);
				h = Mat(m.rows, m.cols, CV_8UC1);
				s = Mat(m.rows, m.cols, CV_8UC1);
				v = Mat(m.rows, m.cols, CV_8UC1);

				cvtColor(m, hsv, CV_BGR2HSV);
				vector<Mat> planes;
				split(hsv, planes);

				equalizeHist(planes[2], planes[2]);

				merge(planes, hsv);
				cvtColor(hsv, me, CV_HSV2BGR);
				//imshow("histeq",dst);
				// Mat abs;
				// convertScaleAbs(dst,abs);
			}
			else if (m.channels() == 1)
				equalizeHist(m, me);

			vector<float> features;
			vector<Point> locations;
			HOGDescriptor *hog = new HOGDescriptor();
			hog->compute(me, features, Size(32, 32), Size(32, 32), locations);
			//            cout<<"hog";
			Mat m1(features);
			m1 = m1.t();
			m1 = m1.reshape(1, 1);
			images.push_back(m1);
			//            cout<<"images size"<<images.rows<<" "<<images.cols<<" "<<images.channels()<<endl;
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int TrainClassifier()
{
	string fn_csv = string("train.csv");
	// These vectors hold the images and corresponding labels.
	Mat images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
		cout << "read" << endl;
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	Mat trainingData;
	Mat classes;

	//    cout<<"images size"<<images.rows<<" "<<images.cols<<endl;
	//    images=images.reshape(1,4);
	//    cout<<"images size"<<images.rows<<" "<<images.cols<<endl;
	Mat(images).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(labels).copyTo(classes);

	FileStorage fs("SVM.xml", FileStorage::WRITE);
	//  fs << "TrainingData" << trainingData;
	Mat m1;

	fs << "TrainingData" << trainingData;
	fs << "classes" << classes;
	cout << "saved" << endl;
	fs.release();
	return 0;
}


int main() {
	
	cout << " Enter 1 - Train New Data \n Enter 2 - Crop Faces \n Enter 3 - Classify faces\n";
	int value = 0;
	cin >>value;
	if(value==1)
		TrainClassifier();

	if (value == 2)
	{
		cout << " Enter Class Name\n";
		CropFaces cropFaces;
		string className;
		cin >> className;
		_mkdir(("D:\\CropFaces\\" + className).c_str());
		cropFaces.PlayVideoForCropFaces("D:\\CropFaces\\"+className);
	}

	if (value == 3)
	{
		Classification classification;
		classification.PlayVideo();
	}	
	return 0;
}
