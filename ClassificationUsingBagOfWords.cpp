
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/objdetect.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;

using namespace xfeatures2d;

int main(int argc, char * const argv[]) {
	string filepath;

	// detecting keypoints
	int minHessian = 1000;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints;
//	SurfFeatureDetector detector(1000);
	//vector<KeyPoint> keypoints;
	Mat des_image;
	
	// computing descriptors
	Ptr<DescriptorExtractor > extractor = SURF::create();
	Mat descriptors;
	Mat training_descriptors(1, extractor->descriptorSize(), extractor->descriptorType());
	Mat img;

	cout << "------- build vocabulary ---------\n";

	cout << "extract descriptors.." << endl;
	int count = 0;


	std::ifstream file("outdata.csv", ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	Mat m;
	cout << "start" << endl;

	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, ';');
		getline(liness, classlabel);
		//        cout<<path<<" "<<endl;
		if (!path.empty() && !classlabel.empty()) {
			img = imread(path);
			detector->detect(img, keypoints);
			extractor->compute(img, keypoints, descriptors);

		//	Mat des = descriptors.reshape(1, 1);
			if (!descriptors.empty())
			{
				training_descriptors.push_back(descriptors);
			}
			cout << ".";
		}
	}
	cout << endl;
//	closedir(dp);

	cout << "Total descriptors: " << training_descriptors.rows << endl;

	BOWKMeansTrainer bowtrainer(150); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();

	FileStorage fs;
	fs.open("vocabulary.yml", FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	//Ptr<DescriptorMatcher> matcher(new BFMatcher(NORM_L2, true));
//	Ptr<DescriptorMatcher > matcher = FlannBasedMatcher::create(L2<float> > ()); //(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	//setup training data for classifiers
	map<int, Mat> classes_training_data; classes_training_data.clear();

	cout << "------- train SVMs ---------\n";

	Mat response_hist;
	cout << "look in train data" << endl;
	count = 0;
	char buf[255];
	ifstream ifs("training.txt");
	int total_samples = 0;
	do
	{
		ifs.getline(buf, 255);
		string line(buf);
		istringstream iss(line);
		//		cout << line << endl;
		iss >> filepath;
		string filepath1;
		iss >> filepath1;
		Rect r; char delim;
		iss >> r.x >> delim;
		iss >> r.y >> delim;
		iss >> r.width >> delim;
		iss >> r.height;
		//		cout << r.x << "," << r.y << endl;
		int class_;
		iss >> class_;

		img = imread(filepath+" "+filepath1);
		r &= Rect(0, 0, img.cols, img.rows);
		if (r.width != 0) {
			img = img(r); //crop to interesting region
		}
		cout << ".";
		//		putText(img, c_, Point(20,20), CV_FONT_HERSHEY_PLAIN, 2.0, Scalar(255), 2);
		//		imshow("pic",img);
		bowide.compute(img, keypoints, response_hist);

		if (classes_training_data.count(class_) == 0) { //not yet created...
			classes_training_data[class_].create(0, response_hist.cols, response_hist.type());
		}
		classes_training_data[class_].push_back(response_hist);
		total_samples++;
		//		waitKey(0);
	} while (!ifs.eof());
	cout << endl;


	//train 1-vs-all SVMs
	map<int, Ptr<ml::SVM>> classes_classifiers;
	
	for (map<int, Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		int class_ = (*it).first;
		cout << "training class: " << class_ << ".." << endl;

		Mat samples(0, response_hist.cols, response_hist.type());
		//Mat labels(0, 1, CV_32F);
		vector<int> labels;

		//copy class samples and label
		samples.push_back(classes_training_data[class_]);

		for (int i = 0; i < classes_training_data[class_].rows; i++)
		{
			labels.push_back(1);
		}

		//copy rest samples and label
		for (map<int, Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			int not_class_ = (*it1).first;
			if (not_class_ == class_) continue;
			samples.push_back(classes_training_data[not_class_]);

			for (int i = 0; i < classes_training_data[not_class_].rows; i++)
			{
				labels.push_back(0);
			}
		}

		Mat samples_32f; 
		samples.convertTo(samples_32f, CV_32F);

		classes_classifiers[class_] = ml::SVM::create();
		classes_classifiers[class_]->setType(ml::SVM::C_SVC);
		classes_classifiers[class_]->setKernel(ml::SVM::INTER);
		//svm->setGamma(3);

		classes_classifiers[class_]->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
						
		Mat classes;
		Mat(labels).copyTo(classes);
		classes_classifiers[class_]->train(samples_32f, ml::ROW_SAMPLE, classes);
		classes_classifiers[class_]->save(to_string(class_) + "_TrainingData.xml");
	}
	




	cout << "------- test ---------\n";

	std::ifstream file_test("testData.csv", ifstream::in);
	if (!file_test) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line_test;
	cout << "start" << endl;

	while (getline(file_test, line_test)) {

		/*// If the file is a directory (or is in some way invalid) we'll skip it 
		if (stat(filepath.c_str(), &filestat)) continue;
		if (S_ISDIR(filestat.st_mode))         continue;
		if (dirp->d_name[0] == '.')					continue; //hidden file!*/

		cout << "eval file " << line_test << endl;

		Mat img = imread(line_test);
		bowide.compute(img, keypoints, response_hist);

		//test vs. SVMs
		for (map<int, Ptr<ml::SVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second->predict(response_hist);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}

		//		cout << ".";
	}
	cout << endl;

	cout << "done" << endl;
	return 0;
}


