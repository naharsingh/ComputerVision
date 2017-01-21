//Application Name : FaceRecognition
//Programmer : Nahar Singh

#include "stdafx.h"

#include "Classification.h"

String window_name = "Face Recognition";

Classification::Classification()
{	
	fileStorage[0].open("SVM_person1.xml", FileStorage::READ);
	fileStorage[1].open("SVM_person2.xml", FileStorage::READ);
	fileStorage[2].open("SVM_person3.xml", FileStorage::READ);

}

void Classification::LoadTraining()
{
	for (int i = 0; i < 3; i++)
	{
		Mat SVM_TrainingData;
		Mat SVM_Classes;

		svm[i] = ml::SVM::create();
		// edit: the params struct got removed,
		// we use setter/getter now:
		svm[i]->setType(ml::SVM::C_SVC);
		svm[i]->setKernel(ml::SVM::LINEAR);
		//svm->setGamma(3);

		svm[i]->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));


		fileStorage[i]["TrainingData"] >> SVM_TrainingData;
		fileStorage[i]["classes"] >> SVM_Classes;

		svm[i]->train(SVM_TrainingData, ml::ROW_SAMPLE, SVM_Classes);
		//cout << "train svm" << endl;
	}
}
string Classification::Classify(Mat frame)
{
	Mat m, hsv, h, s, v;
	m = frame;
	resize(m, m, Size(85, 85));
	Rect rect(10, 10, m.cols - 10, m.rows - 10);
	Mat croppedRef(m, rect);

	cv::Mat cropped;
	croppedRef.copyTo(cropped);
	m = cropped;
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
		cvtColor(hsv, m, CV_HSV2BGR);
		//imshow("histeq",dst);
		// Mat abs;
		// convertScaleAbs(dst,abs);
	}
	else if (m.channels() == 1)
		equalizeHist(m, m);

	vector<float> features;
	vector<Point> locations;
	HOGDescriptor *hog = new HOGDescriptor();
	hog->compute(m, features, Size(32, 32), Size(32, 32), locations);

	Mat smpl(features);
	smpl = smpl.t();
	smpl = smpl.reshape(1, 1);
	smpl.convertTo(smpl, CV_32FC1);
	//    cout<<"sample size:"<<sample.rows<<" "<<sample.cols<<" "<<sample.channels()<<endl;
	Mat rs;
	int response = 0;

	for (int i = 0; i < 3; i++)
	{
		response = svm[i]->predict(smpl);
	}
	string className="";
	if (response == 0)
	{
		className = "person1";
	}
	if (response == 1)
	{
		className = "person2";
	}
	if (response == 2)
	{
		className = "person3";
	}
	return className;
}
/**
* Detects faces and draws an ellipse around them
*/
void Classification::DetectFaces(Mat frame) {

	std::vector<Rect> faces;
	Mat frame_gray;

	// Convert to gray scale
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	// Equalize histogram
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3,
		0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Iterate over all of the faces
	for (size_t i = 0; i < faces.size(); i++) {

		// Find center of faces
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

		Mat face = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		// Try to detect eyes, inside each face
		/*eyes_cascade.detectMultiScale(face, eyes, 1.1, 2,
		0 | CASCADE_SCALE_IMAGE, Size(30, 30));*/
		Mat croppedRef(frame, faces[i]);

		cv::Mat cropped;
		// Copy the data into new matrix
		croppedRef.copyTo(cropped);
		resize(cropped, cropped, Size(65, 65));
		string className=Classify(cropped);
		//	if (eyes.size() > 0)
		// Draw ellipse around face
		putText(frame, className, cvPoint(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 0, 250), 1, CV_AA);
		rectangle(frame, faces[i], Scalar(255, 0, 255), 1, 8);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}

	// Display frame
	imshow(window_name, frame);

}

void Classification::PlayVideo()
{
	VideoCapture cap(0); // Open default camera
	Mat frame;

	face_cascade.load("haarcascade_frontalface_alt.xml"); // load faces
	//eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml"); // load eyes

	LoadTraining();

	while (cap.read(frame)) {
		//imshow("cap", frame);
		//waitKey(30);
		DetectFaces(frame); // Call function to detect faces
		if (waitKey(30) >= 0)    // pause
			break;
	}
}




