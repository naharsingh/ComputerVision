//Application Name : FaceRecognition
//Programmer : Nahar Singh


#include "stdafx.h"
#include "CropFaces.h"

void CropFaces::DetectAndCropFaces(Mat frame, string output) {

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

		string fileName = output + "\\face_" + to_string(faces[i].x) + ".jpg";
		resize(cropped, cropped, Size(65, 65));
		imwrite(fileName, cropped);
		rectangle(frame, faces[i], Scalar(255, 0, 255), 1, 8);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
	// Display frame
	imshow("DetectAndSave", frame);
}


void CropFaces::PlayVideoForCropFaces(string CropFaces) {

	VideoCapture cap(0); // Open default camera
	Mat frame;
	face_cascade.load("haarcascade_frontalface_alt.xml"); // load faces

	while (cap.read(frame)) {
		DetectAndCropFaces(frame, CropFaces); // Call function to detect faces
		if (waitKey(30) >= 0)    // pause
			break;
	}
}
