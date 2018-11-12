#include "stdafx.h"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define drawCross( center, color, d )                                 \
line( display_image, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( display_image, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

using namespace cv;
using namespace std;

void on_mouse(int e, int x, int y, int d, void *ptr)
{
	Point*p = (Point*)ptr;
	p->x = x;
	p->y = y;
}

int main()
{
	// >>> Kalman Filter Initialization
	int stateSize = 4;  // [x, y, v_x, v_y]
	int measSize = 2;   // [z_x, z_y] // we will only measure mouse cursor x and y
	int contrSize = 0;  // no control input

	unsigned int F_type = CV_32F;

	// initiation of OpenCV Kalman Filter
	cv::KalmanFilter KF(stateSize, measSize, contrSize, F_type);

	// creating state vector
	cv::Mat state(stateSize, 1, F_type);  // [x, y, v_x, v_y] // column Matrix

										  // creating measurement vector
	cv::Mat meas(measSize, 1, F_type);    // [z_x, z_y] // column matrix

										  // Transition state matrix A
										  // Note: set dT at each processing step!
										  // X_k = A*X_k-1
										  // X_k = current state := x_k, y_k, v_x_k
										  // X_k-1 = previous state
										  // A =
										  // [1 0 dT 0]
										  // [0 1 0 dT]
										  // [0 0 1  0]
										  // [0 0 0  1]
										  // observe it is an identity matrix with dT inputs that we will provide later
	cv::setIdentity(KF.transitionMatrix);

	// Measurement Matrix (This is C or H matrix)
	// size of C is measSize x stateSize
	// only those values will set which we can get as measurement in a state vector
	// here out of [x, y, v_x and v_y] we can only measure x, y of the mouse cursor coordianates
	// so we set only element "0" and "5".
	// [1 0 0 0]
	// [0 1 0 0]

	// Process Noise Covariance Matrix := stateSize x stateSize
	//  [Ex 0  0    0]
	//  [0 Ey  0    0]
	//  [0 0 E_v_x  0]
	//  [0 0  0  E_v_y]

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// Measure Noise Covariance Matrix
	//cv::setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

	// <<< Kalman Filter initializationOnThread

	Mat display_image(600, 800, CV_8UC3);
	namedWindow("Mouse Kalman with Velocity");

	char ch = 0;
	double ticks = 0;
	Point mousePos;
	vector<Point> mousev, kalmanv;

	while (ch != 'q' && ch != 'Q')
	{
		state = KF.predict(); // First predict, to update the internal statePre variable
		std::cout << "State post: " << state << std::endl;

		Point predictPt(state.at<float>(0), state.at<float>(1));
		// <<< Kalman Prediction

		// >>> Get Mouse Point
		setMouseCallback("Mouse Kalman with Velocity", on_mouse, &mousePos);
		std::cout << "Mouse Position is: " << mousePos << std::endl;
		// <<< Get Mouse Point

		// >>> Passing the measured values to the measurement vector
		meas.at<float>(0) = mousePos.x;
		meas.at<float>(1) = mousePos.y;
		// <<< Passing the measured values to the measurement vector

		// >>> Kalman Update Phase
		Mat estimated = KF.correct(meas);

		Point statePt(estimated.at<float>(0), estimated.at<float>(1));
		Point measPt(meas.at<float>(0), meas.at<float>(1));

		// <<< Kalman Update Phase

		cv::imshow("Mouse Kalman with Velocity", display_image);
		display_image = Scalar::all(0);

		mousev.push_back(measPt);
		kalmanv.push_back(statePt);

		drawCross(statePt, Scalar(255, 255, 255), 5);
		drawCross(measPt, Scalar(0, 0, 255), 5);

		for (int i = 0; i < mousev.size() - 1; i++)
			line(display_image, mousev[i], mousev[i + 1], Scalar(0, 0, 255), 1);

		for (int i = 0; i < kalmanv.size() - 1; i++)
			line(display_image, kalmanv[i], kalmanv[i + 1], Scalar(255, 0, 0), 1);


		ch = cv::waitKey(10);
	}
}

