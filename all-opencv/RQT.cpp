#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void getcontours(Mat imgdil, Mat img)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgdil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	for(int i = 0; i<contours.size(); i++)
	{
		float peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], conPoly[i], 0.02*peri, true);
		//drawContours(img, conPoly, i, Scalar(255,0,255), 2);
		boundRect[i] = boundingRect(conPoly[i]);
		rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2);
		putText(img, "Obstacle", { boundRect[i].x,boundRect[i].y-5}, FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,0,255), 2);
	}
}

int main()
{
	Mat original_image = imread("target.png"), roi = imread("roi.jpeg");
	Mat hsv_original, hsv_roi, mask, or_im;
	
	or_im = imread("target.png");
	
	cvtColor(original_image, hsv_original, COLOR_BGR2HSV);
	cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
	
    	int hbins = 30, sbins = 32;
  	int histSize[] = {hbins, sbins};
    	float hranges[] = { 0, 180 };
    	float sranges[] = { 0, 256 };
    	const float* ranges[] = { hranges, sranges };
    	MatND hist;
    		
    	int channels[] = {0, 1};
    	calcHist(&hsv_roi, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    	
    	calcBackProject(&hsv_original, 1, channels, hist, mask, ranges, 1, true);
    	
    	Mat blur, col;
    	
    	GaussianBlur(mask, blur, Size(3,3), 3, 0);
	
	cvtColor(blur, blur, COLOR_GRAY2BGR);
	cvtColor(blur, blur, COLOR_BGR2HSV);
	
	int hmax = 179, hmin = 0, smax = 255, smin = 0, vmax = 255, vmin = 201;
    	
    
    	Scalar lower(hmin, smin, vmin);
    	Scalar upper(hmax, smax, vmax);
    	
    	inRange(blur, lower, upper, col);
    	
    	GaussianBlur(col, col, Size(11,11), 11, 0);
	Canny(col, col, 25, 75, 3);
	
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
	dilate(col, col, kernel);
	
	getcontours(col, original_image);
    	
    	imshow("Original", or_im);
    	imshow("Detected", original_image);
    	//imshow("mask", mask);
	//imshow("blur", blur);
	//imshow("col", col);
	waitKey(0);
	
	return 0;	
}