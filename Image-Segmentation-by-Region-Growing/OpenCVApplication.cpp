// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "Functions.h"
#include <queue>
using namespace std;

typedef struct {
	double arie;
	double xc;
	double yc;
} mylist;

Mat rgb;

wchar_t* projectPath;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier lbp_face_cascade;

CascadeClassifier fullbody_cascade;
CascadeClassifier upperbody_cascade;
CascadeClassifier lowerbody_cascade;
CascadeClassifier mcs_upperbody_cascade;


bool isInside(const Mat& img, int i, int j)
{
	return i >= 0 && j >= 0 && i < img.rows && j < img.cols;
}

void Corners(Mat src, Mat &dst) {

	dst = src.clone();
	Mat srcGs;
	cvtColor(src, srcGs, COLOR_BGR2GRAY);

	GaussianBlur(srcGs, srcGs, Size(5, 5), 0, 0);

	vector<Point2f> corners;

	goodFeaturesToTrack(srcGs, corners, 100, 0.01, 10, Mat(), 3, true, 0.04);

	for (Point2f corner : corners)
		circle(dst, corner, 3, Scalar(0, 255, 0), 2, 8, 0);

}

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	//Mat edges;
	Mat corners;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		//Canny(grayFrame,edges,40,100,3);
		Corners(frame, corners);
		imshow("source", frame);
		imshow("gray", grayFrame);
		//imshow("edges", edges);
		imshow("edges", corners);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == EVENT_LBUTTONDOWN){
		printf("Pos(x,y): %d,%d  Color(HSV): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[0],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[2]);
		
	}

	if (event == EVENT_MOUSEMOVE) {
		Mat temp= rgb.clone();

		char msg[100];
		sprintf(msg, "Pos(x,y): %d, %d  Color(HSV): %d, %d, %d",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[0],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[2]);
		putText(temp, msg, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 255, 0), 1, 8);
		imshow("My Window", temp);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		Mat hsv;
		Mat channels[3];
		cvtColor(src, hsv, COLOR_BGR2HSV);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &hsv);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void testContour()
{
	Mat src;
	Mat dst;

	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("src", src);

		// Aplicare FTJ gaussian pt. eliminare zgomote
		// http://opencvexamples.blogspot.com/2013/10/applying-gaussian-filter.html
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat dst = Mat::zeros(src.size(), src.type());

		//de testat pe imaginea cu monede: eight.bmp
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val < 200)
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("Binarizare", dst);

		// --------------------------------- Operatii morfologice ----------------------------------
		//structuring element for morpho operations
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(dst, dst, element, Point(-1, -1), 2);
		erode(dst, dst, element, Point(-1, -1), 2);

		imshow("Postprocesare", dst);

		Labeling("Contur - functii din OpenCV", dst, true);

		// --------------------------- Proprietati geometrice simple ---------------------------------

		// Wait until user press some key
		waitKey(0);
	}
}

int* histogram(Mat src)
{
	int height = src.rows, width = src.cols;
	int* h = (int*)malloc(256 * sizeof(int));

	for (int i = 0; i < 256; i++)
		h[i] = 0;


	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++)
			h[src.at<uchar>(i, j)]++;
	}

	return h;
}


void myBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);

		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		split(hsv, channels);

		// Componentele de culoare ale modelului HSV
		Mat H = channels[0] * 255 / 180;
		Mat S = channels[1];
		Mat V = channels[2];

		imshow("input rgb image", rgb);
		imshow("input hsv image", hsv); // vizualizarea matricii hsv (ca un mat cu 3 canale) nu are semnificatie vizuala utila / relevanta

		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void histogramHSV() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);

		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		split(hsv, channels);

		// Componentele de culoare ale modelului HSV
		Mat H = channels[0] * 255 / 180;
		Mat S = channels[1];
		Mat V = channels[2];

		int histH[256], histS[256], histV[256];

		for (int i = 0; i < 256; i++) {
			histH[i] = 0;
			histS[i] = 0;
			histV[i] = 0;
		}

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				uchar locationH = H.at<uchar>(i, j);
				uchar locationS = S.at<uchar>(i, j);
				uchar locationV = V.at<uchar>(i, j);
				histH[locationH]++;
				histS[locationS]++;
				histV[locationV]++;
			}
		}

		imshow("input image", rgb);

		showHistogram("H histogram", histH, 256, 256, true);
		showHistogram("S histogram", histS, 256, 256, true);
		showHistogram("V histogram", histV, 256, 256, true);

		waitKey();
	}
}

void binarizareManuala() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);

		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		split(hsv, channels);

		// Componentele de culoare ale modelului HSV
		Mat H = channels[0] * 255 / 180;

		int histH[256];

		for (int i = 0; i < 256; i++)
			histH[i] = 0;
		
		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				uchar locationH = H.at<uchar>(i, j);
				histH[locationH]++;
			}
		}

		Mat Hbin = H.clone();

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				if(H.at<uchar>(i, j) > 25)
					Hbin.at<uchar>(i, j) = 0;
				else
					Hbin.at<uchar>(i, j) = 255;
			}
		}

		imshow("input image", rgb);
		imshow("output image", Hbin);

		waitKey();
	}
}

void binarizareAutomata()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);

		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		split(hsv, channels);

		// Componentele de culoare ale modelului HSV
		Mat H = channels[0] * 255 / 180;

		int histH[256];

		for (int i = 0; i < 256; i++)
			histH[i] = 0;

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				uchar locationH = H.at<uchar>(i, j);
				histH[locationH]++;
			}
		}

		int imin = 0, imax = 0;

		for (int i = 255; i >= 0; i--) {
			if (histH[i] > 0) {
				imax = i;
				break;
			}
		}

		for (int i = 0; i < 256; i++) {
			if (histH[i] > 0) {
				imin = i;
				break;
			}
		}

		float T2 = (float)(imin + imax) / 2;
		float T1 = 0.0;
		float error = 0.1;
		while (!((T2 - T1) < error))
		{
			T1 = T2;
			float mean1 = 0, mean2 = 0;
			int N1 = 0, N2 = 0;
			for (int i = imin; i < T1; i++)
			{
				mean1 += i * histH[i];
				N1 += histH[i];
			}
			for (int i = T1 + 1; i < imax; i++)
			{
				if (T1 > 0)
				{
					mean2 += i * histH[i];
					N2 += histH[i];
				}
			}
			T2 = (float)((float)(mean1 / N1) + (float)(mean2 / N2)) / 2;
		}

		Mat Hbin = H.clone();

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				if (H.at<uchar>(i, j) > T2)
					Hbin.at<uchar>(i, j) = 0;
				else
					Hbin.at<uchar>(i, j) = 255;
			}
		}

		imshow("input image", rgb);
		imshow("output image", Hbin);

		waitKey();
	}

}

void liveInfo() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		namedWindow("My Window", 1);

		rgb = src;

		Mat hsv;
		Mat channels[3];
		cvtColor(src, hsv, COLOR_BGR2HSV);

		setMouseCallback("My Window", MyCallBackFunc, &hsv);

		waitKey(0);
	}
}

void labeling(const string& name, const Mat& src, bool output_format)
{
	// dst - matrice RGB24 pt. afisarea rezultatului
	Mat dst = Mat::zeros(src.size(), CV_8UC3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Moments m;
	if (contours.size() > 0)
	{
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			const vector<Point>& c = contours[idx];

			// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
			m = moments(c); // calcul momente imagine
			double arie = m.m00; // aria componentei conexe idx

			if (arie > 2000)
			{
				double xc = m.m10 / m.m00; // coordonata x a CM al componentei conexe idx
				double yc = m.m01 / m.m00; // coordonata y a CM al componentei conexe idx

				Scalar color(rand() & 255, rand() & 255, rand() & 255);

				// https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
				if (output_format) // desenare obiecte pline ~ etichetare
					drawContours(dst, contours, idx, color, FILLED, 8, hierarchy);
				else  //desenare contur obiecte
					drawContours(dst, contours, idx, color, 1, 8, hierarchy);

				Point center(xc, yc);
				int radius = 5;

				// afisarea unor cercuri in jurul centrelor de masa
				//circle(final, center, radius,Scalar(255,255,355), 1, 8, 0);

				// afisarea unor cruci peste centrele de masa
				DrawCross(dst, center, 9, Scalar(255, 255, 255), 1);

				// https://en.wikipedia.org/wiki/Image_moment
				//calcul axa de alungire folosind momentele centarte de ordin 2
				double mc20p = m.m20 / m.m00 - xc * xc; // double mc20p = m.mu20 / m.m00;
				double mc02p = m.m02 / m.m00 - yc * yc; // double mc02p = m.mu02 / m.m00;
				double mc11p = m.m11 / m.m00 - xc * yc; // double mc11p = m.mu11 / m.m00;
				float teta = 0.5 * atan2(2 * mc11p, mc20p - mc02p);
				float teta_deg = teta * 180 / PI;

				printf("ID=%d, arie=%.0f, xc=%0.f, yc=%0.f, teta=%.0f\n", idx, arie, xc, yc, teta_deg);

				//axa de alungire

				float slope = tan(teta);

				Point p1, p2, p3, p4;

				p1.x = (0 - yc) / slope + xc; p1.y = 0;
				p2.x = (src.rows - 1 - yc) / slope + xc; p2.y = src.rows - 1;
				p3.y = slope * (0 - xc) + yc; p3.x = 0;
				p4.y = slope * (src.cols - 1 - xc) + yc; p4.x = src.cols - 1;

				vector<Point> points;
				points.push_back(p1);
				points.push_back(p2);
				points.push_back(p3);
				points.push_back(p4);

				vector<Point> pts;

				for (int i = 0; i < 4; i++) {
					if (points.at(i).x >= 0 && points.at(i).x <= src.cols - 1 && points.at(i).y >= 0 && points.at(i).y <= src.rows - 1)
						pts.push_back(points.at(i));
				}

				line(dst, pts.at(0), pts.at(1), (255, 255, 255), 1, 8);
			}
		}
	}

	imshow(name, dst);
}

void imgSegmentation() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);

		GaussianBlur(rgb, rgb, Size(5, 5), 0, 0);

		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);

		split(hsv, channels);

		// Componentele de culoare ale modelului HSV
		Mat H = channels[0] * 255 / 180;

		Mat Hbin = H.clone();

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				if (H.at<uchar>(i, j) <= 17 + 2.5*5 && H.at<uchar>(i, j) >= 17 - 2.5 * 5)
					Hbin.at<uchar>(i, j) = 255;
				else
					Hbin.at<uchar>(i, j) = 0;
			}
		}

		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

		erode(Hbin, Hbin, element, Point(-1, -1), 2);
		dilate(Hbin, Hbin, element, Point(-1, -1), 4);
		erode(Hbin, Hbin, element, Point(-1, -1), 2);

		//imshow("input rgb image", rgb);
		//imshow("output image", Hbin);

		labeling("label", Hbin, false);

		waitKey();
	}
}

void CallBackFuncL3(int event, int x, int y, int flags, void* param){

	Mat* src = (Mat*)param;

	if (event == EVENT_LBUTTONDOWN) {

		Mat labels = Mat::zeros(src->size(), CV_16UC1);
		queue<Point> que;
		int k = 1;
		int N = 1;
		que.push(Point(x, y));

		labels.at<ushort>(y, x) = k;

		float T = 2.5 * 5;
		int sumHue = 0;

		for (int i = -2; i <= 2; i++) {
			for (int j = -2; j <= 2; j++) {
				int Nx = x + i;
				int Ny = y + j;

				if (Ny >= 0 && Ny < src->rows && Nx >= 0 && Nx < src->cols) {
					sumHue += src->at<uchar>(Ny, Nx);
				}
			}
		}

		float hueAvg = sumHue / 25.0f;

		while (!que.empty()) {
			Point oldest = que.front();
			que.pop();
			int xx = oldest.x;
			int yy = oldest.y;

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					int Nxx = xx + i;
					int Nyy = yy + j;

					if (Nxx >= 0 && Nxx < src->cols && Nyy >= 0 && Nyy < src->rows) {
						if (labels.at<ushort>(Nyy, Nxx) == 0 && abs(src->at<uchar>(Nyy, Nxx) - hueAvg) < T) {
							que.push(Point(Nxx, Nyy));
							labels.at<ushort>(Nyy, Nxx) = k;
							hueAvg = (N * hueAvg + src->at<uchar>(Nyy, Nxx)) / (N + 1);
							N++;
						}
					}
				}
			}
		}

		Mat dst = Mat::zeros(src->size(), CV_8UC1);

		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				if (labels.at<ushort>(i, j) == k)
					dst.at<uchar>(i, j) = 255;
			}
		}

		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

		erode(dst, dst, element, Point(-1, -1), 2);
		dilate(dst, dst, element, Point(-1, -1), 4);
		erode(dst, dst, element, Point(-1, -1), 2);

		imshow("output", dst);
	}
}

void regionGrowingHue() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		namedWindow("My Window", 1);

		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsv;
		Mat channels[3];
		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);
		Mat H = channels[0] * 255 / 180;

		setMouseCallback("My Window", CallBackFuncL3, &H);

		imshow("My Window", src);

		waitKey(0);
	}
}

void regionGrowingValue() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		namedWindow("My Window", 1);

		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsv;
		Mat channels[3];
		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);
		Mat V = channels[2];

		setMouseCallback("My Window", CallBackFuncL3, &V);

		imshow("My Window", src);

		waitKey(0);
	}
}

void cornerDetection() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat dst = src.clone();
		Mat srcGs;
		cvtColor(src, srcGs, COLOR_BGR2GRAY);

		GaussianBlur(srcGs, srcGs, Size(5, 5), 0, 0);

		vector<Point2f> corners;

		goodFeaturesToTrack(srcGs, corners, 100, 0.01, 10, Mat(), 3, true, 0.04);

		for (Point2f corner : corners)
			circle(dst, corner, 3, Scalar(0, 255, 0), 2, 8, 0);


		imshow("output", dst);

		waitKey(0);
	}
}

void cornerSubPix() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat dst = src.clone();
		Mat srcGs;
		cvtColor(src, srcGs, COLOR_BGR2GRAY);

		GaussianBlur(srcGs, srcGs, Size(5, 5), 0, 0);

		vector<Point2f> corners;

		goodFeaturesToTrack(srcGs, corners, 100, 0.01, 10, Mat(), 3, true, 0.04);

		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);
		TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001);

		cornerSubPix(srcGs, corners, winSize, zeroZone, criteria);

		for (Point2f corner : corners) {
			circle(dst, corner, 3, Scalar(0, 255, 0), 2, 8, 0);
			printf("%f, %f\n", corner.x, corner.y);
		}

		imshow("output", dst);

		waitKey(0);
	}
}

void cornerHarris()
{
	int thresh = 200;
	int max_thresh = 255;

	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat srcGs;
		cvtColor(src, srcGs, COLOR_BGR2GRAY);

		GaussianBlur(srcGs, srcGs, Size(5, 5), 0, 0);

		int blockSize = 2;
		int apertureSize = 3;
		double k = 0.04;
		Mat dst = Mat::zeros(src.size(), CV_32FC1);
		cornerHarris(srcGs, dst, blockSize, apertureSize, k);
		Mat dst_norm, dst_norm_scaled;
		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);
		for (int i = 0; i < dst_norm.rows; i++)
		{
			for (int j = 0; j < dst_norm.cols; j++)
			{
				if ((int)dst_norm.at<float>(i, j) > thresh)
				{
					circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
				}
			}
		}
		
		imshow("output", dst_norm_scaled);

		waitKey(0);
	}
}

void cornerHarrisNMS()
{
	int thresh = 200;
	int max_thresh = 255;

	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat srcGs;
		cvtColor(src, srcGs, COLOR_BGR2GRAY);

		GaussianBlur(srcGs, srcGs, Size(5, 5), 0, 0);

		int blockSize = 2;
		int apertureSize = 3;
		double k = 0.04;
		Mat dst = Mat::zeros(src.size(), CV_32FC1);
		cornerHarris(srcGs, dst, blockSize, apertureSize, k);
		Mat dst_norm, dst_norm_scaled;
		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);

		int neighborhood = 10;

		for (int i = neighborhood / 2; i < dst_norm.rows - neighborhood / 2; i++){
			for (int j = neighborhood / 2; j < dst_norm.cols - neighborhood / 2; j++){

				float currentValue = dst_norm.at<float>(i, j);
				if (currentValue > thresh){

					bool isMax = true;

					for (int ki = -neighborhood / 2; ki <= neighborhood / 2; ki++){
						for (int kj = -neighborhood / 2; kj <= neighborhood / 2; kj++){

							if (dst_norm.at<float>(i + ki, j + kj) > currentValue){
								isMax = false;
								break;
							}
						}
						if (!isMax)
							break;
					}

					if (isMax)
						circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
				}
			}
		}

		imshow("output", dst_norm_scaled);
		waitKey(0);
	}
}

void videoBS()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/laboratory.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat frame, gray;
	Mat backgnd;
	Mat diff; 
	Mat dst;
	char c;
	int frameNum = -1;

	const int method = 3;
	//printf("Metoda(1/2/3):");
	//scanf("%d", &method);

	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {

		cap >> frame;

		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}

		++frameNum;

		if (frameNum == 0)
			imshow("sursa", frame);

		cvtColor(frame, gray, COLOR_BGR2GRAY);
	
		GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);
		
		dst = Mat::zeros(gray.size(), gray.type());

		const int channels_gray = gray.channels();

		if (channels_gray > 1)
			return;

		if (frameNum > 0){

			absdiff(gray, backgnd, diff);

			if (method == 1) {
				backgnd = gray.clone();
			}

			if (method == 2) {
				addWeighted(gray, alpha, backgnd, 1.0 - alpha, 0, backgnd);
			}
			
			for (int i = 0; i < diff.rows; i++) {
				for (int j = 0; j < diff.cols; j++) {
					if (diff.at<uchar>(i, j) > Th) {
						dst.at<uchar>(i, j) = 255;
					}
					else if (method == 3) {
						backgnd.at<uchar>(i, j) = alpha * gray.at<uchar>(i, j) + (1.0 - alpha) * backgnd.at<uchar>(i, j);
					}
				}
			}

			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

			erode(dst, dst, element, Point(-1, -1), 1);
			dilate(dst, dst, element, Point(-1, -1), 1);

			imshow("sursa", frame);
			imshow("dest", dst);

			double t = (double)getTickCount(); // Get the current time [s] 
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		}
		else
			backgnd = gray.clone();

		c = waitKey(0); 

		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break;
		}
	}
}

void calcOpticalFlowHS(const Mat& prev, const Mat& crnt, float lambda, int n0, Mat& flow){

	Mat vx = Mat::zeros(crnt.size(), CV_32FC1);
	Mat vy = Mat::zeros(crnt.size(), CV_32FC1);
	Mat Et = Mat::zeros(crnt.size(), CV_32FC1);
	Mat Ex, Ey;

	Sobel(crnt, Ex, CV_32F, 1, 0);
	Sobel(crnt, Ey, CV_32F, 0, 1);

	Mat prev_float, crnt_float;
	prev.convertTo(prev_float, CV_32FC1);
	crnt.convertTo(crnt_float, CV_32FC1);
	Et = crnt_float - prev_float;

	for (int iter = 0; iter < n0; iter++){
		for (int i = 1; i < vx.rows - 1; i++){
			for (int j = 1; j < vx.cols - 1; j++){
				vx.at<float>(i, j) = (vx.at<float>(i - 1, j) + vx.at<float>(i + 1, j) + vx.at<float>(i, j - 1) + vx.at<float>(i, j + 1)) / 4;
				vy.at<float>(i, j) = (vy.at<float>(i - 1, j) + vy.at<float>(i + 1, j) + vy.at<float>(i, j - 1) + vy.at<float>(i, j + 1)) / 4;

				float a = lambda * (Ex.at<float>(i, j) * vx.at<float>(i, j) + Ey.at<float>(i, j) * vy.at<float>(i, j) + Et.at<float>(i, j)) / (1 + lambda * (Ex.at<float>(i, j) * Ex.at<float>(i, j) + Ey.at<float>(i, j) * Ey.at<float>(i, j)));

				vx.at<float>(i, j) = vx.at<float>(i, j) - (a * Ex.at<float>(i, j));
				vy.at<float>(i, j) = vy.at<float>(i, j) - (a * Ey.at<float>(i, j));
			}
		}
	}

	flow = convert2flow(vx, vy);

	Mat Ex_gray, Ey_gray, Et_gray, vx_gray, vy_gray;
	normalize(Ex, Ex_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Ey, Ey_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(Et, Et_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vx, vx_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	normalize(vy, vy_gray, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	imshow("Ex", Ex_gray);
	imshow("Ey", Ey_gray);
	imshow("Et", Et_gray);
	imshow("vx", vx_gray);
	imshow("vy", vy_gray);
}

void opticFlux() {

	Mat crnt; 
	Mat prev; 
	Mat flow; 

	char folderName[MAX_PATH];
	char fname[MAX_PATH];

	if (openFolderDlg(folderName) == 0)
		return;

	FileGetter fg(folderName, "bmp");

	int frameNum = -1;

	while (fg.getNextAbsFile(fname)){
		crnt = imread(fname,IMREAD_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;

		if (frameNum > 0){

			double t = (double)getTickCount();

			calcOpticalFlowHS(prev, crnt, 10.0, 8, flow);

			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

			showFlow("optic flux", prev, flow, 1, 1, true, true, false);
		}

		imshow("crnt", crnt);

		prev = crnt.clone();

		char c = waitKey(0);
		
		if (c == 27) {
			printf("ESC pressed - playback finished\n\n");
			break;
		}
	}

}

void LK() {

	Mat crnt;
	Mat prev;
	Mat flow;

	char folderName[MAX_PATH];
	char fname[MAX_PATH];

	if (openFolderDlg(folderName) == 0)
		return;

	FileGetter fg(folderName, "bmp");

	int frameNum = -1;

	while (fg.getNextAbsFile(fname)) {
		crnt = imread(fname, IMREAD_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;

		if (frameNum > 0) {

			vector<Point2f> prev_pts;
			vector<Point2f> crnt_pts;
			vector<uchar> status;
			vector<float> error;
			Size winSize = Size(21, 21);
			int maxLevel = 3;
			TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
			int flags = 0;
			double minEigThreshold = 1e-4;

			goodFeaturesToTrack(prev, prev_pts, 100, 0.01, 10, Mat(), 3, true, minEigThreshold);
			calcOpticalFlowPyrLK(prev, crnt, prev_pts, crnt_pts, status, error, winSize, maxLevel, criteria);
			showFlowSparse("Dst", prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
		}

		imshow("crnt", crnt);

		prev = crnt.clone();

		char c = waitKey(0);

		if (c == 27) {
			printf("ESC pressed - playback finished\n\n");
			break;
		}
	}

}

void denseOpticalFlow() {

	float minVel = 0.5;

	makeColorwheel();
	make_HSI2RGB_LUT();

	Mat crnt;
	Mat prev;
	Mat flow;

	char folderName[MAX_PATH];
	char fname[MAX_PATH];

	if (openFolderDlg(folderName) == 0)
		return;

	FileGetter fg(folderName, "bmp");

	int frameNum = -1;

	while (fg.getNextAbsFile(fname)) {

		crnt = imread(fname, IMREAD_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		++frameNum;

		if (frameNum > 0) {

			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, 11, 10, 6, 1.5, 0);

			showFlowDense("Flow", crnt, flow, minVel, true);
			imshow("Image", crnt);

			int hist_dir[360] = { 0 };

			for (int i = 0; i < flow.rows; i++) {
				for (int j = 0; j < flow.cols; j++) {
					Point2f f = flow.at<Point2f>(i, j);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;

					float mod = sqrt(f.x * f.x + f.y * f.y);

					if(mod >= minVel)
						hist_dir[dir_deg]++;
				}
			}

			showHistogramDir("Hist", hist_dir, 360, 200, true);

			double t = (double)getTickCount();
			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);
		}

		prev = crnt.clone();

		char c = waitKey(0);

		if (c == 27) {
			printf("ESC pressed - playback finished\n\n");
			break;
		}
	}

}

void FaceDetectandDisplay(const string& window_name, Mat frame, int minFaceSize, int minEyeSize) {

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(minFaceSize, minFaceSize));

	for (int i = 0; i < faces.size(); i++){

		rectangle(frame, faces[i], Scalar(255, 0, 255), 4, 8, 0);

		Rect eyes_rect;
		eyes_rect.x = faces[i].x;
		eyes_rect.y = faces[i].y + 0.2 * faces[i].height;
		eyes_rect.width = faces[i].width;
		eyes_rect.height = 0.35 * faces[i].height;
		Mat eyesROI = frame_gray(eyes_rect);

		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(eyesROI, eyes, 1.1, 2, 0, Size(minEyeSize, minEyeSize));

		for (int j = 0; j < eyes.size(); j++)
			rectangle(frame, Rect(eyes_rect.x + eyes[j].x, eyes_rect.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar(255, 0, 0), 4, 8, 0);

		Rect nose_rect;
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4 * faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35 * faces[i].height;
		Mat noseROI = frame_gray(nose_rect);

		std::vector<Rect> noses;
		nose_cascade.detectMultiScale(noseROI, noses, 1.1, 2, 0, Size(minEyeSize, minEyeSize));

		for (int j = 0; j < noses.size(); j++)
			rectangle(frame, Rect(nose_rect.x + noses[j].x, nose_rect.y + noses[j].y, noses[j].width, noses[j].height), Scalar(0, 255, 0), 4, 8, 0);

		Rect mouth_rect;
		mouth_rect.x = faces[i].x;
		mouth_rect.y = faces[i].y + 0.7 * faces[i].height;
		mouth_rect.width = faces[i].width;
		mouth_rect.height = 0.3 * faces[i].height;
		Mat mouthROI = frame_gray(mouth_rect);

		std::vector<Rect> mouths;
		mouth_cascade.detectMultiScale(mouthROI, mouths, 1.1, 2, 0, Size(minEyeSize, minEyeSize));

		for (int j = 0; j < mouths.size(); j++)
			rectangle(frame, Rect(mouth_rect.x + mouths[j].x, mouth_rect.y + mouths[j].y, mouths[j].width, mouths[j].height), Scalar(0, 0, 255), 4, 8, 0);

	}
	imshow(window_name, frame);
}

void FaceDetectSimple(const string& window_name, Mat frame, int minFaceSize, int minEyeSize) {

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(minFaceSize, minFaceSize));

	for (int i = 0; i < faces.size(); i++)
		rectangle(frame, faces[i], Scalar(255, 0, 255), 2, 8, 0);
	
	imshow(window_name, frame);
}

void FaceDetectLBP(const string& window_name, Mat frame, int minFaceSize, int minEyeSize) {

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	lbp_face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(minFaceSize, minFaceSize));

	for (int i = 0; i < faces.size(); i++)
		rectangle(frame, faces[i], Scalar(255, 0, 255), 4, 8, 0);

	imshow(window_name, frame);
}

void faceDetection() {
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";

	if (!face_cascade.load(face_cascade_name)){
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name)){
		printf("Error loading eyes cascades !\n");
		return;
	}
	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("Error loading mouth cascades !\n");
		return;
	}
	if (!nose_cascade.load(nose_cascade_name)) {
		printf("Error loading nose cascades !\n");
		return;
	}

	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;

		FaceDetectandDisplay("Result", dst, minFaceSize, minEyeSize);

		waitKey(0);
	}
}

void faceDetectionVideo() {

	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) {
		printf("Error loading face cascades !\n");
		return;
	}

	_wchdir(projectPath);

	VideoCapture cap("Videos/Megamind.avi");
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat dst = frame.clone();

		double t = (double)getTickCount();

		FaceDetectSimple("Video", dst, 30, 6);

		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("%.3f [ms]\n", t * 1000);

		c = waitKey(100);
		if (c == 27) {
			printf("ESC pressed - capture finished\n");
			break;
		};
	}
}

void faceDetectionLBP() {
	String lbp_face_cascade_name = "lbpcascade_frontalface.xml";

	if (!lbp_face_cascade.load(lbp_face_cascade_name)) {
		printf("Error loading face cascades !\n");
		return;
	}

	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		Mat dst = src.clone();
		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5;

		FaceDetectLBP("Result", dst, minFaceSize, minEyeSize);

		waitKey(0);
	}
}

void faceValidation() {

	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) {
		printf("Error loading face cascades !\n");
		return;
	}

	_wchdir(projectPath);

	//VideoCapture cap("Videos/test_msv1_short.avi");
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat frame;
	Mat gray;
	Mat backgnd;
	Mat diff;
	Mat dst;
	Mat temp;
	char c;
	int frameNum = -1;

	const unsigned char Th = 25;
	const double alpha = 0.05;

	for (;;) {

		cap >> frame;

		if (frame.empty()){
			printf("End of video file\n");
			break;
		}

		++frameNum;

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);

		dst = Mat::zeros(gray.size(), gray.type());

		const int channels_gray = gray.channels();

		if (channels_gray > 1)
			return;

		if (frameNum > 0) {

			double t = (double)getTickCount();

			std::vector<Rect> faces;
			Mat frame_gray;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			equalizeHist(frame_gray, frame_gray);

			face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, Size(30, 30));

			Rect faceROI = faces[0];

			absdiff(gray, backgnd, diff);

			backgnd = gray.clone();

			for (int i = 0; i < diff.rows; i++) {
				for (int j = 0; j < diff.cols; j++) {
					if (diff.at<uchar>(i, j) > Th) 
						dst.at<uchar>(i, j) = 255;				
				}
			}

			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

			erode(dst, dst, element, Point(-1, -1), 1);
			dilate(dst, dst, element, Point(-1, -1), 1);

			temp = dst(faceROI);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			Mat roi = Mat::zeros(temp.rows, temp.cols, CV_8UC3);

			vector<mylist> candidates;
			candidates.clear();

			findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			Moments m;
			if (contours.size() > 0){
				int idx = 0;
				for (; idx >= 0; idx = hierarchy[idx][0])
				{
					const vector<Point>& c = contours[idx];
					m = moments(c);
					double arie = m.m00;
					double xc = m.m10 / m.m00;
					double yc = m.m01 / m.m00;
					Scalar color(rand() & 255, rand() & 255, rand() & 255);
					drawContours(roi, contours, idx, color, FILLED, 8, hierarchy);

					mylist elem;
					elem.arie = arie;
					elem.xc = xc;
					elem.yc = yc;
					candidates.push_back(elem);
				}
			}

			Scalar color = Scalar(0, 0, 255);

			if (candidates.size() >= 2) {
				sort(candidates.begin(), candidates.end(), [](const mylist& a, const mylist& b) {return a.arie > b.arie;});

				mylist ochi1 = candidates[0];
				mylist ochi2 = candidates[1];

				double dp = std::abs(ochi1.xc - ochi2.xc);
				double ky = 0.1;
				double kx1 = 0.3, kx2 = 0.5;

				if (std::abs(ochi1.yc - ochi2.yc) < ky * faceROI.height &&
					dp > kx1 * faceROI.width && dp < kx2 * faceROI.width &&
					((ochi1.xc > faceROI.width / 2 && ochi2.xc < faceROI.width / 2) ||
						(ochi2.xc > faceROI.width / 2 && ochi1.xc < faceROI.width / 2))) {

					DrawCross(roi, Point(ochi1.xc, ochi1.yc), 20, Scalar(0, 0, 255), 1);
					DrawCross(roi, Point(ochi2.xc, ochi2.yc), 20, Scalar(255, 0, 0), 1);
					color = Scalar(0, 255, 0);
				}
			}

			rectangle(frame, faceROI, color, 2, 8, 0);

			char msg[100];
			t = ((double)getTickCount() - t) / getTickFrequency();
			sprintf(msg, "%.2f[ms]", t * 1000);
			putText(frame, msg, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.7, color, 2, 8);

			imshow("original", frame);
			imshow("dst", dst);
			imshow("roi", roi);
		}
		else
			backgnd = gray.clone();

		c = waitKey(0);

		if (c == 27) {
			printf("ESC pressed - playback finished\n");
			break;
		}
	}
}

void detectBody(const string& window_name, Mat frame) {
	float minBodyHeight = 150.0f;

	std::vector<Rect> fullBodies;
	std::vector<Rect> lowerBodies;
	std::vector<Rect> upperBodies;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	fullbody_cascade.detectMultiScale(frame_gray, fullBodies, 1.1, 2, 0, Size(minBodyHeight * 0.4f, minBodyHeight));
	lowerbody_cascade.detectMultiScale(frame_gray, lowerBodies, 1.1, 2, 0, Size(minBodyHeight * 0.4f, minBodyHeight * 0.5f));
	upperbody_cascade.detectMultiScale(frame_gray, upperBodies, 1.1, 2, 0, Size(minBodyHeight * 0.4f, minBodyHeight * 0.5f));


	for (int i = 0; i < fullBodies.size(); i++)
	{
		Point p1(fullBodies[i].x, fullBodies[i].y);
		Point p2(fullBodies[i].x + fullBodies[i].width, fullBodies[i].y + fullBodies[i].height);

		rectangle(frame, p1, p2, Scalar(255, 255, 0), 2, 8, 0);

	}

	for (int i = 0; i < lowerBodies.size(); i++)
	{
		Point p1(lowerBodies[i].x, lowerBodies[i].y);
		Point p2(lowerBodies[i].x + lowerBodies[i].width, lowerBodies[i].y + lowerBodies[i].height);

		rectangle(frame, p1, p2, Scalar(0, 255, 255), 2, 8, 0);

	}

	for (int i = 0; i < upperBodies.size(); i++)
	{
		Point p1(upperBodies[i].x, upperBodies[i].y);
		Point p2(upperBodies[i].x + upperBodies[i].width, upperBodies[i].y + upperBodies[i].height);

		rectangle(frame, p1, p2, Scalar(255, 0, 255), 2, 8, 0);

	}

	int xOffset = minBodyHeight * 0.5f;
	int yOffset = minBodyHeight * 2.5f;
	int xAvg = 0;
	int xStd = 0;
	int luyDif = 0;
	int fuyDif = 0;
	int lfyDif = 0;
	std::vector<Rect> persons;
	std::vector<float> personsCF;
	persons.clear();
	personsCF.clear();

	uchar* fproc = (uchar*)calloc(fullBodies.size(), sizeof(uchar));
	uchar* uproc = (uchar*)calloc(upperBodies.size(), sizeof(uchar));
	uchar* lproc = (uchar*)calloc(lowerBodies.size(), sizeof(uchar));

	for (int i = 0; i < fullBodies.size(); i++){
		for (int j = 0; j < upperBodies.size(); j++){
			for (int k = 0; k < lowerBodies.size(); k++){
				if (fproc[i] == 0 && uproc[j] == 0 && lproc[k] == 0) {
					xAvg = (RectCenter(fullBodies[i]).x + RectCenter(upperBodies[j]).x + RectCenter(lowerBodies[k]).x) / 3;
					xStd = (abs(RectCenter(fullBodies[i]).x - xAvg) + abs(RectCenter(upperBodies[j]).x - xAvg) + abs(RectCenter(lowerBodies[k]).x - xAvg)) / 3;
					luyDif = RectCenter(lowerBodies[k]).y - RectCenter(upperBodies[j]).y;
					fuyDif = RectCenter(fullBodies[i]).y - RectCenter(upperBodies[j]).y;
					lfyDif = RectCenter(lowerBodies[k]).y - RectCenter(fullBodies[i]).y;

					if (xStd < xOffset && luyDif > 0 && fuyDif > 0 && lfyDif > 0) {
						float areaRatio = float(RectArea(fullBodies[i])) / RectArea(upperBodies[j] | lowerBodies[k]);
						if (0.7 < areaRatio && areaRatio < 1.3) {
							persons.push_back(fullBodies[i] & (upperBodies[j] | lowerBodies[k]));
							personsCF.push_back(0.99);
							fproc[i] = uproc[j] = lproc[k] = 1;
						}
					}
				}
			}
		}
	}

	for (int j = 0; j < upperBodies.size(); j++) {
		for (int k = 0; k < lowerBodies.size(); k++) {
			if (uproc[j] == 0 && lproc[k] == 0) {
				int luyDif = RectCenter(lowerBodies[k]).y - RectCenter(upperBodies[j]).y;
				if (luyDif > 0 && luyDif < yOffset) {
					persons.push_back(lowerBodies[k] | upperBodies[j]);
					personsCF.push_back(0.66);
					uproc[j] = lproc[k] = 1;
				}
			}
		}
	}

	for (int i = 0; i < fullBodies.size(); i++) {
		for (int j = 0; j < upperBodies.size(); j++) {
			if (fproc[i] == 0 && uproc[j] == 0) {
				if (RectArea(upperBodies[j] & fullBodies[i]) > 0.5 * RectArea(upperBodies[j])) {
					persons.push_back(upperBodies[j] | fullBodies[i]);
					personsCF.push_back(0.66);
					fproc[i] = uproc[j] = 1;
				}
			}
		}
	}

	for (int i = 0; i < fullBodies.size(); i++) {
		for (int k = 0; k < lowerBodies.size(); k++) {
			if (fproc[i] == 0 && lproc[k] == 0) {
				if (RectArea(lowerBodies[k] & fullBodies[i]) > 0.5 * RectArea(lowerBodies[k])) {
					persons.push_back(lowerBodies[k] | fullBodies[i]);
					personsCF.push_back(0.66);
					fproc[i] = lproc[k] = 1;
				}
			}
		}
	}

	for (int i = 0; i < fullBodies.size(); i++) {
		if (fproc[i] == 0) {
			persons.push_back(fullBodies[i]);
			personsCF.push_back(0.33);
		}
	}
	for (int j = 0; j < upperBodies.size(); j++) {
		if (uproc[j] == 0) {
			persons.push_back(upperBodies[j]);
			personsCF.push_back(0.33);
		}
	}
	for (int k = 0; k < lowerBodies.size(); k++) {
		if (lproc[k] == 0) {
			persons.push_back(lowerBodies[k]);
			personsCF.push_back(0.33);
		}
	}

	for (int i = 0; i < persons.size(); i++) {
		rectangle(frame, persons[i], Scalar(0, 255, 0), 2);
		std::string scoreText = cv::format("%.2f", personsCF[i]);
		Point textPos(persons[i].x, persons[i].y - 10);
		putText(frame, scoreText, textPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
	}

	imshow(window_name, frame);
}

void bodyDetection() {
	String fullbody_cascade_name = "haarcascade_fullbody.xml";
	String lowerbody_cascade_name = "haarcascade_lowerbody.xml";
	String upperbody_cascade_name = "haarcascade_upperbody.xml";
	String mcs_upperbody_cascade_name = "haarcascade_mcs_upperbody.xml";

	if (!fullbody_cascade.load(fullbody_cascade_name)){
		printf("Error loading face cascades !\n");
		return;
	}

	if (!lowerbody_cascade.load(lowerbody_cascade_name)){
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!upperbody_cascade.load(upperbody_cascade_name)){
		printf("Error loading eyes cascades !\n");
		return;
	}

	if (!mcs_upperbody_cascade.load(mcs_upperbody_cascade_name)){
		printf("Error loading eyes cascades !\n");
		return;
	}

	char fname[MAX_PATH];

	while (openFileDlg(fname)){
		Mat src = imread(fname, IMREAD_COLOR);
		Mat dst1 = src.clone();
		detectBody("body", dst1);
		waitKey();
	}
}

void mergeOverlappingDetections(vector<Rect>& bodies, vector<double>& weights) {
	vector<bool> keep(bodies.size(), true);

	for (size_t i = 0; i < bodies.size(); ++i) {
		if (!keep[i]) continue;

		for (size_t j = i + 1; j < bodies.size(); ++j) {
			if (!keep[j]) continue;


			Rect intersection = bodies[i] & bodies[j];
			double intersectionArea = (double)intersection.area();


			double areaI = (double)bodies[i].area();
			double areaJ = (double)bodies[j].area();


			double minArea = min(areaI, areaJ);
			if (intersectionArea / minArea > 0.85) {

				int centerX1 = bodies[i].x + bodies[i].width / 2;
				int centerX2 = bodies[j].x + bodies[j].width / 2;

				if (abs(centerX1 - centerX2) < (bodies[i].width + bodies[j].width) / 4) {

					if (areaI > areaJ) {
						keep[j] = false;
					}
					else {
						keep[i] = false;
						break;
					}
				}
			}
		}
	}


	vector<Rect> filteredBodies;
	vector<double> filteredWeights;
	for (size_t i = 0; i < bodies.size(); ++i) {
		if (keep[i]) {
			filteredBodies.push_back(bodies[i]);
			filteredWeights.push_back(weights[i]);
		}
	}

	bodies = filteredBodies;
	weights = filteredWeights;
}


vector<int> computeHorizontalProjection(const Mat& roi) {
	vector<int> projection(roi.rows, 0);
	for (int y = 0; y < roi.rows; ++y) {
		for (int x = 0; x < roi.cols; ++x) {
			projection[y] += roi.at<uchar>(y, x);
		}
	}
	return projection;
}

int adjustBottomMargin(const Mat& roi) {

	Mat invertedRoi = 255 - roi;


	vector<int> projection = computeHorizontalProjection(invertedRoi);


	int meanProjection = 0;
	for (int value : projection) {
		meanProjection += value;
	}
	meanProjection /= projection.size();
	int threshold = meanProjection * 0.4;

	printf("\nMean Projection: %d, Threshold: %d\n", meanProjection, threshold);


	for (int y = roi.rows - 1; y >= 0; --y) {
		if (projection[y] > threshold) {
			printf("\nprojection[%d] = %d\n", y, projection[y]);
			printf("\n Y is : %d\n", y);
			return y;
		}
	}

	return roi.rows - 1;
}


void BodyDetectandDisplayHOG() {
	Scalar color_green(0, 255, 0);
	Scalar color_magenta(255, 0, 255);
	Scalar color_yellow(0, 255, 255);
	Scalar color_cyan(255, 255, 0);
	Scalar color_blue(255, 0, 0);
	Scalar color_red(0, 0, 255);

	Mat frame;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		frame = imread(fname);
		Mat cpyFrame = frame.clone();
		HOGDescriptor hog;
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

		vector<Rect> bodies;
		vector<double> weights;
		hog.detectMultiScale(frame, bodies, weights);


		mergeOverlappingDetections(bodies, weights);

		for (size_t i = 0; i < bodies.size(); ++i) {
			Mat regiune = frame(bodies[i]);

			Mat gray;
			cvtColor(regiune, gray, COLOR_BGR2GRAY);
			GaussianBlur(gray, gray, Size(5, 5), 0.8, 0.8);
			//equalizeHist(gray, gray);



			int newBottom = adjustBottomMargin(gray);
			//imshow("gray region" + to_string(i), gray);
			printf("\n\n New Bottom is at: %d where the region height is: %d\n\n", newBottom, gray.rows);


			int adjustedHeight = min(newBottom, bodies[i].height);


			Rect adjustedROI(bodies[i].x, bodies[i].y, bodies[i].width, adjustedHeight);

			adjustedROI &= Rect(0, 0, frame.cols, frame.rows);


			rectangle(cpyFrame, adjustedROI, color_cyan, 2);

			Mat newRegiune = regiune(Rect(0, 0, adjustedROI.width, adjustedROI.height)); // Ajustăm regiunea
			//imshow("Regiune ajustata " + to_string(i), newRegiune);

			rectangle(frame, bodies[i], color_cyan, 1, 8, 0);
		}

		imshow("copy", cpyFrame);
		imshow("output", frame);
		waitKey();
	}
}


void regionGrowing(Mat& regionLabels, Mat& intensityChannel, int threshold, int& regionCount) {
	for (int row = 0; row < regionLabels.rows; row++) {
		for (int col = 0; col < regionLabels.cols; col++) {
			if (regionLabels.at<ushort>(row, col) == 0) {
				regionCount++;

				queue<Point> pixelQueue;
				pixelQueue.push(Point(col, row));

				regionLabels.at<ushort>(row, col) = regionCount;
				double averageIntensity = static_cast<int>(intensityChannel.at<uchar>(row, col));
				int pixelTotal = 1;

				while (!pixelQueue.empty()) {
					Point currentPixel = pixelQueue.front();
					pixelQueue.pop();

					int currentX = currentPixel.x;
					int currentY = currentPixel.y;

					for (int offsetY = -1; offsetY <= 1; ++offsetY) {
						for (int offsetX = -1; offsetX <= 1; ++offsetX) {
							if (offsetX == 0 && offsetY == 0) continue;

							int neighborX = currentX + offsetX;
							int neighborY = currentY + offsetY;

							if (isInside(intensityChannel, neighborY, neighborX) &&
								regionLabels.at<ushort>(neighborY, neighborX) == 0) {

								double intensityDifference = static_cast<int>(intensityChannel.at<uchar>(neighborY, neighborX)) - averageIntensity;
								double distance = sqrt(intensityDifference * intensityDifference);

								if (distance < threshold) {
									pixelQueue.push(Point(neighborX, neighborY));
									regionLabels.at<ushort>(neighborY, neighborX) = regionCount;

									averageIntensity = (averageIntensity * pixelTotal + static_cast<int>(intensityChannel.at<uchar>(neighborY, neighborX))) / (pixelTotal + 1);
									pixelTotal++;
								}
							}
						}
					}
				}
			}
		}
	}
}


void regionGrowingCombined(Mat& regionLabels, Mat& channelA, Mat& channelB, Mat& channelU, Mat& channelV, int T, int Ta, int Tb, int Tu, int Tv, int& regionCount) {
	for (int row = 0; row < regionLabels.rows; row++) {
		for (int col = 0; col < regionLabels.cols; col++) {
			if (regionLabels.at<ushort>(row, col) == 0) {
				regionCount++;

				queue<Point> pixelQueue;
				pixelQueue.push(Point(col, row));

				regionLabels.at<ushort>(row, col) = regionCount;
				double avgA = static_cast<int>(channelA.at<uchar>(row, col));
				double avgB = static_cast<int>(channelB.at<uchar>(row, col));
				double avgU = static_cast<int>(channelU.at<uchar>(row, col));
				double avgV = static_cast<int>(channelV.at<uchar>(row, col));
				int pixelTotal = 1;

				while (!pixelQueue.empty()) {
					Point currentPixel = pixelQueue.front();
					pixelQueue.pop();

					int currentX = currentPixel.x;
					int currentY = currentPixel.y;

					for (int offsetY = -1; offsetY <= 1; ++offsetY) {
						for (int offsetX = -1; offsetX <= 1; ++offsetX) {
							if (offsetX == 0 && offsetY == 0) continue;

							int neighborX = currentX + offsetX;
							int neighborY = currentY + offsetY;

							if (isInside(channelA, neighborY, neighborX) &&
								regionLabels.at<ushort>(neighborY, neighborX) == 0) {

								double diffA = static_cast<int>(channelA.at<uchar>(neighborY, neighborX)) - avgA;
								double diffB = static_cast<int>(channelB.at<uchar>(neighborY, neighborX)) - avgB;
								double diffU = static_cast<int>(channelU.at<uchar>(neighborY, neighborX)) - avgU;
								double diffV = static_cast<int>(channelV.at<uchar>(neighborY, neighborX)) - avgV;
								double distance = sqrt(diffA * diffA + diffB * diffB + diffU * diffU + diffV * diffV);

								if (distance < T) {
								//if(abs(diffA) < Ta && abs(diffB) < Tb && abs(diffU) < Tu && abs(diffV) < Tv){

									pixelQueue.push(Point(neighborX, neighborY));
									regionLabels.at<ushort>(neighborY, neighborX) = regionCount;

									avgA = (avgA * pixelTotal + static_cast<int>(channelA.at<uchar>(neighborY, neighborX))) / (pixelTotal + 1);
									avgB = (avgB * pixelTotal + static_cast<int>(channelB.at<uchar>(neighborY, neighborX))) / (pixelTotal + 1);
									avgU = (avgU * pixelTotal + static_cast<int>(channelU.at<uchar>(neighborY, neighborX))) / (pixelTotal + 1);
									avgV = (avgV * pixelTotal + static_cast<int>(channelV.at<uchar>(neighborY, neighborX))) / (pixelTotal + 1);
									pixelTotal++;
								}
							}
						}
					}
				}
			}
		}
	}
}


Mat visualizeRegions(const Mat& labels, int k) {
	Mat output;
	Mat normalizedLabels;
	labels.convertTo(normalizedLabels, CV_8UC1, 255.0 / k);

	applyColorMap(normalizedLabels, output, COLORMAP_RAINBOW);

	return output;
}


void Proiect() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty())
			continue;

		Mat org = src.clone();

		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat Lab, Luv;
		cvtColor(src, Lab, COLOR_BGR2Lab);
		cvtColor(src, Luv, COLOR_BGR2Luv);

		vector<Mat> labChannels(3), luvChannels(3);
		split(Lab, labChannels);
		split(Luv, luvChannels);

		Mat labelsA = Mat::zeros(Lab.size(), CV_16UC1);
		Mat labelsB = Mat::zeros(Lab.size(), CV_16UC1);
		Mat labelsU = Mat::zeros(Luv.size(), CV_16UC1);
		Mat labelsV = Mat::zeros(Luv.size(), CV_16UC1);
		Mat labelsCombined = Mat::zeros(Luv.size(), CV_16UC1);

		int Ta = 8, Tb = 15, Tu = 10, Tv = 12, Tcombined = 20;
		int kA = 1, kB = 1, kU = 1, kV = 1, kCombined = 1;

		regionGrowing(labelsA, labChannels[1], Ta, kA);
		regionGrowing(labelsB, labChannels[2], Tb, kB);
		regionGrowing(labelsU, luvChannels[1], Tu, kU);
		regionGrowing(labelsV, luvChannels[2], Tv, kV);
		regionGrowingCombined(labelsCombined, labChannels[1], labChannels[2], luvChannels[1], luvChannels[2], Tcombined, Ta, Tb, Tu, Tv, kCombined);

		Mat regionA = visualizeRegions(labelsA, kA);
		Mat regionB = visualizeRegions(labelsB, kB);
		Mat regionU = visualizeRegions(labelsU, kU);
		Mat regionV = visualizeRegions(labelsV, kV);
		Mat regionCombined = visualizeRegions(labelsCombined, kCombined);

		imshow("A", regionA);
		imshow("B", regionB);
		imshow("U", regionU);
		imshow("V", regionV);
		imshow("Combined", regionCombined);

		imshow("Original", org);

		waitKey(0);
	}
}



int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Test video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - OpenCV labeling\n");
		printf(" 14 - BGR to HSV\n");
		printf(" 15 - HSV histogram\n");
		printf(" 16 - Binarizare manuala\n");
		printf(" 17 - Binarizare automata\n");
		printf(" 18 - Live Info\n");
		printf(" 19 - Segmentation\n");
		printf(" 20 - Region Growing Hue\n");
		printf(" 21 - Region Growing Value\n");
		printf(" 22 - Corner Detection\n");
		printf(" 23 - Corner Sub-Pix\n");
		printf(" 24 - Corner Harris\n");
		printf(" 25 - Corner Harris NMS\n");
		printf(" 26 - Video Background Subtraction\n");
		printf(" 27 - Horn-Schunk\n");
		printf(" 28 - LK\n");
		printf(" 29 - Dense Optical Flow\n");
		printf(" 30 - Face Detection\n");
		printf(" 31 - Face Detection Video\n");
		printf(" 32 - Face Detection LBP\n");
		printf(" 33 - Face Validation\n");
		printf(" 34 - Person Detection\n");
		printf(" 35 - Person Detection HOG\n");
		printf(" 36 - Proiect\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				testContour();
				break;
			case 14:
				myBGR2HSV();
				break;
			case 15:
				histogramHSV();
				break;
			case 16:
				binarizareManuala();
				break;
			case 17:
				binarizareAutomata();
				break;
			case 18:
				liveInfo();
				break;
			case 19:
				imgSegmentation();
				break;
			case 20:
				regionGrowingHue();
				break;
			case 21:
				regionGrowingValue();
				break;
			case 22:
				cornerDetection();
				break;
			case 23:
				cornerSubPix();
				break;
			case 24:
				cornerHarris();
				break;
			case 25:
				cornerHarrisNMS();
				break;
			case 26:
				videoBS();
				break;
			case 27:
				opticFlux();
				break;
			case 28:
				LK();
				break;
			case 29:
				denseOpticalFlow();
				break;
			case 30:
				faceDetection();
				break;
			case 31:
				faceDetectionVideo();
				break;
			case 32:
				faceDetectionLBP();
				break;
			case 33:
				faceValidation();
				break;
			case 34:
				bodyDetection();
				break;
			case 35:
				BodyDetectandDisplayHOG();
				break;
			case 36:
				Proiect();
				break;
		}
	}
	while (op!=0);
	return 0;
}