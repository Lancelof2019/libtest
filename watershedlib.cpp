#include<iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include<queue>
#include <vector>
#include<random>
#include <cv.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <queue>
#include<string>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;

Mat WaterSegment(Mat src) {
  int row = src.rows;
  int col = src.cols;
  //1. 将RGB图像灰度化
  Mat grayImage;
  cv::cvtColor(src,grayImage,cv::COLOR_BGR2GRAY);

  //2. 使用大津法转为二值图，并做形态学闭合操作
  threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
  //3. 形态学闭操作
  Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
  morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
  //4. 距离变换
  distanceTransform(grayImage, grayImage, DIST_L2, DIST_MASK_3, 5);
  //5. 将图像归一化到[0, 1]范围
  normalize(grayImage, grayImage, 0, 1, NORM_MINMAX);
  //6. 将图像取值范围变为8位(0-255)
  grayImage.convertTo(grayImage, CV_8UC1);
  //7. 再使用大津法转为二值图，并做形态学闭合操作
  threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
  morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
  //8. 使用findContours寻找marks
  vector<vector<Point>> contours;
  findContours(grayImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
  Mat marks = Mat::zeros(grayImage.size(), CV_32SC1);
  for (size_t i = 0; i < contours.size(); i++)
  {
    //static_cast<int>(i+1)是为了分水岭的标记不同，区域1、2、3...这样才能分割
    drawContours(marks, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i + 1)), 2);
  }
  //9. 对原图做形态学的腐蚀操作
  Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
  morphologyEx(src, src, MORPH_ERODE, k);
  //10. 调用opencv的分水岭算法
  watershed(src, marks);
  //11. 随机分配颜色
  vector<Vec3b> colors;
  for (size_t i = 0; i < contours.size(); i++) {
    int r = theRNG().uniform(0, 255);
    int g = theRNG().uniform(0, 255);
    int b = theRNG().uniform(0, 255);
    colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
  }

  // 12. 显示
  Mat dst = Mat::zeros(marks.size(), CV_8UC3);
  int index = 0;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      index = marks.at<int>(i, j);
      if (index > 0 && index <= contours.size()) {
        dst.at<Vec3b>(i, j) = colors[index - 1];
      }
      else if (index == -1)
      {
        dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
      }
      else {
        dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
      }
    }
  }
  return dst;
}


int main(){


clock_t start,end;
start=clock();
Mat img=imread("/opt/testimages/SHX8-038_1568286487.jpg");

imshow("original image",img);
waitKey(0);
Mat imgoutput=WaterSegment(img);
imshow("final output",imgoutput);
waitKey(0);
end=clock();
cout<<"CLOCKS_PER_SEC :"<<CLOCKS_PER_SEC<<endl;
cout<<"The total time the algorithm used is :"<<(double)(end - start)/CLOCKS_PER_SEC<<"s"<<endl;
return 0;


}
