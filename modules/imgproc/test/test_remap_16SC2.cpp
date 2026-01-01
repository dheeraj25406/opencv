#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

TEST(Imgproc_Remap, LinearInterpolation_16SC2_Equals_32FC2)
{
    // Small input image
    Mat src = (Mat_<uchar>(2, 2) <<
        10, 20,
        30, 40
    );

    Mat dst16, dst32;

    Size sz = src.size();

    // floating–point map
    Mat map32x(sz, CV_32FC2);
    for (int y = 0; y < sz.height; y++)
        for (int x = 0; x < sz.width; x++)
            map32x.at<Vec2f>(y,x) = Vec2f((float)x + 0.25f, (float)y + 0.25f);

    // convert to CV_16SC2
    Mat map16x;
    map32x.convertTo(map16x, CV_16SC2, (1<<5)); // same scale remap() uses internally

    remap(src, dst32, map32x, noArray(), INTER_LINEAR);
    remap(src, dst16, map16x, noArray(), INTER_LINEAR);

    ASSERT_EQ(dst32.type(), dst16.type());
    ASSERT_EQ(dst32.size(), dst16.size());

    for (int y = 0; y < dst32.rows; y++)
        for (int x = 0; x < dst32.cols; x++)
            ASSERT_EQ(dst32.at<uchar>(y,x), dst16.at<uchar>(y,x));
}

}}