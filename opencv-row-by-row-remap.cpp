// Benchmark for incremental `cv::remap()` vs normal `cv::remap()`.

#include <chrono>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
  // Create a sample image
  Mat src = Mat::zeros(4800, 6400, CV_8UC3);
  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      src.at<Vec3b>(y, x)[0] = x % 256; // Blue channel
      src.at<Vec3b>(y, x)[1] = y % 256; // Green channel
      src.at<Vec3b>(y, x)[2] = (x + y) % 256; // Red channel
    }
  }

  const int dest_rows =  8000;
  const int dest_cols = 10000;

  cout << "dest_rows: " << dest_rows << endl;
  cout << "dest_cols: " << dest_cols << endl;

  // Create a sample mapping function
  cv::Mat_<cv::Vec<float, 2>> mapping(dest_rows, dest_cols);
  for (int y = 0; y < dest_rows; y++) {
    float *mappingRowPtr = mapping.ptr<float>(y);
    for (int x = 0; x < dest_cols; x++) {
      Eigen::Map<Eigen::Vector2f>(mappingRowPtr + x * 2) = Eigen::Vector2f{
        sin(float(x) / dest_cols) * src.cols,
            float(y) / dest_rows  * src.rows
      };
    }
  }

  const auto now = chrono::high_resolution_clock::now;
  const auto isMatEqual = [](const Mat & a, const Mat & b) {
    return sum(a != b) == Scalar(0,0,0);
  };

  vector<int> interp_methods = {
    INTER_LINEAR,
    INTER_LANCZOS4
  };
  vector<string> interp_methods_names = {
    "INTER_LINEAR",
    "INTER_LANCZOS4"
  };

  for (size_t interp_method_index = 0; interp_method_index < interp_methods.size(); interp_method_index++) {
    int interp_method = interp_methods.at(interp_method_index);
    string interp_methods_name = interp_methods_names.at(interp_method_index);

    cout << endl;
    cout << "Interpolation: " << interp_methods_name << endl;

    Mat dest = Mat::zeros(dest_rows, dest_cols, CV_8UC3);
    {
      auto start_time = now();
      cv::remap(src, dest, mapping, {}, interp_method);
      cout << "Full remap took            " << chrono::duration_cast<chrono::milliseconds>(now() - start_time).count() << " ms" << endl;
      // imshow("Remapped", dest);
      // waitKey(0);
    }


    Mat dest_row_incremental(dest_rows, dest_cols, CV_8UC3);
    {
      auto start_time = now();
      for (int r = 0; r < dest_rows; ++r) {
        remap(src, dest_row_incremental.row(r), mapping.row(r), {}, interp_method);
      }
      cout << "Single-row-incremental remap took " << chrono::duration_cast<chrono::milliseconds>(now() - start_time).count() << " ms" << endl;
      // imshow("dest_row_incremental", dest_row_incremental);
      // waitKey(0);
    }
    if (!isMatEqual(dest, dest_row_incremental))
      cout << "Single-row-incremental is NOT EQUAL!" << endl;


    const int MULTI_ROW_STEPS[] = { 128, 256, 512 }; // how many rows to remap() per loop iteration
    for (const int multi_row_step : MULTI_ROW_STEPS) {

      cout << "multi_row_step: " << multi_row_step << endl;

      Mat dest_multi_row_incremental(dest_rows, dest_cols, CV_8UC3);
      {
        auto start_time = now();
        for (int start_row = 0; start_row < dest_rows; start_row += multi_row_step) {
          Range row_range(start_row, min(start_row + multi_row_step, dest_rows));
          Range col_range = Range::all();
          remap(src, dest_multi_row_incremental(row_range, col_range), mapping(row_range, col_range), {}, interp_method);
        }
        cout << "Multi-row-incremental (range-based) remap took " << chrono::duration_cast<chrono::milliseconds>(now() - start_time).count() << " ms" << endl;
        // imshow("dest_multi_row_incremental", dest_multi_row_incremental);
        // waitKey(0);
      }
      if (!isMatEqual(dest, dest_multi_row_incremental))
        cout << "multi-row-incremental (range-based) is NOT EQUAL!" << endl;


      Mat dest_multi_row_incremental_roi(dest_rows, dest_cols, CV_8UC3);
      {
        auto start_time = now();
        for (int start_row = 0; start_row < dest_rows; start_row += multi_row_step) {
          Rect roi(0, start_row, dest_cols, min(multi_row_step, dest_rows - start_row));
          remap(src, dest_multi_row_incremental_roi(roi), mapping(roi), {}, interp_method);
        }
        cout << "Multi-row-incremental (ROI based) remap took " << chrono::duration_cast<chrono::milliseconds>(now() - start_time).count() << " ms" << endl;
        // imshow("dest_multi_row_incremental_roi", dest_multi_row_incremental_roi);
        // waitKey(0);
      }
      if (!isMatEqual(dest, dest_multi_row_incremental_roi))
        cout << "multi-row-incremental (ROI based) is NOT EQUAL!" << endl;


      Mat dest_multi_row_incremental_copying(dest_rows, dest_cols, CV_8UC3);
      {
        auto start_time = now();
        Mat buf;
        for (int start_row = 0; start_row < dest_rows; start_row += multi_row_step) {
          Range row_range(start_row, min(start_row + multi_row_step, dest_rows));
          Range col_range = Range::all();
          remap(src, buf, mapping(row_range, col_range), {}, interp_method);
          buf.copyTo(dest_multi_row_incremental_copying(row_range, col_range));
        }
        cout << "Multi-row-incremental (range-based, copying) remap took " << chrono::duration_cast<chrono::milliseconds>(now() - start_time).count() << " ms" << endl;
        // imshow("dest_multi_row_incremental_copying", dest_multi_row_incremental_copying);
        // waitKey(0);
      }
      if (!isMatEqual(dest, dest_multi_row_incremental_copying))
        cout << "multi-row-incremental (range-based, copying) is NOT EQUAL!" << endl;

    }

  }

  return 0;
}
