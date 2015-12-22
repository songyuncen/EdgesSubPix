#include "EdgesSubPix.h"
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

static void getCannyKernel(OutputArray _d, double alpha)
{
    int r = cvRound(alpha * 3);
    int ksize = 2 * r + 1;

    _d.create(ksize, 1, CV_16S, -1, true);

    Mat k = _d.getMat();

    vector<float> kerF(ksize, 0.0f);
    kerF[r] = 0.0f;
    double a2 = alpha * alpha;
    float sum = 0.0f;
    for (int x = 1; x <= r; ++x)
    {
        float v = (float)(-x * std::exp(-x * x / (2 * a2)));
        sum += v;
        kerF[r + x] = v;
        kerF[r - x] = -v;
    }
    float scale = 128 / sum;
    for (int i = 0; i < ksize; ++i)
    {
        kerF[i] *= scale;
    }
    Mat temp(ksize, 1, CV_32F, &kerF[0]);
    temp.convertTo(k, CV_16S);
}

// non-maximum supression and hysteresis
static void postCannyFilter(const Mat &src, Mat &dx, Mat &dy, int low, int high, Mat &dst)
{
    ptrdiff_t mapstep = src.cols + 2;
    AutoBuffer<uchar> buffer((src.cols + 2)*(src.rows + 2) + mapstep * 3 * sizeof(int));

    // L2Gradient comparison with square
    high = high * high;
    low = low * low;

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep;
    mag_buf[2] = mag_buf[1] + mapstep;
    memset(mag_buf[0], 0, mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
    (Top-Left Origin)

    1   2   3
    *  *  *
    * * *
    0*******0
    * * *
    *  *  *
    3   2   1
    */

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

#if CV_SSE2
    bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            int j = 0, width = src.cols;
#if CV_SSE2
            if (haveSSE2)
            {
                for (; j <= width - 8; j += 8)
                {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                    __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                    __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                    __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
            }
#elif CV_NEON
            for (; j <= width - 8; j += 8)
            {
                int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j, v_dst);

                v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j + 4, v_dst);
            }
#endif
            for (; j < width; ++j)
                _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];

            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm - 1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i - 1);
        const short* _y = dy.ptr<short>(i - 1);

        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3 / 2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
        __ocv_canny_push:
            if (!prev_flag && m > high && _map[j - mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

//---------------------------------------------------------------------
//          INTERFACE FUNCTION
//---------------------------------------------------------------------
void EdgesSubPix(Mat &gray, double alpha, int low, int high,
    vector<Contour> &contours, OutputArray hierarchy,
    int mode, Point2f offset)
{
    Mat blur;
    GaussianBlur(gray, blur, Size(0, 0), alpha, alpha);

    Mat d;
    getCannyKernel(d, alpha);
    Mat one = Mat::ones(Size(1, 1), CV_16S);
    Mat dx, dy;
    sepFilter2D(blur, dx, CV_16S, d, one);
    sepFilter2D(blur, dy, CV_16S, one, d);

    // non-maximum supression & hysteresis threshold
    Mat edge = Mat::zeros(gray.size(), CV_8UC1);
    double scale = 128.0;                          // sum of half Canny filter is 128
    int lowThresh = cvRound(scale * low);
    int highThresh = cvRound(scale * high);
    postCannyFilter(gray, dx, dy, lowThresh, highThresh, edge);
}

void EdgesSubPix(Mat &gray, double alpha, int low, int high,
    vector<Contour> &contours, Point2f offset)
{
    vector<Vec4i> hierarchy;
    EdgesSubPix(gray, alpha, low, high, contours, hierarchy, RETR_LIST);
}
