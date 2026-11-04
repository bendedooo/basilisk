/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
*/

#include "impactFlashDetection.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


namespace {

// Same as numpy.percentile(values, p) (linear interpolation).
double percentileOf(const cv::Mat& image, double p)
{
    std::vector<float> values;
    values.reserve(image.total());

    if (image.isContinuous()) {
        const float* data = image.ptr<float>(0);
        values.assign(data, data + image.total());
    } else {
        for (int r = 0; r < image.rows; r++) {
            const float* row = image.ptr<float>(r);
            values.insert(values.end(), row, row + image.cols);
        }
    }

    if (values.empty()) {
        return 0.0;
    }

    double position = (p / 100.0) * (values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(position));
    size_t upper = std::min(lower + 1, values.size() - 1);
    double fraction = position - lower;

    std::nth_element(values.begin(), values.begin() + lower, values.end());
    double lowerValue = values[lower];

    std::nth_element(values.begin(), values.begin() + upper, values.end());
    double upperValue = values[upper];

    return lowerValue + fraction * (upperValue - lowerValue);
}

}  // namespace


ImpactFlashDetection::ImpactFlashDetection()
{
    this->backgroundAlpha = 0.35;
    this->thresholdSigma = 5.0;
    this->highThresholdRatio = 1.0;
    this->lowThresholdRatio = 0.5;
    this->warmupFrames = 30;             // frames used to learn background AND noise

    this->minimumArea = 2;
    this->maximumArea = 5000;
    this->blurSize = 1;
    this->noiseSF = 2.0;

    // Adaptive per-pixel threshold (replaces mu + 5 sigma)
    this->noiseBeta = 0.05;              // memory of the per-pixel noise variance (~1/beta frames)
    this->kHigh = 6.0;                   // seed threshold, in units of local noise sigma
    this->kLow = 3.0;                    // region-growing threshold (hysteresis), in sigma
    this->sigmaMin = 0.5;                // gray levels; uint8 quantization floor
    this->varianceWindow = 5;            // spatial smoothing of the variance map (px)
    this->sigmaDilate = 1;               // local max of sigma over (2n+1)^2 px (moving edges)
    this->limbMargin = 4;                // reject candidates whose peak is this close to the limb
    this->discFraction = 0.05;           // disc = background > fraction * 99.5th percentile

    this->warmupCounter = 0;
    this->backgroundFreezeUntilNanos = 0;
    this->candidateActive = false;
    this->lastImageTimeTag = 0;
    this->haveLastImage = false;

}


ImpactFlashDetection::~ImpactFlashDetection() = default;


void ImpactFlashDetection::Reset(uint64_t CurrentSimNanos)
{
    this->backgroundGray.release();
    this->noiseVariance.release();

    this->warmupCounter = 0;
    this->backgroundFreezeUntilNanos = 0;
    this->candidateActive = false;
    this->lastImageTimeTag = 0;
    this->haveLastImage = false;

    if (!this->imageInMsg.isLinked()) {
        bskLogger.bskError(
            "ImpactFlashDetection.imageInMsg wasn't connected."
        );
    }
}


void ImpactFlashDetection::UpdateState(uint64_t CurrentSimNanos)
{
    OpNavCirclesMsgPayload output =
        this->opnavCirclesOutMsg.zeroMsgPayload;

    if (!this->imageInMsg.isLinked()) {
        this->opnavCirclesOutMsg.write(
            &output, this->moduleID, CurrentSimNanos
        );
        return;
    }

    CameraImageMsgPayload input = this->imageInMsg();

    if (input.valid != 1 ||
        input.imagePointer == nullptr ||
        input.imageBufferLength <= 0) {

        this->opnavCirclesOutMsg.write(
            &output, this->moduleID, CurrentSimNanos
        );
        return;
    }


    // ------------------------------------------------------------
    // 0. Process each camera image only once
    // ------------------------------------------------------------
    // If this module runs faster than the camera, the same image is
    // read again. Processing it twice would give a ~0 difference and
    // shrink the learned noise, which later causes false detections.

    if (this->haveLastImage &&
        input.timeTag == this->lastImageTimeTag) {

        this->opnavCirclesOutMsg.write(
            &output, this->moduleID, CurrentSimNanos
        );
        return;
    }

    this->lastImageTimeTag = input.timeTag;
    this->haveLastImage = true;


    // ------------------------------------------------------------
    // 1. Decode input image
    // ------------------------------------------------------------

    std::vector<unsigned char> encoded(
        static_cast<unsigned char*>(input.imagePointer),
        static_cast<unsigned char*>(input.imagePointer)
            + input.imageBufferLength
    );

    cv::Mat frame = cv::imdecode(encoded, cv::IMREAD_COLOR);

    if (frame.empty()) {
        bskLogger.bskLog(
            BSK_WARNING,
            "ImpactFlashDetection: unable to decode input image."
        );

        this->opnavCirclesOutMsg.write(
            &output, this->moduleID, CurrentSimNanos
        );
        return;
    }


    // ------------------------------------------------------------
    // 2. Convert image to grayscale
    // ------------------------------------------------------------

    cv::Mat gray;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    if (this->blurSize > 1) {

        int kernel = this->blurSize % 2 == 0
            ? this->blurSize + 1
            : this->blurSize;

        cv::GaussianBlur(
            gray,
            gray,
            cv::Size(kernel, kernel),
            0.0
        );
    }


    // ------------------------------------------------------------
    // 3. Convert grayscale image to floating point
    // ------------------------------------------------------------

    cv::Mat grayFloat;

    gray.convertTo(grayFloat, CV_32F);


    // ------------------------------------------------------------
    // 4. Initialize background if necessary
    // ------------------------------------------------------------

    if (this->backgroundGray.empty() ||
        this->backgroundGray.size() != grayFloat.size()) {

        this->backgroundGray = grayFloat.clone();

        this->warmupCounter = 0;
    }


    // ------------------------------------------------------------
    // 5. Warm-up: learn the background AND the per-pixel noise
    // ------------------------------------------------------------

    double alpha = std::max(
        0.0,
        std::min(1.0, this->backgroundAlpha)
    );

    double beta = std::max(
        0.0,
        std::min(1.0, this->noiseBeta)
    );

    if (this->warmupCounter < this->warmupFrames) {

        if (this->warmupCounter == 0 ||
            this->noiseVariance.empty() ||
            this->noiseVariance.size() != grayFloat.size()) {

            this->backgroundGray = grayFloat.clone();
            this->noiseVariance = cv::Mat::zeros(
                grayFloat.size(), CV_32F
            );

        } else {

            cv::Mat warmupDifference = grayFloat - this->backgroundGray;
            cv::Mat warmupSquared = warmupDifference.mul(warmupDifference);

            // Running mean of the squared difference
            this->noiseVariance +=
                (warmupSquared - this->noiseVariance)
                * (1.0 / this->warmupCounter);

            cv::accumulateWeighted(
                grayFloat,
                this->backgroundGray,
                alpha
            );
        }

        this->warmupCounter++;

        this->opnavCirclesOutMsg.write(
            &output,
            this->moduleID,
            CurrentSimNanos
        );

        return;
    }


    // ------------------------------------------------------------
    // 6. Signed difference image
    // ------------------------------------------------------------

    cv::Mat differenceFloat;

    cv::subtract(
        grayFloat,
        this->backgroundGray,
        differenceFloat
    );


    // ------------------------------------------------------------
    // 7. Per-pixel adaptive threshold from the learned noise
    // ------------------------------------------------------------

    int window = std::max(1, this->varianceWindow);

    cv::Mat smoothedVariance;

    cv::blur(
        this->noiseVariance,
        smoothedVariance,
        cv::Size(window, window)
    );

    cv::Mat localSigma;

    cv::max(
        smoothedVariance,
        this->sigmaMin * this->sigmaMin,
        smoothedVariance
    );

    cv::sqrt(smoothedVariance, localSigma);

    if (this->sigmaDilate > 0) {

        int size = 2 * this->sigmaDilate + 1;

        cv::dilate(
            localSigma,
            localSigma,
            cv::Mat::ones(size, size, CV_8U)
        );
    }

    cv::Mat snr;

    cv::divide(differenceFloat, localSigma, snr);

    // Moon disc from the background, shrunk by limbMargin (thesis 3.2.2)
    double discLevel =
        this->discFraction * percentileOf(this->backgroundGray, 99.5);

    cv::Mat discMask;

    cv::compare(
        this->backgroundGray,
        discLevel,
        discMask,
        cv::CMP_GT
    );

    cv::Mat discInside;

    int erodeSize = 2 * std::max(0, this->limbMargin) + 1;

    cv::erode(
        discMask,
        discInside,
        cv::Mat::ones(erodeSize, erodeSize, CV_8U)
    );


    // ------------------------------------------------------------
    // 8. High and low threshold masks (in units of local sigma)
    // ------------------------------------------------------------

    cv::Mat highMask;
    cv::Mat lowMask;

    cv::compare(snr, this->kHigh, highMask, cv::CMP_GT);
    cv::compare(snr, this->kLow, lowMask, cv::CMP_GT);


// ------------------------------------------------------------
// 9. Connected-component analysis using low threshold
// ------------------------------------------------------------

cv::Mat labels;
cv::Mat stats;
cv::Mat centroids;

int componentCount =
    cv::connectedComponentsWithStats(
        lowMask,
        labels,
        stats,
        centroids,
        8,
        CV_32S
    );


struct Candidate {
    int label;
    int area;
    double cx;
    double cy;
};

std::vector<Candidate> candidates;




for (int label = 1; label < componentCount; label++) {

    int area = stats.at<int>(
        label,
        cv::CC_STAT_AREA
    );

    if (area < this->minimumArea ||
        area > this->maximumArea) {
        continue;
    }

    // Create a mask for the current connected component.
    cv::Mat componentMask;

    cv::compare(
        labels,
        label,
        componentMask,
        cv::CMP_EQ
    );

    // Check whether this component contains
    // at least one high-threshold pixel.
    cv::Mat highPixels;

    cv::bitwise_and(
        componentMask,
        highMask,
        highPixels
    );

    if (cv::countNonZero(highPixels) == 0) {
        continue;
    }

    // Thesis 3.2.2: the strongest pixel must lie inside the disc
    cv::Point peakLocation;

    cv::minMaxLoc(
        snr,
        nullptr,
        nullptr,
        nullptr,
        &peakLocation,
        componentMask
    );

    if (discInside.at<unsigned char>(peakLocation) == 0) {
        continue;
    }

    Candidate candidate;

    candidate.label = label;
    candidate.area = area;

    candidate.cx = centroids.at<double>(
        label,
        0
    );

    candidate.cy = centroids.at<double>(
        label,
        1
    );

    candidates.push_back(candidate);


}

    // ------------------------------------------------------------
    // 10. Sort candidates by area
    // ------------------------------------------------------------

    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.area > b.area;
        }
    );


    // ------------------------------------------------------------
    // 11. Write candidate measurements
    // ------------------------------------------------------------

    output.timeTag = input.timeTag;
    output.cameraID = input.cameraID;

    int numberOfCandidates = std::min(
        static_cast<int>(candidates.size()),
        MAX_CIRCLE_NUM
    );

    for (int i = 0; i < numberOfCandidates; i++) {

        const Candidate& candidate = candidates[i];

        double equivalentRadius = std::sqrt(
            static_cast<double>(candidate.area) / CV_PI
        );

        output.planetIds[i] = 2;

        output.circlesCenters[2 * i] =
            candidate.cx;

        output.circlesCenters[2 * i + 1] =
            candidate.cy;

        output.circlesRadii[i] =
            equivalentRadius;
    }

    if (numberOfCandidates > 0) {

        output.valid = 1;

        output.uncertainty[0] = this->noiseSF;
        output.uncertainty[4] = this->noiseSF;
        output.uncertainty[8] = this->noiseSF;
    }


// ------------------------------------------------------------
// 12. Temporarily freeze background after candidate detection
// ------------------------------------------------------------

bool candidateDetected = !candidates.empty();

// Start the freeze only when a candidate first appears.
// Do not extend it on every consecutive detection.
if (candidateDetected && !this->candidateActive) {
    this->backgroundFreezeUntilNanos =
        CurrentSimNanos + 198000000ULL;
}

this->candidateActive = candidateDetected;

// Resume normal background updating after the freeze.
if (CurrentSimNanos >= this->backgroundFreezeUntilNanos) {
    cv::accumulateWeighted(
        grayFloat,
        this->backgroundGray,
        alpha
    );

    cv::Mat differenceSquared =
        differenceFloat.mul(differenceFloat);

    cv::accumulateWeighted(
        differenceSquared,
        this->noiseVariance,
        beta
    );
}

    // ------------------------------------------------------------
    // 13. Publish output
    // ------------------------------------------------------------

    this->opnavCirclesOutMsg.write(
        &output,
        this->moduleID,
        CurrentSimNanos
    );
}