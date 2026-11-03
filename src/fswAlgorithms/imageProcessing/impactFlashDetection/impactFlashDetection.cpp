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


ImpactFlashDetection::ImpactFlashDetection()
{
    this->backgroundAlpha = 0.35;
    this->thresholdSigma = 5.0;
    this->highThresholdRatio = 1.0;
    this->lowThresholdRatio = 0.5;
    this->warmupFrames = 10;

    this->minimumArea = 2;
    this->maximumArea = 5000;
    this->blurSize = 1;
    this->noiseSF = 2.0;

    this->warmupCounter = 0;

}


ImpactFlashDetection::~ImpactFlashDetection() = default;


void ImpactFlashDetection::Reset(uint64_t CurrentSimNanos)
{
    this->backgroundGray.release();

    this->warmupCounter = 0;

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
    // 5. Background warm-up
    // ------------------------------------------------------------

    double alpha = std::max(
        0.0,
        std::min(1.0, this->backgroundAlpha)
    );

    if (this->warmupCounter < this->warmupFrames) {

        this->backgroundGray = grayFloat.clone();

        this->warmupCounter++;

        this->opnavCirclesOutMsg.write(
            &output,
            this->moduleID,
            CurrentSimNanos
        );

        return;
    }




    // ------------------------------------------------------------
    // 6. Calculate positive difference image
    // ------------------------------------------------------------

    cv::Mat differenceFloat;

    cv::subtract(
        grayFloat,
        this->backgroundGray,
        differenceFloat
    );

    // Keep only positive intensity changes.
    cv::Mat positiveDifference;

    cv::threshold(
        differenceFloat,
        positiveDifference,
        0.0,
        0.0,
        cv::THRESH_TOZERO
    );


    // ------------------------------------------------------------
    // 7. Calculate adaptive threshold
    // ------------------------------------------------------------

cv::Scalar imageMean;
cv::Scalar imageStdDev;

cv::meanStdDev(
    positiveDifference,
    imageMean,
    imageStdDev
);

double adaptiveThreshold =
    imageMean[0] +
    this->thresholdSigma * imageStdDev[0];

    // ------------------------------------------------------------
    // 8. Generate binary candidate mask
    // ------------------------------------------------------------


double highThreshold =
    adaptiveThreshold * this->highThresholdRatio;

double lowThreshold =
    adaptiveThreshold * this->lowThresholdRatio;

cv::Mat highMask;
cv::Mat lowMask;

cv::compare(
    positiveDifference,
    highThreshold,
    highMask,
    cv::CMP_GT
);

cv::compare(
    positiveDifference,
    lowThreshold,
    lowMask,
    cv::CMP_GT
);


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

// Mask of accepted flash regions.
// This will also be used to prevent flash pixels
// from contaminating the background update.
cv::Mat candidateMask =
    cv::Mat::zeros(lowMask.size(), CV_8UC1);


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

    // Mark accepted flash pixels.
    candidateMask.setTo(255, componentMask);
}

    // ------------------------------------------------------------
    // 10. Sort candidates by area
    // ------------------------------------------------------------

    std::sort(
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
// 12. Update background excluding detected flash pixels
// ------------------------------------------------------------

cv::Mat backgroundUpdateMask;

cv::bitwise_not(
    candidateMask,
    backgroundUpdateMask
);

cv::accumulateWeighted(
    grayFloat,
    this->backgroundGray,
    alpha,
    backgroundUpdateMask
);


    // ------------------------------------------------------------
    // 13. Publish output
    // ------------------------------------------------------------

    this->opnavCirclesOutMsg.write(
        &output,
        this->moduleID,
        CurrentSimNanos
    );
}