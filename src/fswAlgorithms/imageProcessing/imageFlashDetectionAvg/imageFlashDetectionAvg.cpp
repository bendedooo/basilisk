/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
*/

#include "imageFlashDetectionAvg.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <iomanip>
#include <cstdio>

ImageFlashDetectionAvg::ImageFlashDetectionAvg()
{
    this->numFramesToAverage = 5;
    this->differenceThreshold = 25;
    this->minimumArea = 2;
    this->maximumArea = 5000;
    this->blurSize = 1;
    this->noiseSF = 2.0;
}

ImageFlashDetectionAvg::~ImageFlashDetectionAvg() = default;

void ImageFlashDetectionAvg::Reset(uint64_t CurrentSimNanos)
{
    this->frameHistory.clear();

    if (!this->imageInMsg.isLinked()) {
        bskLogger.bskError("ImageFlashDetectionAvg.imageInMsg wasn't connected.");
    }

    if (this->numFramesToAverage < 1) {
        bskLogger.bskError("ImageFlashDetectionAvg.numFramesToAverage must be at least 1.");
        this->numFramesToAverage = 1;
    }
    if (this->minimumArea < 1) {
        bskLogger.bskLog(BSK_WARNING, "ImageFlashDetectionAvg.minimumArea was below 1; setting it to 1.");
        this->minimumArea = 1;
    }
    if (this->maximumArea < this->minimumArea) {
        bskLogger.bskLog(BSK_WARNING, "ImageFlashDetectionAvg.maximumArea was below minimumArea; setting it equal to minimumArea.");
        this->maximumArea = this->minimumArea;
    }
}

void ImageFlashDetectionAvg::UpdateState(uint64_t CurrentSimNanos)
{
    OpNavCirclesMsgPayload output = this->opnavCirclesOutMsg.zeroMsgPayload;

    if (!this->imageInMsg.isLinked()) {
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    CameraImageMsgPayload input = this->imageInMsg();
    if (input.valid != 1 || input.imagePointer == nullptr || input.imageBufferLength <= 0) {
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    const auto* imageBegin = static_cast<const unsigned char*>(input.imagePointer);
    std::vector<unsigned char> encoded(imageBegin, imageBegin + input.imageBufferLength);
    cv::Mat frame = cv::imdecode(encoded, cv::IMREAD_COLOR);

    if (frame.empty()) {
        bskLogger.bskLog(BSK_WARNING, "ImageFlashDetectionAvg: unable to decode input image.");
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    if (this->blurSize > 1) {
        const int kernel = (this->blurSize % 2 == 0) ? this->blurSize + 1 : this->blurSize;
        cv::GaussianBlur(gray, gray, cv::Size(kernel, kernel), 0.0);
    }

    // A resolution change invalidates the old pixelwise history.
    if (!this->frameHistory.empty() &&
        this->frameHistory.back().size() != gray.size()) {
        this->frameHistory.clear();
    }

    const int requiredFrames = std::max(1, this->numFramesToAverage);

    // Warm-up: collect N frames before trying to detect a change.
    // The current frame is not included in the baseline used for its own detection.
    if (static_cast<int>(this->frameHistory.size()) < requiredFrames) {
        this->frameHistory.push_back(gray.clone());
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    cv::Mat average32F = cv::Mat::zeros(gray.size(), CV_32FC1);
    for (const cv::Mat& historicalFrame : this->frameHistory) {
        cv::Mat historical32F;
        historicalFrame.convertTo(historical32F, CV_32FC1);
        average32F += historical32F;
    }
    average32F *= (1.0 / static_cast<double>(this->frameHistory.size()));

    cv::Mat averageGray;
    average32F.convertTo(averageGray, CV_8UC1);

    // Saturating subtraction retains only positive brightness changes.
    cv::Mat positiveDifference;
    cv::subtract(gray, averageGray, positiveDifference);

    cv::Mat binaryMask;
    cv::threshold(positiveDifference, binaryMask,
                  this->differenceThreshold, 255, cv::THRESH_BINARY);

    cv::Mat labels, stats, centroids;
    const int componentCount = cv::connectedComponentsWithStats(
        binaryMask, labels, stats, centroids, 8, CV_32S);

    int bestLabel = -1;
    int bestArea = 0;

    // Preserve the existing detector's largest-qualifying-region behavior.
    for (int label = 1; label < componentCount; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= this->minimumArea &&
            area <= this->maximumArea &&
            area > bestArea) {
            bestArea = area;
            bestLabel = label;
        }
    }

    output.timeTag = input.timeTag;
    output.cameraID = input.cameraID;

    if (bestLabel >= 0) {
        const double cx = centroids.at<double>(bestLabel, 0);
        const double cy = centroids.at<double>(bestLabel, 1);
        const double equivalentRadius = std::sqrt(static_cast<double>(bestArea) / std::acos(-1.0));

        // Brightness diagnostics for the selected candidate.
        cv::Mat componentMask = (labels == bestLabel);

        // Peak intensity and mean intensity within the detected component.
        double peakIntensity = 0.0;
        cv::minMaxLoc(gray, nullptr, &peakIntensity, nullptr, nullptr, componentMask);

        const double meanCandidate = cv::mean(gray, componentMask)[0];

        // Mean temporal difference inside the detected component.
        const double meanTemporalDifference =
            cv::mean(positiveDifference, componentMask)[0];

        // Measure local background in a ring around the candidate.
        // The ring excludes the candidate itself.
        const int radius = std::max(5, static_cast<int>(std::ceil(equivalentRadius)) + 5);

        cv::Mat outerMask = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::circle(outerMask,
                   cv::Point(cvRound(cx), cvRound(cy)),
                   radius, cv::Scalar(255), cv::FILLED);

        cv::Mat ringMask;
        cv::Mat inverseComponentMask;
        cv::bitwise_not(componentMask, inverseComponentMask);
        cv::bitwise_and(outerMask, inverseComponentMask, ringMask);

        const double localBackground =
            cv::countNonZero(ringMask) > 0
                ? cv::mean(gray, ringMask)[0]
                : 0.0;

        const double localContrast = peakIntensity - localBackground;
        const double relativeContrast =
            (localContrast / std::max(1.0, localBackground)) * 100.0;

        std::fprintf(stderr,
            "\n[ImageFlashDetectionAvg]"
            " time = %.2f s"
            ", center = (%.2f, %.2f)"
            ", area = %d"
            ", peak = %.2f"
            ", candidate mean = %.2f"
            ", local background = %.2f"
            ", peak-background = %.2f"
            ", relative contrast = %.2f%%"
            ", mean temporal difference = %.2f\n",
            input.timeTag / 1.0e9,
            cx, cy,
            bestArea,
            peakIntensity,
            meanCandidate,
            localBackground,
            localContrast,
            relativeContrast,
            meanTemporalDifference
        );
        std::fflush(stderr);


        output.valid = 1;
        output.planetIds[0] = 2; // Moon ID used by the current OpNav scenario
        output.circlesCenters[0] = cx;
        output.circlesCenters[1] = cy;
        output.circlesRadii[0] = equivalentRadius;
        output.uncertainty[0] = this->noiseSF;
        output.uncertainty[4] = this->noiseSF;
        output.uncertainty[8] = this->noiseSF;
	

    }

    // Update history only after comparing the current frame against the old history.
    this->frameHistory.push_back(gray.clone());
    while (static_cast<int>(this->frameHistory.size()) > requiredFrames) {
        this->frameHistory.pop_front();
    }

    this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
}
