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
    this->differenceThreshold = 25;
    this->minimumArea = 2;
    this->maximumArea = 5000;
    this->blurSize = 1;
    this->useFirstFrameAsReference = 1;
    this->noiseSF = 2.0;
    this->referenceInitialized = 0;
}

ImpactFlashDetection::~ImpactFlashDetection() = default;

void ImpactFlashDetection::Reset(uint64_t CurrentSimNanos)
{
    this->referenceGray.release();
    this->referenceInitialized = 0;
    if (!this->imageInMsg.isLinked()) {
        bskLogger.bskError("ImpactFlashDetection.imageInMsg wasn't connected.");
    }
}

void ImpactFlashDetection::UpdateState(uint64_t CurrentSimNanos)
{
    OpNavCirclesMsgPayload output = this->opnavCirclesOutMsg.zeroMsgPayload;
    CameraImageMsgPayload input = this->imageInMsg.zeroMsgPayload;

    if (!this->imageInMsg.isLinked()) {
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    input = this->imageInMsg();
    if (input.valid != 1 || input.imagePointer == nullptr || input.imageBufferLength <= 0) {
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    std::vector<unsigned char> encoded(
        static_cast<unsigned char*>(input.imagePointer),
        static_cast<unsigned char*>(input.imagePointer) + input.imageBufferLength);
    cv::Mat frame = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (frame.empty()) {
        bskLogger.bskLog(BSK_WARNING, "ImpactFlashDetection: unable to decode input image.");
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    if (this->blurSize > 1) {
        int kernel = this->blurSize % 2 == 0 ? this->blurSize + 1 : this->blurSize;
        cv::GaussianBlur(gray, gray, cv::Size(kernel, kernel), 0.0);
    }

    // The first valid frame initializes the baseline; it is not a detection.
    if (!this->referenceInitialized || this->referenceGray.size() != gray.size()) {
        gray.copyTo(this->referenceGray);
        this->referenceInitialized = 1;
        this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
        return;
    }

    cv::Mat positiveDifference;
    cv::subtract(gray, this->referenceGray, positiveDifference);
    cv::Mat binaryMask;
    cv::threshold(positiveDifference, binaryMask, this->differenceThreshold, 255, cv::THRESH_BINARY);

    // Remove isolated noise while preserving small compact flash-like regions.
    cv::Mat labels, stats, centroids;
    int componentCount = cv::connectedComponentsWithStats(
        binaryMask, labels, stats, centroids, 8, CV_32S);

    int bestLabel = -1;
    int bestArea = 0;
    for (int label = 1; label < componentCount; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= this->minimumArea && area <= this->maximumArea && area > bestArea) {
            bestArea = area;
            bestLabel = label;
        }
    }

    output.timeTag = input.timeTag;
    output.cameraID = input.cameraID;
    if (bestLabel >= 0) {
        const double cx = centroids.at<double>(bestLabel, 0);
        const double cy = centroids.at<double>(bestLabel, 1);
        const double equivalentRadius = std::sqrt(static_cast<double>(bestArea) / CV_PI);
        output.valid = 1;
        output.planetIds[0] = 2; // Moon ID used by the current OpNav scenario
        output.circlesCenters[0] = cx;
        output.circlesCenters[1] = cy;
        output.circlesRadii[0] = equivalentRadius;
        output.uncertainty[0] = this->noiseSF;
        output.uncertainty[4] = this->noiseSF;
        output.uncertainty[8] = this->noiseSF;
    }

    // Fixed baseline detects a sustained albedo patch; previous-frame mode
    // instead detects only changes relative to the immediately preceding frame.
    if (this->useFirstFrameAsReference != 1) {
        gray.copyTo(this->referenceGray);
    }

    this->opnavCirclesOutMsg.write(&output, this->moduleID, CurrentSimNanos);
}
