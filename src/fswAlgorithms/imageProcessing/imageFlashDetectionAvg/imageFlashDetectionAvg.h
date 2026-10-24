/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
*/

#ifndef IMAGE_FLASH_DETECTION_AVG_H
#define IMAGE_FLASH_DETECTION_AVG_H

#include <stdint.h>
#include <deque>
#include <opencv2/core.hpp>
#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
#include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"

/*! Detect localized positive brightness changes relative to the pixelwise
 * average of the previous N valid camera frames.
 *
 * The current frame is deliberately excluded from the reference average.
 * The first N valid frames initialize the rolling history and do not produce
 * detections.
 */
class ImageFlashDetectionAvg: public SysModel {
public:
    ImageFlashDetectionAvg();
    ~ImageFlashDetectionAvg();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

public:
    ReadFunctor<CameraImageMsgPayload> imageInMsg; //!< Input camera image
    Message<OpNavCirclesMsgPayload> opnavCirclesOutMsg; //!< Detection output

    int32_t numFramesToAverage; //!< [-] Number of previous frames in reference average
    int32_t differenceThreshold; //!< [pixel intensity] Minimum positive difference
    int32_t minimumArea; //!< [pixels] Minimum connected-component area
    int32_t maximumArea; //!< [pixels] Maximum connected-component area
    int32_t blurSize; //!< [pixels] Odd Gaussian blur kernel size; 1 disables blur
    double noiseSF; //!< [-] Scale factor for output uncertainty

    BSKLogger bskLogger;

private:
    std::deque<cv::Mat> frameHistory;
};

#endif
