/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
*/

#ifndef IMPACT_FLASH_DETECTION_H
#define IMPACT_FLASH_DETECTION_H

#include <stdint.h>
#include <string>
#include <opencv2/core.hpp>

#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
#include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"

/*! Detect localized positive brightness changes in sequential camera images
 *  using an adaptive background model and adaptive threshold.
 */
class ImpactFlashDetection: public SysModel {
public:
    ImpactFlashDetection();
    ~ImpactFlashDetection();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

public:
    ReadFunctor<CameraImageMsgPayload> imageInMsg;
    Message<OpNavCirclesMsgPayload> opnavCirclesOutMsg;

    double backgroundAlpha;       //!< [-] Background update coefficient
    double thresholdSigma;        //!< [-] Standard-deviation multiplier
    int32_t warmupFrames;          //!< [frames] Frames used to initialize background
    int32_t minimumArea;            //!< [pixels] Minimum connected-component area
    int32_t maximumArea;            //!< [pixels] Maximum connected-component area
    int32_t blurSize;               //!< [pixels] Odd Gaussian blur kernel size
    double noiseSF;                 //!< [-] Output uncertainty scale factor
    double noiseBeta;            //!< [-] memory of the per-pixel noise variance (~1/beta frames)
    double kHigh;                //!< [sigma] seed threshold in units of local noise sigma
    double kLow;                 //!< [sigma] region-growing (hysteresis) threshold
    double sigmaMin;             //!< [gray levels] lower bound of the local noise sigma
    int varianceWindow;          //!< [px] spatial smoothing of the variance map
    int sigmaDilate;             //!< [px] local max of sigma over (2n+1)^2 px
    int limbMargin;              //!< [px] reject candidates whose peak is this close to the limb
    double discFraction;         //!< [-] disc = background > fraction * 99.5th percentile
    double highThresholdRatio; //!< [-] High threshold multiplier
    double lowThresholdRatio; //!< [-] Low threshold multiplier




    BSKLogger bskLogger;

private:
    cv::Mat backgroundGray;         //!< Floating-point background image
    cv::Mat noiseVariance;               //!< per-pixel noise variance
    uint64_t lastImageTimeTag;           //!< time tag of the last processed image
    bool haveLastImage;                  //!< an image has been processed
    int32_t warmupCounter;          //!< Number of background initialization frames
    uint64_t backgroundFreezeUntilNanos;
    bool candidateActive;
};

#endif