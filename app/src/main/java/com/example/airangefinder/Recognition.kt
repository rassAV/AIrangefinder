package com.example.airangefinder

import android.graphics.RectF

data class Recognition(
    var labelId: Int,
    var labelName: String,
    var labelScore: Float,
    var confidence: Float,
    var location: RectF?
) {

    fun gettingLabelId(): Int {
        return labelId
    }

    fun gettingLabelName(): String {
        return labelName
    }

    fun gettingLabelScore(): Float {
        return labelScore
    }

    fun gettingConfidence(): Float {
        return confidence
    }

    fun gettingLocation(): RectF {
        return RectF(location)
    }

    fun settingLocation(location: RectF?) {
        this.location = location
    }

    fun settingLabelName(labelName: String) {
        this.labelName = labelName
    }

    fun settingLabelId(labelId: Int) {
        this.labelId = labelId
    }

    fun settingLabelScore(labelScore: Float?) {
        this.labelScore = labelScore!!
    }

    fun settingConfidence(confidence: Float) {
        this.confidence = confidence
    }

    override fun toString(): String {
        var resultString = ""
        resultString += "$labelId "
        if (labelName != null) {
            resultString += "$labelName "
        }
        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence!! * 100.0f)
        }
        if (location != null) {
            resultString += location.toString() + " "
        }
        return resultString.trim { it <= ' ' }
    }
}