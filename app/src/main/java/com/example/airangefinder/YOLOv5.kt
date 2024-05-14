package com.example.airangefinder

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import android.util.Size
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Arrays
import java.util.PriorityQueue

class YOLOv5 {
    val inputSize = Size(320, 320)
    val outputSize = intArrayOf(1, 6300, 85)

    private val IS_INT8 = false
    private val DETECT_THRESHOLD = 0.25f
    private val IOU_THRESHOLD = 0.45f
    private val IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f

    val labelFile = "coco_label.txt"
    private var BITMAP_HEIGHT = 0
    private var BITMAP_WIDTH = 0
    var input5SINT8QuantParams = MetadataExtractor.QuantizationParams(0.003921568859368563f, 0)
    var output5SINT8QuantParams = MetadataExtractor.QuantizationParams(0.006305381190031767f, 5)
    var modelFile: String? = null

        private set
    private var tflite: Interpreter? = null
    private var associatedAxisLabels: List<String>? = null
    var options = Interpreter.Options()

    fun setModelFile(modelFile: String) {
        this.modelFile = modelFile
    }

    fun initialModel(activity: Context?) {
        try {
            val tfliteModel: ByteBuffer = FileUtil.loadMappedFile(
                activity!!,
                modelFile!!
            )
            tflite = Interpreter(tfliteModel, options)
            associatedAxisLabels = FileUtil.loadLabels(
                activity,
                labelFile
            )
        } catch (e: IOException) {
            Log.e("ERROR", "Error reading model or label: ", e)
        }
    }

    fun detect(bitmap: Bitmap): ArrayList<Recognition> {
        BITMAP_HEIGHT = bitmap.height
        BITMAP_WIDTH = bitmap.width

        var YOLOv5input: TensorImage
        val imageProcessor: ImageProcessor
        if (IS_INT8) {
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(
                    QuantizeOp(
                        input5SINT8QuantParams.zeroPoint.toFloat(),
                        input5SINT8QuantParams.scale
                    )
                )
                .add(CastOp(DataType.UINT8))
                .build()
            YOLOv5input = TensorImage(DataType.UINT8)
        } else {
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
            YOLOv5input = TensorImage(DataType.FLOAT32)
        }
        YOLOv5input.load(bitmap)
        YOLOv5input = imageProcessor.process(YOLOv5input)

        var probabilityBuffer: TensorBuffer
        probabilityBuffer = if (IS_INT8) {
            TensorBuffer.createFixedSize(outputSize, DataType.UINT8)
        } else {
            TensorBuffer.createFixedSize(outputSize, DataType.FLOAT32)
        }

        if (null != tflite) {
            Log.d(
                ">>> ",
                YOLOv5input.tensorBuffer.flatSize.toString() + " " + probabilityBuffer.flatSize
            )
            tflite!!.run(YOLOv5input.buffer, probabilityBuffer.buffer)
        }

        if (IS_INT8) {
            val tensorProcessor = TensorProcessor.Builder()
                .add(
                    DequantizeOp(
                        output5SINT8QuantParams.zeroPoint.toFloat(),
                        output5SINT8QuantParams.scale
                    )
                )
                .build()
            probabilityBuffer = tensorProcessor.process(probabilityBuffer)
        }

        val recognitionArray = probabilityBuffer.floatArray
        val allRecognitions = ArrayList<Recognition>()
        for (i in 0 until outputSize[1]) {
            val gridStride = i * outputSize[2]
            val x = recognitionArray[0 + gridStride] * BITMAP_WIDTH
            val y = recognitionArray[1 + gridStride] * BITMAP_HEIGHT
            val w = recognitionArray[2 + gridStride] * BITMAP_WIDTH
            val h = recognitionArray[3 + gridStride] * BITMAP_HEIGHT
            val xmin = Math.max(0.0, x - w / 2.0).toInt()
            val ymin = Math.max(0.0, y - h / 2.0).toInt()
            val xmax = Math.min(BITMAP_WIDTH.toDouble(), x + w / 2.0).toInt()
            val ymax = Math.min(BITMAP_HEIGHT.toDouble(), y + h / 2.0).toInt()
            val confidence = recognitionArray[4 + gridStride]
            val classScores =
                Arrays.copyOfRange(recognitionArray, 5 + gridStride, outputSize[2] + gridStride)

            var labelId = 0
            var maxLabelScores = 0f
            for (j in classScores.indices) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j]
                    labelId = j
                }
            }
            val r = Recognition(
                labelId,
                "",
                maxLabelScores,
                confidence,
                RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
            )
            allRecognitions.add(
                r
            )
        }

        val nmsRecognitions = nms(allRecognitions)
        val nmsFilterBoxDuplicationRecognitions = nmsAllClass(nmsRecognitions)

        for (recognition in nmsFilterBoxDuplicationRecognitions) {
            val labelId: Int? = recognition.gettingLabelId()
            val labelName = associatedAxisLabels!![labelId!!]
            recognition.settingLabelName(labelName)
        }
        return nmsFilterBoxDuplicationRecognitions
    }

    protected fun nms(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()

        for (i in 0 until outputSize[2] - 5) {
            val pq = PriorityQueue<Recognition>(
                6300,
                object : Comparator<Recognition?> {
                    override fun compare(l: Recognition?, r: Recognition?): Int {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return java.lang.Float.compare(r?.gettingConfidence()?: Float.MIN_VALUE, l?.gettingConfidence()?: Float.MIN_VALUE)
                    }
                })

            for (j in allRecognitions.indices) {
                if (allRecognitions[j].gettingLabelId() === i && allRecognitions[j].gettingConfidence() > DETECT_THRESHOLD) {
                    pq.add(allRecognitions[j])
                }
            }

            while (pq.size > 0) {
                val a = arrayOfNulls<Recognition>(pq.size)
                val detections: Array<Recognition> = pq.toArray(a)
                val max = detections[0]
                nmsRecognitions.add(max)
                pq.clear()
                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    if (boxIou(max.gettingLocation(), detection.gettingLocation()) < IOU_THRESHOLD) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsRecognitions
    }

    protected fun nmsAllClass(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()
        val pq = PriorityQueue<Recognition>(
            100,
            object : Comparator<Recognition?> {
                override fun compare(l: Recognition?, r: Recognition?): Int = java.lang.Float.compare(r?.gettingConfidence() ?: Float.MIN_VALUE, l?.gettingConfidence() ?: Float.MIN_VALUE)
            })

        for (j in allRecognitions.indices) {
            if (allRecognitions[j].gettingConfidence() > DETECT_THRESHOLD) {
                pq.add(allRecognitions[j])
            }
        }
        while (pq.size > 0) {
            val a = arrayOfNulls<Recognition>(pq.size)
            val detections: Array<Recognition> = pq.toArray(a)
            val max = detections[0]
            nmsRecognitions.add(max)
            pq.clear()
            for (k in 1 until detections.size) {
                val detection = detections[k]
                if (boxIou(
                        max.gettingLocation(),
                        detection.gettingLocation()
                    ) < IOU_CLASS_DUPLICATED_THRESHOLD
                ) {
                    pq.add(detection)
                }
            }
        }
        return nmsRecognitions
    }

    protected fun boxIou(a: RectF, b: RectF): Float {
        val union = boxUnion(a, b)
        return if (union <= 0) 1f else boxIntersection(a, b) / union
    }

    protected fun boxIntersection(a: RectF, b: RectF): Float {
        val maxLeft = if (a.left > b.left) a.left else b.left
        val maxTop = if (a.top > b.top) a.top else b.top
        val minRight = if (a.right < b.right) a.right else b.right
        val minBottom = if (a.bottom < b.bottom) a.bottom else b.bottom
        val w = minRight - maxLeft
        val h = minBottom - maxTop
        return if (w < 0 || h < 0) 0f else w * h
    }

    protected fun boxUnion(a: RectF, b: RectF): Float = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - boxIntersection(a, b)
}