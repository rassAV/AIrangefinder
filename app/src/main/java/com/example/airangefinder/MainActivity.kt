package com.example.airangefinder

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.DialogInterface
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.media.Image
import android.os.Bundle
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.util.Size
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import java.lang.Math.abs
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private var preview: ImageView? = null
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null

    private val YOLOv5: YOLOv5 = YOLOv5()
    private val boxPaint = Paint()
    private val boxCenterPaint = Paint()

    private var THOUSANDTH : Float = 0.35f
    private var CALIBRATING = false
    private var calibrationObject: Recognition? = null
    private val widthLabels = arrayOf("person", "bottle", "traffic light", "stop sign", "chair")
    private val sizes = mapOf(
        "person" to 0.5f,
        "car" to 1.5f,
        "bottle" to 0.08f,
        "bicycle" to 1.0f,
        "motorbike" to 1.0f,
        "bus" to 3.5f,
        "train" to 5.0f,
        "truck" to 3.5f,
        "traffic light" to 0.3f,
        "stop sign" to 0.7f,
        "sports ball" to 0.24f,
        "chair" to 0.55f
    )

    private val cameraRequestLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { permissionGranted ->
        if (permissionGranted) {
            initializeCamera()
        } else {
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)

        YOLOv5!!.setModelFile("yolov5s-fp16.tflite")
        YOLOv5.initialModel(this)

        boxPaint.strokeWidth = 5f
        boxPaint.style = Paint.Style.STROKE
        boxPaint.color = Color.RED

        boxCenterPaint.strokeWidth = 10f
        boxCenterPaint.style = Paint.Style.STROKE
        boxCenterPaint.color = Color.GREEN

        cameraRequestLauncher.launch(Manifest.permission.CAMERA)

        val buttonSaveImage: Button = findViewById(R.id.btnCalibrate)
        buttonSaveImage.setOnClickListener {
            answerDialog()
        }
    }

    override fun onPause() {
        super.onPause()
        val sharedPreferences = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putFloat("keySaveTH", THOUSANDTH)
        editor.apply()
    }

    override fun onResume() {
        super.onResume()
        val sharedPreferences = getSharedPreferences("MyPrefs", Context.MODE_PRIVATE)
        THOUSANDTH = sharedPreferences.getFloat("keySaveTH", 0.35f)
    }

    private fun answerDialog() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Калибровка")
        builder.setMessage("Вы хотите начать калибровку?")

        builder.setPositiveButton("Да") { dialogInterface: DialogInterface, _: Int ->
            dialogInterface.dismiss()
            descriptionDialog()
        }

        builder.setNegativeButton("Нет") { dialogInterface: DialogInterface, _: Int ->
            dialogInterface.dismiss()
        }

        val dialog = builder.create()
        dialog.show()
    }

    private fun descriptionDialog() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Описание")
        builder.setMessage("Измеряйте на одной высоте с вами, один и тот же объект на расстоянии от камеры, ровно 5 метров, несколько секунд, пока калибровка не завершится.")

        builder.setPositiveButton("ОК") { dialogInterface: DialogInterface, _: Int ->
            dialogInterface.dismiss()
            CALIBRATING = true
        }

        val dialog = builder.create()
        dialog.show()
    }

    private fun enteredDialog() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Введите наиболее точную высоту объекта в сантиметрах:")

        val input = EditText(this)
        builder.setView(input)

        builder.setPositiveButton("ОК") { dialog: DialogInterface, _: Int ->
            val size = input.text.toString().toIntOrNull()
            if (size != null) {
                val location = calibrationObject?.gettingLocation()
                val irregularDist = (size.toFloat() / 100) * 1000 / (location!!.bottom - location!!.top)
                var difference = 5.0f
                var TH = 0.0f
                var bestTH = 0.0f
                while (TH <= 4.0f) {
                    val distance = irregularDist / TH
                    if (abs(distance - 5.0f) <= difference) {
                        difference = abs(distance - 5.0f)
                        bestTH = TH
                    }
                    TH += 0.01f
                }
                THOUSANDTH = bestTH
            } else {
                val location = calibrationObject?.gettingLocation()
                val labelName = calibrationObject?.gettingLabelName()
                var irregularDist = sizes[labelName] !!* 1000 / (location!!.bottom - location!!.top)
                if (labelName in widthLabels) {
                    irregularDist = sizes[labelName] !!* 1000 / (location!!.right - location!!.left)
                }
                var difference = 5.0f
                var TH = 0.0f
                var bestTH = 0.0f
                while (TH <= 4.0f) {
                    val distance = irregularDist / TH
                    if (abs(distance - 5.0f) <= difference) {
                        difference = abs(distance - 5.0f)
                        bestTH = TH
                    }
                    TH += 0.01f
                }
                THOUSANDTH = bestTH
            }
            dialog.dismiss()
        }

        val dialog = builder.create()
        dialog.show()
    }

    private fun translateYUV(image: Image, context: Context?): Bitmap {
        val crop = image.cropRect

        val width = crop.width()
        val height = crop.height()

        val planes = image.planes
        val rowData = ByteArray(planes[0].rowStride)
        val bufferSize = width * height * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8
        val yuvBytes = ByteBuffer.allocateDirect(bufferSize)

        var channelOffset = 0
        var outputStride = 0
        for (planeIndex in 0..2) {
            if (planeIndex == 0) {
                channelOffset = 0
                outputStride = 1
            } else if (planeIndex == 1) {
                channelOffset = width * height + 1
                outputStride = 2
            } else if (planeIndex == 2) {
                channelOffset = width * height
                outputStride = 2
            }

            val buffer = planes[planeIndex].buffer
            val rowStride = planes[planeIndex].rowStride
            val pixelStride = planes[planeIndex].pixelStride

            val shift = if (planeIndex == 0) 0 else 1
            val widthShifted = width shr shift
            val heightShifted = height shr shift

            buffer.position(rowStride * (crop.top shr shift) + pixelStride * (crop.left shr shift))
            for (row in 0 until heightShifted) {
                val length: Int
                if (pixelStride == 1 && outputStride == 1) {
                    length = widthShifted
                    buffer[yuvBytes.array(), channelOffset, length]
                    channelOffset += length
                } else {
                    length = (widthShifted - 1) * pixelStride + 1
                    buffer[rowData, 0, length]
                    for (col in 0 until widthShifted) {
                        yuvBytes.array()[channelOffset] = rowData[col * pixelStride]
                        channelOffset += outputStride
                    }
                }
                if (row < heightShifted - 1) {
                    buffer.position(buffer.position() + rowStride - length)
                }
            }
        }

        val rs = RenderScript.create(context)
        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)

        val allocationRgb = Allocation.createFromBitmap(rs, bitmap)
        val allocationYuv = Allocation.createSized(rs, Element.U8(rs), yuvBytes.array().size)
        val scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

        allocationYuv.copyFrom(yuvBytes.array())
        scriptYuvToRgb.setInput(allocationYuv)
        scriptYuvToRgb.forEach(allocationRgb)
        allocationRgb.copyTo(bitmap)

        allocationYuv.destroy()
        allocationRgb.destroy()
        rs.destroy()

        return bitmap
    }

    private fun initializeCamera() {
        val distanceField = findViewById<TextView>(R.id.distanceTextView)
        val labelField = findViewById<TextView>(R.id.labelTextView)

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture!!.addListener({
            val cameraProvider = cameraProviderFuture!!.get()

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1024, 768))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()
            imageAnalysis.setAnalyzer(
                ContextCompat.getMainExecutor(this@MainActivity)
            ) { image ->
                @androidx.camera.core.ExperimentalGetImage
                val img = image.image
                val bitmap = translateYUV(img!!, this@MainActivity)
                val matrix = Matrix().apply { postRotate(90f) }
                val bitmapR = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

                val recognitions: ArrayList<Recognition> = YOLOv5!!.detect(bitmapR!!)
                val mutableBitmap = bitmapR!!.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutableBitmap)
                var centerObject: Recognition? = null
                var nearestDistance = (bitmap.width * bitmap.height).toDouble()
                for (recognition in recognitions) {
                    val confidence = recognition.gettingConfidence()
                    val labelName = recognition.gettingLabelName()
                    if (confidence > 0.6 && labelName in sizes) {
                        val size = sizes[labelName]
                        val location = recognition.gettingLocation()
                        canvas.drawRect(location, boxPaint)
                        val distance = Math.sqrt(((bitmap.width / 2 - (location.left + location.right) / 2) * (bitmap.width / 2 - (location.left + location.right) / 2) + (bitmap.height / 2 - (location.top + location.bottom) / 2) * (bitmap.height / 2 - (location.top + location.bottom)).toDouble()))
                        if (distance <= nearestDistance) {
                            nearestDistance = distance
                            centerObject = recognition
                        }
                    }
                }

                val location = centerObject?.gettingLocation()
                val labelName = centerObject?.gettingLabelName()
                val confidence = centerObject?.gettingConfidence()

                if (location != null && labelName != null && confidence != null) {
                    canvas.drawRect(location, boxCenterPaint)
                    var irregularDist = sizes[labelName] !!* 1000 / (location!!.bottom - location!!.top)
                    if (labelName in widthLabels) {
                        irregularDist = sizes[labelName] !!* 1000 / (location!!.right - location!!.left)
                    }
                    distanceField.setText("${irregularDist / THOUSANDTH}")
                    labelField.setText(labelName + " : " + confidence)

                    if (CALIBRATING) {
                        calibrationObject = centerObject
                        CALIBRATING = false
                        enteredDialog()
                    }
                }

                preview!!.setImageBitmap(mutableBitmap)
                image.close()
            }
            cameraProvider.bindToLifecycle(this@MainActivity, cameraSelector, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }
}