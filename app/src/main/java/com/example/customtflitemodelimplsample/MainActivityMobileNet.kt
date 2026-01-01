package com.example.customtflitemodelimplsample

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivityMobileNet : AppCompatActivity() {

    // Intrepreter from tflite runtime
    private var interpreter: Interpreter? = null

    // Use Executor for threads
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_mobile_net)

        createInterpreter()

        val bitmap = loadBitmapFromAsset()
        findViewById<ImageView>(R.id.originalImage).setImageBitmap(bitmap)

        executor.execute {
            val t0 = SystemClock.elapsedRealtime()
            val inputBuffer = convertBitmapToByteBuffer(bitmap, 224, 224)
            val t1 = SystemClock.elapsedRealtime()
            Log.d("MobileNet", "Preprocessing = ${t1 - t0} ms")

            val output = runInference(inputBuffer)
            val t2 = SystemClock.elapsedRealtime()
            Log.d("MobileNet", "Inference = ${t2 - t1} ms")

            val topIndex = getTopPrediction(output)
            val t3 = SystemClock.elapsedRealtime()
            Log.d("MobileNet", "Postprocessing = ${t3 - t2} ms")

            runOnUiThread {
                findViewById<TextView>(R.id.resultText)
                    .text = "Predicted class index: $topIndex"
            }
        }
    }

    private fun loadBitmapFromAsset(): Bitmap {
        val inputStream: InputStream = applicationContext.assets.open("test-image.jpeg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        return bitmap.scale(224, 224, false)
    }

    private fun runInference(inputBuffer: ByteBuffer): Array<FloatArray> {
        val output = Array(1) { FloatArray(1001) }
        interpreter?.run(inputBuffer, output)
        return output
    }

    private fun getTopPrediction(output: Array<FloatArray>): Int {
        val scores = output[0]
        var maxIndex = 0
        var maxScore = scores[0]

        for (i in 1 until scores.size) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxIndex = i
            }
        }
        return maxIndex
    }

    private fun convertBitmapToByteBuffer(
        bitmap: Bitmap,
        width: Int,
        height: Int
    ): ByteBuffer {

        val buffer =
            ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var index = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val px = pixels[index++]

                val r = Color.red(px) / 255.0f
                val g = Color.green(px) / 255.0f
                val b = Color.blue(px) / 255.0f

                buffer.putFloat(r)
                buffer.putFloat(g)
                buffer.putFloat(b)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun createInterpreter() {
        val options = Interpreter.Options()
        interpreter = Interpreter(
            FileUtil.loadMappedFile(this, TFLITE_MODEL_MOBILENET),
            options
        )
    }

    companion object {
        const val TFLITE_MODEL_MOBILENET = "mobilenet_v2.tflite"
    }

}