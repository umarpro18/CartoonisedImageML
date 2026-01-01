package com.example.customtflitemodelimplsample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.createBitmap
import androidx.core.graphics.get
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


/**
 * Convert given image to cartoon using ML model (tflite file)
 *
 * Steps: converting bitmap into bytebuffer
 *  → Resize image
 *  → Extract its pixels
 *  → Normalize (match training scale usually -1.0 to 1.0 or 0.0 to 1.0)
 *  → Pack into ByteBuffer (store it in contiguous memory)
 *  → Inference
 *  → Output
 *
 * Steps: Converting bytebuffer output to bitmap
 *  → Extract its pixels (already denormalized)
 *  → Pack into Bitmap
 */
class MainActivity : AppCompatActivity() {

    private var interpreter: Interpreter? = null

    // Use Executor for threads
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Create interpreter and load model
        createInterpreter()

        // Load bitmap from assets
        val bitmap = loadBitmapFromAsset()
        findViewById<ImageView>(R.id.originalImage).setImageBitmap(bitmap)

        executor.execute {
            // Preprocessing: Converting bitmap into bytebuffer
            val t0 = SystemClock.elapsedRealtime()
            val byteBuffer = convertBitmapToByteBuffer(bitmap, 224, 224)
            val t1 = SystemClock.elapsedRealtime()
            Log.d("MainActivity", "Preprocessing time = ${t1 - t0}")

            // Run inference
            val inferenceResult = runInference(byteBuffer)
            val t2 = SystemClock.elapsedRealtime()
            Log.d("MainActivity", "Inference time = ${t2 - t1}")

            // Postprocessing: Convert output byte buffer array to image
            val image = convertOutputArrayToImage(inferenceResult)
            val t3 = SystemClock.elapsedRealtime()
            Log.d("MainActivity", "Postprocessing time = ${t3 - t2}")

            runOnUiThread {
                displayResultImage(image)
            }
        }
    }

    private fun displayResultImage(finalBitmap: Bitmap) {
        findViewById<ImageView>(R.id.cartoonImage).setImageBitmap(finalBitmap)
    }

    private fun convertOutputArrayToImage(inferenceResult: Array<Array<Array<FloatArray>>>): Bitmap {
        val output = inferenceResult[0]
        val bitmap = createBitmap(224, 224)
        val pixels = IntArray(224 * 224)

        var index = 0

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val b = (output[y][x][0] + 1) * 127.5
                val r = (output[y][x][1] + 1) * 127.5
                val g = (output[y][x][2] + 1) * 127.5

                val a = 0xFF
                pixels[index] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
                index++
            }
        }
        bitmap.setPixels(pixels, 0, 224, 0, 0, 224, 224)
        return bitmap
    }

    private fun createInterpreter() {
        // CPU delegate by default
        val tfLiteOptions = Interpreter.Options()
        interpreter = getInterpreter(this, TFLITE_MODEL_NAME, tfLiteOptions)

        // GPU delegate
        /*val tfLiteOptions = Interpreter.Options()
        val gpuDelegate = GpuDelegate()
        tfLiteOptions.addDelegate(gpuDelegate)*/

        interpreter = getInterpreter(this, TFLITE_MODEL_NAME, tfLiteOptions)
    }

    private fun loadBitmapFromAsset(): Bitmap {
        val inputStream: InputStream = applicationContext.assets.open("test-image.jpeg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        return bitmap.scale(224, 224, false)
    }

    private fun runInference(byteBuffer: ByteBuffer): Array<Array<Array<FloatArray>>> {
        val outputArr = Array(1) {
            Array(224) {
                Array(224) {
                    FloatArray(3)
                }
            }
        }
        interpreter?.run(byteBuffer, outputArr)
        return outputArr
    }

    private fun getInputImage(width: Int, height: Int): ByteBuffer {
        val inputImage =
            ByteBuffer.allocateDirect(1 * width * height * 3 * 4)// input image will be required input shape of tflite model
        inputImage.order(ByteOrder.nativeOrder())
        inputImage.rewind()
        return inputImage
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        // these value can be different for each channel if they are not then you may have single value instead of an array
        val mean = arrayOf(127.5f, 127.5f, 127.5f)
        val standard = arrayOf(127.5f, 127.5f, 127.5f)

        val inputImage = getInputImage(width, height)
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        for (y in 0 until width) {
            for (x in 0 until height) {
                val px = bitmap[x, y]

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = (r - mean[0]) / standard[0]
                val gf = (g - mean[0]) / standard[0]
                val bf = (b - mean[0]) / standard[0]

                inputImage.putFloat(bf)
                inputImage.putFloat(rf)
                inputImage.putFloat(gf)
            }
        }
        return inputImage
    }

    private fun getInterpreter(
        context: Context,
        modelName: String,
        tfLiteOptions: Interpreter.Options
    ): Interpreter {
        return Interpreter(FileUtil.loadMappedFile(context, modelName), tfLiteOptions)
    }

    companion object {
        const val TFLITE_MODEL_NAME = "lite_model_cartoongan.tflite"
    }
}

/**
 * Improvements to consider:
 *
 * High-impact fixes (do these first)
 * 	1.	Replace bitmap[x, y] with getPixels()
 * 	2.	Reuse ByteBuffer, IntArray, output arrays
 * 	3.	Move inference off the main thread
 * 	4.	Fix output normalization correctness
 *
 * Medium-impact improvements
 * 	•	Avoid creating new Bitmap every inference if possible
 * 	•	Pre-scale image once
 * 	•	Use setPixels() exactly as you already do (good)
 */

/**
 * Performance measurements:
 * Step 1: Identifying the parts and segregating
 *  A. Preprocessing  (Bitmap → ByteBuffer)
 *  B. Inference      (interpreter.run)
 *  C. Postprocessing (Output → Bitmap)
 *
 * Step 2: Basic time logging between the above major steps
 * Result:
 *  Preprocessing ≈ 22 ms
 *  Inference     ≈ 225 ms
 *  Postprocessing ≈ 2 ms
 *  Total: 250 ms approx.
 * Conclusion:
 *  We conclude that the inference takes up 90% of the time
 *  We see frames are skipped due to all three steps are blocking ui thread (250ms lost), so it skipped 16 frames approx.
 *  -> 16.6 ms per frame android usually draws (60 fps common)
 *  Now we confirm the above inference time to correlate CPU profiler time.
 *      Interpreter.run() jumps into native C++
 * 	    CPU Profiler (Java/Kotlin) can’t see inside ML kernels
 * 	    It shows up as one big native block (screen shots uploaded)
 * 	    Result: (It correlates with manual timing with cpu 260ms approx.)
 * 	    Why you could not able to see ML stuffs at cpu profiler
 * 	    	•	Interpreter.run() jumps into native C++
 * 	        •	CPU Profiler (Java/Kotlin) can’t see inside ML kernels
 * 	        •	It shows up as one big native block
 * 	    Next steps:
 * 	        Off load inference running to bg thread (AsyncTask)
 * 	        Try GPu delegates
 *
 *  Step 3: Off load inference running to bg thread (AsyncTask)
 *   Time remained same but we actually off loaded UI thread stuffs to make the UI smoother
 *
 *  Step 4: Try GPU delegates - This model does not support GPU delegate basically due to the operators and graphs in the model
 *
 *  Step 5: Lets explore NNAPI delegate to use NPU possibly
 *
 *
 */

/**
 * MobileNet metadata:
 * Input
 * 1. [1, 224, 224, 3]
 * 2. Type: Float32
 * 3. Normalization [0.0 - 1.0] (divide by 255.0) since no operator present
 *
 * Output
 * 1. [1 - 1001] class score
 */