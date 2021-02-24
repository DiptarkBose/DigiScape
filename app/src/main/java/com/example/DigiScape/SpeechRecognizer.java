package com.example.DigiScape;

// Copyright 2019 Alpha Cephei Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


import static java.lang.String.format;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder.AudioSource;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.kaldi.KaldiRecognizer;
import org.kaldi.Model;
import org.kaldi.RecognitionListener;
import org.kaldi.SpkModel;

import com.example.DigiScape.MFCC;

/**
 * Main class to access recognizer functions. After configuration this class
 * starts a listener thread which records the data and recognizes it using
 * VOSK engine. Recognition events are passed to a client using
 * {@link RecognitionListener}
 *
 */
public class SpeechRecognizer {

    protected static final String TAG = SpeechRecognizer.class.getSimpleName();

    private final KaldiRecognizer recognizer;

    private final int sampleRate;
    private final static float BUFFER_SIZE_SECONDS = 0.4f;
    private int bufferSize;
    private final AudioRecord recorder;

    private Thread recognizerThread;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    //-------------------------------Custom Global Variables-------------------------------------
    public double maxAmplitude =0.0;
    public double[] mfccFeeder = new double[4096];
    public double[][] mfccFeatures = new double[15][20];
    public double mfccAvg = 0.0;
    public List<Integer> mfccAvgList = new ArrayList<>();
    public FFT fft = new FFT(FFT.FFT_NORMALIZED_POWER, 4096, FFT.WND_HANNING);
    public double avgSpecFlatness;
    public float rms = 0f;
    public double avgSpecCentroid = 0.0;
    public int numCrossing = 0;
    public float frequencyZCR = 0f;
    public double frequencySC = 0.0;
    public double spectralRolloff = 0.0;
    public double compactness = 0.0;
    public double variability = 0.0;
    //-------------------------------------------------------------------------------------------

    private final Collection<RecognitionListener> listeners = new HashSet<RecognitionListener>();


    public SpeechRecognizer(Model model) throws IOException {
        recognizer = new KaldiRecognizer(model, 16000.0f);
        sampleRate = 16000;
        bufferSize = Math.round(sampleRate * BUFFER_SIZE_SECONDS);
        recorder = new AudioRecord(
                AudioSource.VOICE_RECOGNITION, sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, bufferSize * 2);

        if (recorder.getState() == AudioRecord.STATE_UNINITIALIZED) {
            recorder.release();
            throw new IOException(
                    "Failed to initialize recorder. Microphone might be already in use.");
        }
    }

    public SpeechRecognizer(Model model, SpkModel spkModel) throws IOException {
        recognizer = new KaldiRecognizer(model, spkModel, 16000.0f);
        sampleRate = 16000;
        bufferSize = Math.round(sampleRate * BUFFER_SIZE_SECONDS);
        recorder = new AudioRecord(
                AudioSource.VOICE_RECOGNITION, sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, bufferSize * 2);

        if (recorder.getState() == AudioRecord.STATE_UNINITIALIZED) {
            recorder.release();
            throw new IOException(
                    "Failed to initialize recorder. Microphone might be already in use.");
        }
    }

    /**
     * Adds listener.
     */
    public void addListener(RecognitionListener listener) {
        synchronized (listeners) {
            listeners.add(listener);
        }
    }

    /**
     * Removes listener.
     */
    public void removeListener(RecognitionListener listener) {
        synchronized (listeners) {
            listeners.remove(listener);
        }
    }

    /**
     * Starts recognition. Does nothing if recognition is active.
     *
     * @return true if recognition was actually started
     */
    public boolean startListening() {
        if (null != recognizerThread)
            return false;

        recognizerThread = new RecognizerThread();
        recognizerThread.start();
        return true;
    }

    /**
     * Starts recognition. After specified timeout listening stops and the
     * endOfSpeech signals about that. Does nothing if recognition is active.
     *
     * @timeout - timeout in milliseconds to listen.
     *
     * @return true if recognition was actually started
     */
    public boolean startListening(int timeout) {
        if (null != recognizerThread)
            return false;

        recognizerThread = new RecognizerThread(timeout);
        recognizerThread.start();
        return true;
    }

    private boolean stopRecognizerThread() {
        if (null == recognizerThread)
            return false;

        try {
            recognizerThread.interrupt();
            recognizerThread.join();
        } catch (InterruptedException e) {
            // Restore the interrupted status.
            Thread.currentThread().interrupt();
        }

        recognizerThread = null;
        return true;
    }

    /**
     * Stops recognition. All listeners should receive final result if there is
     * any. Does nothing if recognition is not active.
     *
     * @return true if recognition was actually stopped
     */
    public boolean stop() {
        boolean result = stopRecognizerThread();
        if (result) {
            mainHandler.post(new ResultEvent(recognizer.Result(), true));
        }
        return result;
    }

    /**
     * Cancels recognition. Listeners do not receive final result. Does nothing
     * if recognition is not active.
     *
     * @return true if recognition was actually canceled
     */
    public boolean cancel() {
        boolean result = stopRecognizerThread();
        recognizer.Result(); // Reset recognizer state
        return result;
    }

    /**
     * Shutdown the recognizer and release the recorder
     */
    public void shutdown() {
        recorder.release();
    }

    private final class RecognizerThread extends Thread {

        private int remainingSamples;
        private int timeoutSamples;
        private final static int NO_TIMEOUT = -1;

        public RecognizerThread(int timeout) {
            if (timeout != NO_TIMEOUT)
                this.timeoutSamples = timeout * sampleRate / 1000;
            else
                this.timeoutSamples = NO_TIMEOUT;
            this.remainingSamples = this.timeoutSamples;
        }

        public RecognizerThread() {
            this(NO_TIMEOUT);
        }

        @Override
        public void run() {

            recorder.startRecording();
            if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_STOPPED) {
                recorder.stop();
                IOException ioe = new IOException(
                        "Failed to start recording. Microphone might be already in use.");
                mainHandler.post(new OnErrorEvent(ioe));
                return;
            }

            short[] buffer = new short[bufferSize];

            while (!interrupted()
                    && ((timeoutSamples == NO_TIMEOUT) || (remainingSamples > 0))) {
                int nread = recorder.read(buffer, 0, buffer.length);

                /*
                    START of Custom Calculations
                    Lines 250 - 375
                */
                maxAmplitude = 0; numCrossing=0;
                double[] curFrame = new double[4096];
                double[] im = new double[4096];
                double total = 0.0;
                double average = 0.0;
                for(int i=0; i<4096; i++) {

                    //Zero Crossing Calculation
                    if ((buffer[i] > 0 && buffer[i+1] <= 0) || (buffer[i] < 0 && buffer[i+1] >= 0))
                        numCrossing++;

                    //RMS Intermediate Calculation
                    rms += (buffer[i] * buffer[i]);

                    //Max Amplitude Calculation
                    if (Math.abs(buffer[i]) >= maxAmplitude)
                        maxAmplitude = Math.abs(buffer[i]);

                    //Creating Feeders for FFT and MFCC
                    curFrame[i] = buffer[i];
                    mfccFeeder[i] = buffer[i];
                }

                //Strongest Frequency by ZCR Calculation
                float numSecondsRecorded = (float)4096 / (float)sampleRate;
                float numCycles = numCrossing / 2;
                frequencyZCR = numCycles / numSecondsRecorded;
                frequencyZCR = (float)Math.round(frequencyZCR * 1000) / 1000;

                //RMS Calculation
                rms = (float)Math.sqrt(rms / 4096);
                rms = (float)Math.round(rms * 10000) / 10000;

                //FFT Calculation
                fft.transform(curFrame, im);

                //MFCC Calculation
                MFCC mfcc = new MFCC(sampleRate);
                try {
                    mfccFeatures = mfcc.process(mfccFeeder);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                double[] power_spectrum = new double[2048];
                double[] magnitude_spectrum = new double[2048];
                double geometricMean = 0;
                double arithmeticMean = 0;

                for(int i = 0; i < 2048; i++)
                {
                    //Power Spectrum Calculation
                    power_spectrum[i] = (Math.pow(curFrame[i], 2) + Math.pow(im[i], 2)) / 4096;

                    //Magnitude Spectrum Calculation
                    magnitude_spectrum[i] = Math.sqrt(Math.pow(curFrame[i], 2) + Math.pow(im[i], 2)) / 4096;

                    //Spectral Variability Intermediate Calculation
                    total += magnitude_spectrum[i];

                    //Spectral Flatness Intermediate Calculation
                    geometricMean += Math.log(power_spectrum[i]);
                    arithmeticMean += power_spectrum[i];
                }

                //Spectral Flatness Calculation
                geometricMean = geometricMean / 2048;
                geometricMean = Math.exp(geometricMean);
                arithmeticMean = arithmeticMean/2048;
                avgSpecFlatness = geometricMean / arithmeticMean;
                avgSpecFlatness = (double)Math.round(avgSpecFlatness * 10000) / 10000;

                //Spectral Variability, Centroid, and Compactness Calculation
                average = (total / 2048);
                double sum = 0.0;
                total = 0.0;
                double weighted_total = 0.0;
                compactness = 0.0;
                for (int i = 0; i < 2048; i++)
                {
                    //Spectral Variability Intermediate Calculation
                    double diff = magnitude_spectrum[i] - average;
                    sum = sum + (diff * diff);
                    total += power_spectrum[i];
                    weighted_total += i * power_spectrum[i];

                    //Compactness Calculation
                    if(i>0 && i<2047)
                        if ((magnitude_spectrum[i - 1] > 0.0) && (magnitude_spectrum[i] > 0.0) && (magnitude_spectrum[i + 1] > 0.0))
                            compactness += Math.abs(20.0 * Math.log(magnitude_spectrum[i]) - 20.0 * (Math.log(magnitude_spectrum[i - 1]) + Math.log(magnitude_spectrum[i]) + Math.log(magnitude_spectrum[i + 1])) / 3.0);
                }
                //Spectral Variability Calculation
                variability = Math.sqrt(sum / ((double) (2047)));
                variability = (double) Math.round(variability * 1000) / 1000;

                //Compactness Calculation
                compactness = (double)Math.round(compactness * 1000) / 1000;

                //Spectral Centroid Calculation
                avgSpecCentroid = weighted_total / total;
                avgSpecCentroid=(double)Math.round(avgSpecCentroid * 1000) / 1000;

                // Strongest Frequency by Spectral Centroid Calculation
                frequencySC=avgSpecCentroid*3.9;
                frequencySC=(double)Math.round(frequencySC * 1000) / 1000;


                //Spectral Rolloff Calculations
                double cutoff = 0.85;
                double threshold = total * cutoff;
                total = 0.0;
                int point = 0;
                for (int i = 0; i < 2048; i++) {

                    //Spectral Rolloff Intermediate Calculation
                    total += power_spectrum[i];
                    if (total >= threshold) {
                        point = i;
                        i = 2048;
                    }
                }

                //Spectral Rolloff Calculation
                spectralRolloff = ((double) point) / 2048;
                spectralRolloff = (double)Math.round(spectralRolloff * 1000) / 1000;

                /*
                    END of Custom Calculations
                */


                if (nread < 0) {
                    throw new RuntimeException("error reading audio buffer");
                } else {
                    boolean isFinal = recognizer.AcceptWaveform(buffer, nread);
                    if (isFinal) {
                        mainHandler.post(new ResultEvent(recognizer.Result(), true));
                    } else {
                        mainHandler.post(new ResultEvent(recognizer.PartialResult(), false));
                    }
                }

                if (timeoutSamples != NO_TIMEOUT) {
                    remainingSamples = remainingSamples - nread;
                }
            }

            recorder.stop();

            // Remove all pending notifications.
            mainHandler.removeCallbacksAndMessages(null);

            // If we met timeout signal that speech ended
            if (timeoutSamples != NO_TIMEOUT && remainingSamples <= 0) {
                mainHandler.post(new TimeoutEvent());
            }
        }
    }

    private abstract class RecognitionEvent implements Runnable {
        public void run() {
            RecognitionListener[] emptyArray = new RecognitionListener[0];
            for (RecognitionListener listener : listeners.toArray(emptyArray))
                execute(listener);
        }

        protected abstract void execute(RecognitionListener listener);
    }

    private class ResultEvent extends RecognitionEvent {
        protected final String hypothesis;
        private final boolean finalResult;

        ResultEvent(String hypothesis, boolean finalResult) {

            //----------------------Adding results to the hypothesis---------------------------

            //Max Amplitude
            hypothesis += ("\nMax Amplitude: "+maxAmplitude+"\n");

            //Amplitude RMS
            hypothesis+="Amplitude RMS: " +rms+" \n";

            //MFCC
            mfccAvg = 0.0;
            for(int i=0; i<15; i++)
                for (int j = 1; j <= 12; j++)
                    mfccAvg += mfccFeatures[i][j];

            mfccAvg = mfccAvg / 84;
            mfccAvg = (double)Math.round(mfccAvg * 1000) / 1000;
            hypothesis += ("MFCC Average: "+mfccAvg+"\n");
            mfccAvgList.add((int)mfccAvg);

            //Spectral Flatness
            hypothesis+="Spectral Flatness: " +avgSpecFlatness+" \n";

            //Spectral Centroid
            hypothesis+="Spectral Centroid: " +avgSpecCentroid+" \n";

            //Strongest Frequency by Spectral Centroid
            hypothesis+="Strongest Frequency (SC): " +frequencySC+" Hz\n";

            //ZCR
            hypothesis+="Zero-Crossings: " +numCrossing+" \n";

            //Strongest Frequency by ZCR
            hypothesis+="Strongest Frequency (ZCR): " +frequencyZCR+" Hz\n";

            //Spectral Rolloff Point
            hypothesis+="Spectral Rolloff Point: " +spectralRolloff+"\n";

            //Compactness
            hypothesis+="Compactness: " +compactness+"\n";

            //Spectral Variability
            hypothesis+="Spectral Variability: " +variability+"\n";

            //----------------------------------------------------------------------------------
            this.hypothesis = hypothesis;
            this.finalResult = finalResult;
        }

        @Override
        protected void execute(RecognitionListener listener) {
            if (finalResult)
                listener.onResult(hypothesis);
            else
                listener.onPartialResult(hypothesis);
        }
    }

    private class OnErrorEvent extends RecognitionEvent {
        private final Exception exception;

        OnErrorEvent(Exception exception) {
            this.exception = exception;
        }

        @Override
        protected void execute(RecognitionListener listener) {
            listener.onError(exception);
        }
    }

    private class TimeoutEvent extends RecognitionEvent {
        @Override
        protected void execute(RecognitionListener listener) {
            listener.onTimeout();
        }
    }
}
