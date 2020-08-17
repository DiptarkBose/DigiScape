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

    //---------------------------------Custom variables-----------------------------------------
    public double maxAmplitude;                                        //Variable for dealing with Amplitude Measurements
    public double[] mfccFeeder = new double[1024];                     //Variable for dealing with MFCC Measurements
    public double[][] mfccFeatures = new double[3][20];                //Variable for dealing with MFCC Measurements
    public double mfccAvg;
    public List<Integer> mfccAvgList = new ArrayList<>();
    public FFT fft = new FFT(FFT.FFT_NORMALIZED_POWER, 1024, FFT.WND_HANNING);
    public double avgSpecFlatness;
    public float rms=0f;
    public double avgSpecCentroid;
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

                //--------------Custom Code Snippet for Calculating Amplitude & RMS---------------
                maxAmplitude = 0;
                int i;
                double[] curFrame = new double[1024];
                for(i=0; i<buffer.length; i++) {
                    rms+=(buffer[i]*buffer[i]);
                    if (Math.abs(buffer[i]) >= maxAmplitude)
                        maxAmplitude = Math.abs(buffer[i]);
                    if(i<1024)
                        curFrame[i]=buffer[i];
                }
                rms = (float)Math.sqrt(rms / buffer.length);
                rms=(float)Math.round(rms*10000)/10000;
                //-------------------------------------------------------------------------------
                //--------------Custom Code Snippet for Calculating MFCC-------------------------
                MFCC mfcc = new MFCC(sampleRate);
                for(i=0; i<1024; i++)
                    mfccFeeder[i]=buffer[i];
                try {
                    mfccFeatures = mfcc.process(mfccFeeder);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                //-------------------------------------------------------------------------------
                //--------------Custom Code Snippet for Calculating Spectral Flatness------------
                //Provides a measure of the peakiness of the average spectrum.
                fft.transform(curFrame, null);
                double geometricMean = 1;
                double arithmeticMean = 1;
                for(int band = 0; band < 1024; band++)
                {
                    geometricMean += Math.log(curFrame[band])/1024;
                    arithmeticMean+=curFrame[band]/1024;
                }
                avgSpecFlatness = Math.exp(geometricMean)/arithmeticMean;
                avgSpecFlatness=(double)Math.round(avgSpecFlatness*10000)/10000;
                //-------------------------------------------------------------------------------
                //--------------Custom Code Snippet for Calculating Spectral Centroid------------
                //Provides a measure of the average spectral center of mass of a chunk's frames. Taken from https://www.ee.columbia.edu/~ronw/code/MEAPsoft/doc/html/AvgSpecCentroid_8java-source.html
                double num = 0;
                double den = 0;
                for(int band = 0; band < 1024; band++)
                {
                    double freqCenter = band*(8000)/(1023);
                    //Convert back to linear power
                    double p = Math.pow(10,curFrame[band]/10);
                    num += freqCenter*p;
                    den += p;
                }
                avgSpecCentroid += num/den;
                avgSpecCentroid=(double)Math.round(avgSpecCentroid*10000)/10000;
                //-------------------------------------------------------------------------------
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
            mfccAvg=0;
            hypothesis+=("\nAmplitude: "+maxAmplitude+"\n");
            hypothesis+="\nMFCC Features: \n";
            for(int i=0; i<mfccFeatures.length; i++) {
                hypothesis+="[";
                for (int j = 1; j <= 12; j++) {
                    hypothesis += ((double)Math.round(mfccFeatures[i][j]*100)/100) + " ";
                    mfccAvg += mfccFeatures[i][j];
                }
                hypothesis+="]\n";
            }
            mfccAvg=mfccAvg/36;
            mfccAvg = (double)Math.round(mfccAvg*100)/100;
            hypothesis+=("MFCC Average: "+mfccAvg+"\n");
            mfccAvgList.add((int)mfccAvg);
            hypothesis+="Spectral Flatness: " +avgSpecFlatness+" \n";
            hypothesis+="Spectral Centroid: " +avgSpecCentroid+" \n";
            hypothesis+="Amplitude RMS: " +rms+" \n";
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
