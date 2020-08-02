package lib.com.sun.media.sound;

import lib.sound.midi.MidiUnavailableException;
import lib.sound.midi.Receiver;
import lib.sound.midi.Transmitter;



/** MidiDevice that can use reference counting for open/close.
 * This interface is intended to be used by MidiSystem.getTransmitter() and
 * MidiSystem.getReceiver().
 * 
 * @version %I%, %E%
 * @author Matthias Pfisterer
 */
public interface ReferenceCountingDevice {
    /** Retrieve a Receiver that opens the device implicitly.
     * This method is similar to MidiDevice.getReceiver(). However, by calling this one,
     * the device is opened implicitly. This is needed by MidiSystem.getReceiver().
     */
    public Receiver getReceiverReferenceCounting() throws MidiUnavailableException;

    /** Retrieve a Transmitter that opens the device implicitly.
     * This method is similar to MidiDevice.getTransmitter(). However, by calling this one,
     * the device is opened implicitly. This is needed by MidiSystem.getTransmitter().
     */
    public Transmitter getTransmitterReferenceCounting() throws MidiUnavailableException;
}