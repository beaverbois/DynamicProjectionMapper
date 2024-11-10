import cv2
from consts import Consts
import pyfreenect2
import signal
import numpy as np

class Camera:
    def __init__(self):
        # create camera
        # self.cam = cv2.VideoCapture(Consts.CAMERA_INDEX)

        # set resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, Consts.CAMERA_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Consts.CAMERA_HEIGHT)

    def getFrame(self):
        _, frame = self.cam.read()
        return frame

class Kinect:
    def __init__(self):
        # This is pretty much a straight port of the Protonect program bundled with
        # libfreenect2.

        # Initialize device
        serialNumber = pyfreenect2.getDefaultDeviceSerialNumber()
        kinect = pyfreenect2.Freenect2Device(serialNumber)

        # Set up signal handler
        self.terminated = False

        def sigint_handler(signum, frame):
            print("Got SIGINT, shutting down...")
            self.terminated = True
            kinect.stop()

        signal.signal(signal.SIGINT, sigint_handler)

        # Set up frame listener
        self.frameListener = pyfreenect2.SyncMultiFrameListener(pyfreenect2.Frame.COLOR,
                                                        pyfreenect2.Frame.IR,
                                                        pyfreenect2.Frame.DEPTH)

        print(self.frameListener)
        kinect.setColorFrameListener(self.frameListener)
        kinect.setIrAndDepthFrameListener(self.frameListener)

        # Start recording
        kinect.start()

        # Print useful info
        print("Kinect serial: %s" % kinect.serial_number)
        print("Kinect firmware: %s" % kinect.firmware_version)


        self.registration = pyfreenect2.Registration(kinect)

        # # Initialize OpenCV stuff
        # cv2.namedWindow("RGB")
        # cv2.namedWindow("Depth")

    def getFrame(self):
        if self.terminated:
            return
        frames = self.frameListener.waitForNewFrame()
        rgbFrame = frames.getFrame(pyfreenect2.Frame.COLOR)
        depthFrame = frames.getFrame(pyfreenect2.Frame.DEPTH)
        (undistorted, registered, big) = self.registration.apply(rgbFrame=rgbFrame, depthFrame=depthFrame)

        depth_frame = big.getDepthData()
        print(depth_frame)
        rgb_frame = cv2.flip(np.array(rgbFrame.getRGBData()[:, :, :3], dtype=np.uint16), 1)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        self.frameListener.release(frames)

        return rgb_frame, depth_frame

    
if __name__ == "__main__":
    kinect = Kinect()
    for i in range(1000):
        kinect.getFrame()
