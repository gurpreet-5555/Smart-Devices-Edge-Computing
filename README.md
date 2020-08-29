# Smart-Devices (Edge Computing with NVIDIA Jetson Nano)
Solution that enables IoT supported devices to "see" human activity and turn on/off automatically. <br>Computer Vision &amp; Object Detection is used to detect human presence and trigger signal to IoT device for automatic operation.
<br><br>
This project utilizes performs human detection at the edge without any need to send the video feed to cloud for inference.<br>
Human Detection is performed at the edge using <b>Nvidia Jetson Nano SBC (Single-Board Computer) at frame rates of 20-25 FPS</b>.<br>

<img src="images/Jetson_Nano.jpg" width="600" height="300" />

<br>

<br>
In this project, we'll be utilizing 'ssd-mobilenet-v2' object detection architecture specially optimized for Nvidia GPUs (Graphic Processing Units) known as <b>TensorRT</b>.<br>
TensorRT is a high-performance neural network inference optimizer and runtime engine that optimizes a network by combining layers for improved performance, low memory consumption and faster inference.

<br>
You can explore more object detection models through below link:

<a href="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html">NVIDIA TensorRT Documentation</a>

<br> <br>

<h2>Features:</h2>
<ul>
  <li>Control an IoT enabled device automatically by detecting human activity</li>
  <li>Perform object detection on Edge Device @ 20-25 FPS</li>
  <li>Real-time person detection</li>
  <li>Specify confidence threshold for detection to compensate for poor lighting/image quality</li>
  <li>Countdown Timer for starting & stopping a device</li>
  <li>Support for multiple streaming protocols from IP cameras such as rtsp, http. Can also be used with static video file depending on installed codecs(.mp4, .avi etc..)</li>
</ul>  

<h2>Requirements:</h2>
<ul>
  <li>Python 3.6</li>
  <li>OpenCV 4.2.0</li>
  <li>Tensorflow 1.15</li>
  <li>NumPy 1.18.2</li>  
</ul>

<h2>How to use?</h2>
<ul>
  <li>Clone Repository</li>
  <pre>https://github.com/gurpreet-5555/Intelligent-Devices.git</pre>  </ul>
<ul>  <li>Navigate to main directory</li>
  <pre>cd Intelligent-Devices/main</pre> </ul>
<ul><li>Specify your code in <b>device_controller.py</b> to start and stop IoT powered device.</li>
<pre>Sample code is included to turn on/off Philips Hue Lamps</pre></ul>
<ul><li>Execute Program</li>
<pre>python start_controller.py --confidence 0.4 --stream http://192.168.1.43:8080/video --startthreshold 10 --stopthreshold 60</pre>
<pre>Arguments -
confidence : Confidence threshold for person detection. Default value is 0.2 (Optional)
stream : Source of video feed (http, rtsp etc) or video file. (Required)
startthreshold: Time to wait in seconds before device starts once human activity is detected. Default value is 5 seconds. (Optional)
stopthreshold: Time to wait in seconds before device stops once no human activity is detected in video stream. Default value is 30 seconds. (Optional)
</pre></ul>
