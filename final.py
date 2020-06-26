# USAGE
# python final.py -v cv.mp4
# python final.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
import math

#获取滤镜颜色
def getBGR(table, b, g, r):
    #计算标准颜色表中颜色的位置坐标
    x = int(g/4 + int(b/32) * 63)
    y = int(r/4 + int((b%32) / 4) * 63)
    #返回滤镜颜色表中对应的颜色
    return table[x][y]

compressWidth = 600
windowName = 'Beauty'
trackbar1 = 'Beauty'
trackbar2 = 'Filter'
cv.namedWindow(windowName)
cv.createTrackbar(trackbar1, windowName, 6, 10, lambda x: None)
cv.createTrackbar(trackbar2, windowName, 0, 20, lambda x: None)
lj1 = cv.imread('./filter/lj1.png')
lj2 = cv.imread('./filter/lj2.png')
lj3 = cv.imread('./filter/lj3.png')
lj4 = cv.imread('./filter/lj4.png')
lj5 = cv.imread('./filter/lj5.png')
lj6 = cv.imread('./filter/lj6.png')
lj7 = cv.imread('./filter/lj7.png')
lj8 = cv.imread('./filter/lj8.png')
lj9 = cv.imread('./filter/lj9.png')
lj10 = cv.imread('./filter/lj10.png')
lj11 = cv.imread('./filter/lj11.png')
lj12 = cv.imread('./filter/lj12.png')
lj13 = cv.imread('./filter/lj13.png')
lj14 = cv.imread('./filter/lj14.png')
lj15 = cv.imread('./filter/lj15.png')
lj16 = cv.imread('./filter/lj16.png')
lj17 = cv.imread('./filter/lj17.png')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
faceLower = (0, 50, 145)
faceUpper = (180, 115, 230)
pts = deque(maxlen=args["buffer"])

fps = 10
size = (compressWidth, int(compressWidth / 16 * 9) - 1)
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv.VideoCapture(args["video"])
	fps = vs.get(cv.CAP_PROP_FPS)
	size = (int(vs.get(cv.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv.CAP_PROP_FRAME_HEIGHT)))
	size = (compressWidth, int(vs.get(cv.CAP_PROP_FRAME_HEIGHT)/vs.get(cv.CAP_PROP_FRAME_WIDTH)*compressWidth))
	# print(size)

# set the video writer
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv.VideoWriter('oops.avi', fourcc, fps, size)

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		print('?')
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=compressWidth)
	ori = frame
	blurred = cv.GaussianBlur(frame, (11, 11), 0)
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

	# construct a mask for the face color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv.inRange(hsv, faceLower, faceUpper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)
	# cv.imshow("Mask", mask)
	# cv.imshow("ReMask", cv.bitwise_not(mask))

	value1 = cv.getTrackbarPos(trackbar1, windowName)
	value2 = 1
	dx = int(value1)
	fc = value1 * 2.5

	temp1 = cv.bilateralFilter(frame, dx, fc, fc)
	temp2 = temp1 - frame + 128
	temp3 = cv.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)
	temp4 = frame + 2 * temp3 - 255
	dst = frame + 2 * cv.GaussianBlur((cv.bilateralFilter(frame, dx, fc, fc) - frame + 128), (2 * value2 - 1, 2 * value2 - 1), 0, 0) - 255

	face = cv.add(dst, np.zeros(np.shape(frame), dtype=np.uint8), mask=mask)
	frame = cv.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=cv.bitwise_not(mask))
	# cv.imshow("Face", face)
	# cv.imshow("FrameFace", frame)
	frame = cv.add(face, frame)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
						   cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		# if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			# cv.circle(frame, (int(x), int(y)), int(radius),
			# 		  (0, 255, 255), 2)

			# cv.circle(frame, center, 5, (0, 0, 255), -1)

	# print(frame[:,:,0].shape)

	# cv.imshow("Smooth", frame)

	for h in range(0, frame.shape[0]):
		for w in range(0, frame.shape[1]):
			b = int(frame[h, w, 0])
			g = int(frame[h, w, 1])
			r = int(frame[h, w, 2])
			i = cv.getTrackbarPos(trackbar2, windowName)
			if i == 1:
				frame[h, w, 2] = int(0.393 * r + 0.769 * g + 0.189 * b) if int(0.393 * r + 0.769 * g + 0.189 * b) < 255 else 255
				frame[h, w, 1] = int(0.349 * r + 0.686 * g + 0.168 * b) if int(0.349 * r + 0.686 * g + 0.168 * b) < 255 else 255
				frame[h, w, 0] = int(0.272 * r + 0.534 * g + 0.131 * b) if int(0.272 * r + 0.534 * g + 0.131 * b) < 255 else 255
			elif i == 2:
				frame[h, w, 2] = int(abs(g - b + g + r) * r / 256) if int(abs(g - b + g + r) * r / 256) < 255 else 255
				frame[h, w, 1] = int(abs(b - g + b + r) * r / 256) if int(abs(b - g + b + r) * r / 256) < 255 else 255
				frame[h, w, 0] = int(abs(b - g + b + r) * g / 256) if int(abs(b - g + b + r) * g / 256) < 255 else 255
			elif i == 3:
				frame[h, w, 2] = r
				frame[h, w, 1] = g
				frame[h, w, 0] = math.sqrt(b) * 12 if math.sqrt(b) * 12 < 255 else 255
			elif i == 4:
				frame[h, w] = getBGR(lj1, b, g, r)
			elif i == 5:
				frame[h, w] = getBGR(lj2, b, g, r)
			elif i == 6:
				frame[h, w] = getBGR(lj3, b, g, r)
			elif i == 7:
				frame[h, w] = getBGR(lj4, b, g, r)
			elif i == 8:
				frame[h, w] = getBGR(lj5, b, g, r)
			elif i == 9:
				frame[h, w] = getBGR(lj6, b, g, r)
			elif i == 10:
				frame[h, w] = getBGR(lj7, b, g, r)
			elif i == 11:
				frame[h, w] = getBGR(lj8, b, g, r)
			elif i == 12:
				frame[h, w] = getBGR(lj9, b, g, r)
			elif i == 13:
				frame[h, w] = getBGR(lj10, b, g, r)
			elif i == 14:
				frame[h, w] = getBGR(lj11, b, g, r)
			elif i == 15:
				frame[h, w] = getBGR(lj12, b, g, r)
			elif i == 16:
				frame[h, w] = getBGR(lj13, b, g, r)
			elif i == 17:
				frame[h, w] = getBGR(lj14, b, g, r)
			elif i == 18:
				frame[h, w] = getBGR(lj15, b, g, r)
			elif i == 19:
				frame[h, w] = getBGR(lj16, b, g, r)
			elif i == 20:
				frame[h, w] = getBGR(lj17, b, g, r)

	# show the frame to our screen
	cv.imshow(windowName, frame)
	# cv.imshow("Origin", ori)
	# print(frame.shape)
	# frame = imutils.resize(frame, width=size[1])
	# print(frame.shape)
	key = cv.waitKey(1) & 0xFF

	# save the frame
	out.write(frame)

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# release the video writer
out.release()

# close all windows
cv.destroyAllWindows()