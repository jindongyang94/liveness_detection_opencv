"""
USAGE
python gather_examples.py -i videos -d face_detector -s 8 -r 1
python gather_examples.py -i videos -d face_detector -s 6 -r 0
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from glob import glob


def bulk_processing(args):
	"""
	Bulk Process all Images in the folder
	"""
	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# Split the videos into two batches: real and fake
	video_dir = args['input']
	video_sub_folders = os.path.sep.join([video_dir, '*/'])
	video_sub_folders = glob(video_sub_folders)

	for sub_folder in video_sub_folders:
		
		# Detect video type and dataset path
		videotype = str(os.path.split(sub_folder)[-2])
		datasetpath = os.path.sep.join(['dataset', videotype])

		# Iterate through all videos in each folder
		videos = glob(os.path.sep.join([sub_folder, '*.mov']))
		videos.extend(glob(os.path.sep.join([sub_folder, '*.mp4'])))

		# number of frames saved thus far
		saved = 0

		# open up existing images in the current folder and append to it instead of overwriting it
		images = glob(os.path.sep.join([datasetpath, "*.png"]))
		images.extend(glob(os.path.sep.join([datasetpath, '*.jpg'])))
		if args['reset']:
			for im in images:
				os.remove(im)
		else:
			saved = len(images)

		for video in videos:

			# open a pointer to the video file stream and initialize the total
			# number of frames read thus far for skipping
			vs = cv2.VideoCapture(video)
			read = 0

			# loop over frames from the video file stream
			while True:
				# grab the frame from the file
				(grabbed, frame) = vs.read()

				# if the frame was not grabbed, then we have reached the end
				# of the stream
				if not grabbed:
					break

				# increment the total number of frames read thus far
				read += 1

				# check to see if we should process this frame
				if read % args["skip"] != 0:
					continue

				# grab the frame dimensions and construct a blob from the frame
				(h, w) = frame.shape[:2]
				blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
					(300, 300), (104.0, 177.0, 123.0))

				# pass the blob through the network and obtain the detections and
				# predictions
				net.setInput(blob)
				detections = net.forward()

				# ensure at least one face was found
				if len(detections) > 0:
					# we're making the assumption that each image has only ONE
					# face, so find the bounding box with the largest probability
					i = np.argmax(detections[0, 0, :, 2])
					confidence = detections[0, 0, i, 2]

					# ensure that the detection with the largest probability also
					# means our minimum probability test (thus helping filter out
					# weak detections)
					if confidence > args["confidence"]:
						# compute the (x, y)-coordinates of the bounding box for
						# the face and extract the face ROI
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						face = frame[startY:endY, startX:endX]

						# write the frame to disk
						p = os.path.sep.join([datasetpath, "{}.png".format(saved)])
						cv2.imwrite(p, face)
						saved += 1
						print("[INFO] saved {} to disk".format(p))

			# do a bit of cleanup
			vs.release()
			cv2.destroyAllWindows()

if __name__ == "__main__":

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, required=True,
		help="path to input folder to all the videos")
	ap.add_argument("-r", "--reset", type=int, default=0,
		help="Option to delete all given images in the ")
	ap.add_argument("-d", "--detector", type=str, required=True,
		help="path to OpenCV's deep learning face detector")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip", type=int, default=16,
		help="# of frames to skip before applying face detection")
	args = vars(ap.parse_args())

	bulk_processing(args)
