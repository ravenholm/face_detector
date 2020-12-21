# import the necessary packages
from imutils import face_utils
import dlib
import cv2
# from pytictoc import TicToc

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = r"C:\Users\Sx\Documents\AI\Intelexica\Weights_Models_Predictors\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# t = TicToc()
cap = cv2.VideoCapture(0)

while (True):
    # t.tic()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # show the output image with the face detections + facial landmarks
    cv2.imshow('frame', frame)
    # t.toc()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()





