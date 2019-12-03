import cv2
import matplotlib.pyplot as plt
def main():

    while True:
        repeat()
        

def repeat():
    # Read image
    ret, frame = cap.read()
    # print(frame.shape)
    # Specify the input image dimensions
    frame = rescale(frame, .6)
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
                   'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee' , 'Right Ankle',
                   'Left Hip' , 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']
    # body_parts = ['Head', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 'Left Elbow',
    #               'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 'Left Knee', 'Left Ankle', 'Chest',
    #               'Background']
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255, size=(inWidth, inHeight), mean=(0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    out = net.forward()
    # print(out.shape)

    H = out.shape[2]
    W = out.shape[3]
    # Empty list to store the detected keypoints
    threshold = .2
    points = []
    for i in range(len(body_parts)):
        # confidence map of corresponding body's part.
        probMap = out[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (inWidth * point[0]) / W
        y = (inHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), inWidth//50, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(body_parts[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    cv2.imshow("Output-Keypoints", frame)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    # for pair in [(i, i + 1) for i in range(len(body_parts) -1)]:
    #     partA = pair[0]
    #     partB = pair[1]
    #
    #     if points[partA] and points[partB]:
    #         cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)
    # cv2.imshow("Output-Skeleton", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def rescale(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

if __name__ == '__main__':

    # Specify the paths for the 2 files
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"

    cap = cv2.VideoCapture(0)

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    main()