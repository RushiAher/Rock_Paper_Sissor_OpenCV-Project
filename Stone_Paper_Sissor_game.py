import numpy as np
import cv2
import math
import random

# image list for computer move
s_p_s_img = ["stone.png","papper.png","sissor.png"]

# Open Camera
capture = cv2.VideoCapture(0)

# creating black image window using numpy zeros method
black_win = np.zeros((480,640,3),np.uint8)

# variable to store privious defect count
store_defect__count = 0

comp_move = black_win



def Set_result(comp_move_select):
    if comp_move_select == "stone.png":
        ret = 0
        cv2.putText(comp_move, "Stone", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, [0, 255, 0], 2)
    elif comp_move_select == "papper.png":
        ret = 4
        cv2.putText(comp_move, "Paper", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, [0, 255, 0], 2)
    else:
        ret = 1
        cv2.putText(comp_move, "Sissor", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, [0, 255, 0], 2)
    return ret


while capture.isOpened():

    # creating result window
    result_win = np.zeros((480, 140, 3), np.uint8)

    # printing result
    result_win = cv2.putText(result_win,"Winner",(10,200),cv2.FONT_HERSHEY_COMPLEX,0.8,[0,255,0])

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (150, 150), (450, 450), (0, 255, 0), 0)
    crop_image = frame[150:450, 150:450]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([20, 150, 255]))

    # defining Kernel
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key = lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point

        count_defects = 0

        # checking defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle < 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 2, [0, 0, 255], -1)


            cv2.line(crop_image, start, end, [0, 255, 0], 2)
        if count_defects == store_defect__count:
            pass
        else:
            comp_move_select = random.choice(s_p_s_img)



        if count_defects==0:
            store_defect__count = count_defects

            cv2.putText(frame, "stone", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            comp_move = cv2.imread(comp_move_select)

            # calling set_result dunction
            res = Set_result(comp_move_select)

            # check result
            if res == 0:
                result = "Tie"

            elif res == 4:
                result = "Computer"

            else:
                result = "User"


        elif count_defects == 1:
            store_defect__count = count_defects

            cv2.putText(frame, "Sissor", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            comp_move = cv2.imread(comp_move_select)

            # calling set_result function
            res = Set_result(comp_move_select)

            # check result
            if res == 1:
                result = "Tie"

            elif res == 4:
                result = "User"

            else:
                result = "Computer"


        elif count_defects == 4:
            store_defect__count = count_defects

            cv2.putText(frame, "Papper", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            comp_move = cv2.imread(comp_move_select)

            # calling set_result function
            res = Set_result(comp_move_select)

            # check result
            if res == 4:
                result = "Tie"

            elif res == 1:
                result = "Computer"

            else:
                result = "User"


        elif count_defects == 2 or count_defects==3 or count_defects==5:
            store_defect__count=count_defects
            comp_move = black_win
            result = "Waiting...."


    except:
        # exception hadling
        comp_move =black_win
        result="Waiting...."


    result_win = cv2.putText(result_win,result,(10,250),cv2.FONT_HERSHEY_COMPLEX,0.8,[0,255,0])

    # merging computer_move window, result window and frame window
    all_image = np.hstack((comp_move,result_win, frame))

    # showing all images
    cv2.imshow("S_P_S_Game",all_image)

    # quit if escape key is pressed
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
