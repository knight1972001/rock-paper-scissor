import time
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import random
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the testing labels
class_labels = [1, 0, 2]
class_labels_Name = ['paper', 'rock', 'scissors']
# Load the trained model from disk
model = keras.models.load_model('rock_paper_scissors_model/rock_paper_scissors_model.h5')

def useLearingTest(frame):
    # Preprocess the frame and make a prediction
    img = cv.resize(frame, (224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Add batch dimension
    img_array /= 255.0 # Normalize the pixel values
    preds = model.predict(img_array)

    # Get the predicted class and print it on the frame
    class_idx = tf.argmax(preds[0]).numpy()
    class_num = class_labels[class_idx]
    print(class_labels_Name[class_idx])
    return class_num



def checkResult(player1Move, player2Move):
    print("Player 1: " + str(player1Move) + " | Player 2: " + str(player2Move))
    if player1Move == 'undefined':
        return 4

    if player1Move == player2Move:
        return 0

    if player1Move == 0 and player2Move == 2:
        return 1

    if (player1Move > player2Move):
        return 1

    return 2

def checkPaper(hand_landmarks):
    # Get the y-coordinate of the thumb and the y-coordinates of the other fingers
    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Check if the fingers high in order
    return thumb_y < index_y and index_y < middle_y and middle_y < ring_y and ring_y < pinky_y

def checkRock(hand_landmarks):
    # Get the y-coordinate of the thumb and the y-coordinates of the other fingers
    thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    middle_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x

    # Check if the thumb_x is higher then other fingers
    return thumb_x < index_x and thumb_x < middle_x

def checkScissors(hand_landmarks):
    # Get the x, y, and z coordinates of the thumb finger tip
    thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

    # Get the x, y, and z coordinates of the index finger tip
    index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

    # Get the x, y, and z coordinates of the ring finger nail
    ring_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
    ring_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

    # Get the x, y, and z coordinates of the pinky finger nail
    pinky_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    pinky_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

    # Calculate the distance between the thumb and index finger
    index_distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2 + (thumb_z - index_z) ** 2)

    # Calculate the distance between the thumb and ring finger nail
    ring_distance = math.sqrt((thumb_x - ring_x) ** 2 + (thumb_y - ring_y) ** 2 + (thumb_z - ring_z) ** 2)

    # Calculate the distance between the thumb and pinky finger nail
    pinky_distance = math.sqrt((thumb_x - pinky_x) ** 2 + (thumb_y - pinky_y) ** 2 + (thumb_z - pinky_z) ** 2)
    print(index_distance)
    print(ring_distance)
    print(pinky_distance)
    # Check if the ring finger nail and pinky finger nail are closer to the thumb than the index finger nail
    return (ring_distance < index_distance or pinky_distance < index_distance)


# Start a video capture, using device's camera
cap = cv.VideoCapture(0)

# Set camera resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(maxHands=1)
mp_hands = mp.solutions.hands

img_counter = 1

timer = 0
stateResult = False
startGame = False
finishedGame = True
testMode = False
player1Move = ""
player2Move = ""
wait = False
resultString = ""
handGesture = ""

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if testMode:
        if startGame:
            player1Move = useLearingTest(frame)
            player2Move = random.randint(0, 2)
            result = checkResult(player1Move, player2Move)
            if result == 1:
                print("You Win!")
            if result == 2:
                print("You Lose!")
            if result == 0:
                print("TIE!")
            if result == 4:
                print("undefined player's Hands")
            startGame = False
    else:
        # Find Hands
        hands, img = detector.findHands(frame)
        player1Move = 'undefined'
        player2Move = 'undefined'

        if hands and startGame == False:
            player1 = hands[0]
            fingersPlayer1 = detector.fingersUp(player1)

            if fingersPlayer1 == [1, 1, 0, 0, 0]:
                startGame = True
                stateResult = False
                finishedGame = False
                handGesture = ""
                resultString = ""
                intialTime = time.time()
                # Add text on screen to show game has started
                cv.putText(frame, "Game started!", (120, 85), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            elif fingersPlayer1 == [1, 1, 0, 0, 1]:
                # Quit the game if all fingers are up
                print("Exiting through finger gesture")
                break

        # Press S to start game (1)

        if startGame and finishedGame is False and wait is False:
            # timer (2)
            if stateResult is False:
                timer = time.time() - intialTime
                # Add timer on the screen
                cv.putText(frame, "Showing result after: ", (120, 115), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
                cv.putText(frame, str(5 - int(timer)), (620, 115), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                if timer > 5:
                    stateResult = True
                    timer = 0

                    player1 = hands[0]
                    fingersPlayer1 = detector.fingersUp(player1)
                    player2Move = random.randint(0, 2)

                    # find player hands for rock
                    if fingersPlayer1 == [0, 0, 0, 0, 0]:
                        player1Move = 0
                        handGesture = "Rock"

                    # find player hands for Paper
                    if fingersPlayer1 == [1, 1, 1, 1, 1]:
                        player1Move = 1
                        handGesture = "Paper"

                    # find player hands for scissors
                    if fingersPlayer1 == [0, 1, 1, 0, 0]:
                        player1Move = 2
                        handGesture = "Scissors"

                    # if player is point straight forward to the camera
                    if (player1Move == 'undefined'):
                        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                            results = hands.process(image)
                            if results.multi_hand_landmarks:
                                hand_landmarks = results.multi_hand_landmarks[0]
                                isPaper = checkPaper(hand_landmarks)
                                isScissors = checkScissors(hand_landmarks)
                                isRock = checkRock(hand_landmarks)

                                if isPaper == True:
                                    player1Move = 1
                                    handGesture = "Paper"
                                elif isScissors == True:
                                    player1Move = 2
                                    handGesture = "Scissors"
                                elif isRock == True:
                                    player1Move = 0
                                    handGesture = "Rock"

                    result = checkResult(player1Move, player2Move)

                    if result == 1:
                        resultString = "You Win!"
                        timer = 0
                        intialTime = time.time()
                    if result == 2:
                        resultString = "You Lose!"
                        timer = 0
                        intialTime = time.time()
                    if result == 0:
                        resultString = "TIE!"
                        timer = 0
                        intialTime = time.time()
                    if result == 4:
                        resultString = "undefined player's Hands"
                        timer = 0
                        intialTime = time.time()
                    startGame = False

        cv.putText(frame, handGesture, (120, 145), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
        cv.putText(frame, resultString, (120, 175), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        if wait:
            timer = time.time() - intialTime
            if timer >= 5:
                stateResult = False
                startGame = False
                finishedGame = True
                wait = False

    # Display the frame
    cv.imshow('frame', frame)
    key = cv.waitKey(25)
    # #Press S to restart the game
    if key == ord('s'):
        startGame = True
        intialTime = time.time()

    # #Press M to change into test mode
    if key == ord('m'):
        if testMode:
            print("You are back to the normal mode...")
            testMode = False
        else:
            print("You are now in the learning data testing mode...")
            testMode = True

    # Press Q on keyboard to exit
    if key & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()