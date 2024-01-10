import time
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import random

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

# Start a video capture, using device's camera
cap = cv.VideoCapture(0)

# Set camera resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(maxHands=1)

img_counter = 1

timer = 0
stateResult = False
startGame = False
finishedGame = True
player1Move = ""
player2Move = ""
wait = False
resultString = ""

# Check if video file opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Find Hands
    hands, img = detector.findHands(frame)
    player1Move = 'undefined'
    player2Move = 'undefined'

    if hands:
        player1 = hands[0]
        fingersPlayer1 = detector.fingersUp(player1)

        if fingersPlayer1 == [1, 1, 0, 0, 0]:
            startGame = True
            stateResult = False
            finishedGame = False
            intialTime = time.time()
            # Add text on screen to show game has started
            cv.putText(frame, "Game started!", (120, 85), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        elif fingersPlayer1 == [0, 1, 0, 0, 1]:
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

                player2Move = random.randint(0, 2)

                # find player hands for rock
                if fingersPlayer1 == [0, 0, 0, 0, 0]:
                    player1Move = 0
                    cv.putText(frame, "Rock", (120, 145), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

                # find player hands for Paper
                if fingersPlayer1 == [1, 1, 1, 1, 1]:
                    player1Move = 1
                    cv.putText(frame, "Paper", (120, 145), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

                # find player hands for scissors
                if fingersPlayer1 == [0, 1, 1, 0, 0]:
                    player1Move = 2
                    cv.putText(frame, "Scissors", (120, 145), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)

                result = checkResult(player1Move, player2Move)

                if result == 1:
                    resultString = "You Win!"
                    timer = 0
                    intialTime = time.time()
                    wait = True
                if result == 2:
                    resultString = "You Lose!"
                    timer = 0
                    intialTime = time.time()
                    wait = True
                if result == 0:
                    resultString = "TIE!"
                    timer = 0
                    intialTime = time.time()
                    wait = True
                if result == 4:
                    resultString = "undefined player's Hands"
                    timer = 0
                    intialTime = time.time()
                    wait = True

    if wait:
        timer = time.time() - intialTime
        cv.putText(frame, resultString, (120, 175), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        if timer >= 5:
            stateResult = False
            startGame = False
            finishedGame = True
            wait = False

    # Add this line to display the camera frame
    cv.imshow('frame', frame)

    # Press Q on keyboard to exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()