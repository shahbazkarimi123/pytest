import cv2
from pyeegrec.iris import IrisRecognition

# Initialize the IrisRecognition class
iris_recognition = IrisRecognition()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
print('hi')
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect iris in the frame
    iris_result = iris_recognition.detect_iris(frame)

    if iris_result is not None:
        # If iris is detected, iris_result will contain information about the detected iris
        iris_code = iris_result['iris_code']
        print("Iris Code:", iris_code)

        # You can use the iris code for authentication or further processing
        # Add your authentication logic here
        
    # Display the frame with detected iris
    cv2.imshow('Iris Authentication', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()