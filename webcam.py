import cv2


import numpy as np
def compare_iris_features(iris_features, stored_iris_features):
    # Calculate Euclidean distance between extracted features and stored features
    distances = []
    for stored_feature in stored_iris_features:
        distance = np.linalg.norm(iris_features - stored_feature)
        distances.append(distance)

    # Return the minimum distance as the match score
    match_score = min(distances)

    return match_score

def extract_iris_features(iris):
    # Apply Gabor filtering to extract texture features
    gabor_filters = generate_gabor_filters()
    filtered_responses = []
    for filter in gabor_filters:
        filtered_response = cv2.filter2D(iris, -1, filter)
        filtered_responses.append(filtered_response)

    # Extract statistical measures as features
    mean_features = []
    std_features = []
    for filtered_response in filtered_responses:
        mean = np.mean(filtered_response)
        std = np.std(filtered_response)
        mean_features.append(mean)
        std_features.append(std)

    # Combine texture features and statistical measures
    iris_features = np.concatenate((mean_features, std_features))

    return iris_features

def segment_iris(eye):
    # Convert eye region to grayscale
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to detect iris boundaries
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough Circle Transform to detect circular iris boundary
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, np.pi/180, 100, 300, 10)

    # Extract iris region based on detected circle parameters
    if circles is not None:
        (x, y, r) = circles[0]
        iris_region = cv2.circle(eye, (x, y), r, (0, 255, 0), 2)
    else:
        iris_region = None

    return iris_region
def detect_eyes(frame):
    # Load the eye cascade classifier
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes using the Haar cascade classifier
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 2)

    # Convert detected eye regions to bounding boxes
    eye_boxes = []
    for (x, y, w, h) in eyes:
        eye_boxes.append((x, y, x+w, y+h))

    return eye_boxes

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Detect eyes in the frame
    eyes = detect_eyes(frame)

    # For each detected eye
    for eye in eyes:
        # Segment the iris from the eye
        iris = segment_iris(eye)

        # Extract features from the segmented iris
        iris_features = extract_iris_features(iris)

        # Match the extracted features with stored iris patterns
        match_score = compare_iris_features(iris_features)

        # Make an authentication decision based on the match score
        if match_score > 0.7:
            print("Authentication successful")
        else:
            print("Authentication failed")

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Check for user input to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam object and close all windows
cap.release()
cv2.destroyAllWindows()


def generate_gabor_filters(ksize=5, num_filters=8):
    filters = []
    for theta in range(num_filters):
        # Generate Gabor filter kernel
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=1.0, theta=2*np.pi*theta / num_filters, lambd=1.0, psi=0)

        # Normalize the filter kernel
        kernel = kernel / np.linalg.norm(kernel)

        # Add the filter kernel to the list of filters
        filters.append(kernel)

    return filters