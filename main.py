import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your image for face recognition
first_face = cv2.imread("photo1.jpg", cv2.IMREAD_GRAYSCALE)

Second_face = cv2.imread("photo2.jpg", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

while True:
    
    _, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    
    for (x, y, w, h) in faces:
        
        roi = gray[y:y + h, x:x + w]

        
        first_face_resized = cv2.resize(first_face, (w, h))

        second_face_resized = cv2.resize(first_face, (w, h))

        # Compare the histograms of the two faces (simple check)
        similarity_first_face = cv2.compareHist(cv2.calcHist([roi], [0], None, [256], [0, 256]),
                                     cv2.calcHist([first_face_resized], [0], None, [256], [0, 256]),
                                     cv2.HISTCMP_CORREL)

        similarity_second_face = cv2.compareHist(cv2.calcHist([roi], [0], None, [256], [0, 256]),
                                     cv2.calcHist([first_face_resized], [0], None, [256], [0, 256]),
                                     cv2.HISTCMP_CORREL)

        # If the faces are similar (histograms are correlated), draw a rectangle around the face
        if similarity_first_face > 0.7 or similarity_second_face > 0.7 :
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)



    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
