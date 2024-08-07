import threading
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_matches = False

# Define reference images along with their names
reference_images = {
    "Tanvi": cv2.imread("C:\\Users\\Tanvi\\photos\\1714112927629.jpg"),
    "Shagun": cv2.imread("C:\\Users\\Tanvi\\Downloads\\1713771469094.jpg"),
    "Anshika": cv2.imread("C:\\Users\\Tanvi\\photos\\1714112927647.jpg"),
}
    # Add more reference images here as needed


def check_face(frame):
    global face_matches
    try:
        for name, reference_img in reference_images.items():
            result = DeepFace.verify(frame, reference_img)
            if result['verified']:
                face_matches = True
                return name
            
    except Exception as e:
        print("Error in face verification:", e)
        face_matches = False
        return None
    

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                thread = threading.Thread(target=check_face, args=(frame,))
                thread.start()
                matched_name = thread.join()
                 
            except Exception as e:
                print("Error starting thread:", e)

        counter += 1

    
        
        if face_matches:
                cv2.putText(frame, "MATCH!".format(matched_name), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame using matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
