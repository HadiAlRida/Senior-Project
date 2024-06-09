#working code 
import cv2
import networkx as nx
import matplotlib.pyplot as plt
# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('R.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Create a face array to store the detected faces
face_array = []
for (x, y, w, h) in faces:
    face_array.append(gray_image[y:y+h, x:x+w])
#Print the face array
print("Face Array:")
for i, face in enumerate(face_array):
    print(f"Face {i+1}:")
    print(face)

# Create a graph
G = nx.Graph()

# Add nodes (faces) to the graph
for i, (x, y, w, h) in enumerate(faces):
    G.add_node(i, pos=(x, y), size=(w, h))

# Add edges between nodes based on some criteria (e.g., proximity)
for i in range(len(faces)):
    for j in range(i+1, len(faces)):
        dist = ((faces[i][0] - faces[j][0])**2 + (faces[i][1] - faces[j][1])**2)**0.5
        if dist < 200:  # Adjust this threshold as needed
            G.add_edge(i, j)

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
node_size = nx.get_node_attributes(G, 'size')
nx.draw(G, pos, with_labels=False, node_size=[s[0]*s[1]/100 for s in node_size.values()])
plt.show()

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()