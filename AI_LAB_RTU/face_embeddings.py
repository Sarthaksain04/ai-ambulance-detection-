import os
import pickle
from deepface import DeepFace

# Folder containing images of known faces
known_faces_dir = "known_faces"

# Initialize an empty dictionary for storing embeddings
embeddings_db = {}

# Load the FaceNet model
facenet_model = DeepFace.build_model('Facenet')
print("FaceNet model loaded successfully.")

# Generate embeddings for each image in the folder
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        person_name = os.path.splitext(filename)[0]  # Use the file name as the person's name
        image_path = os.path.join(known_faces_dir, filename)

        try:
            # Generate face embedding (remove the model argument here)
            embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
            embeddings_db[person_name] = embedding
            print(f"Processed {person_name}: {embedding[:5]}...")  # Print first 5 values of the embedding
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save the embeddings to a file
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_db, f)

print("Saved face embeddings to face_embeddings.pkl")

# # The code snippet above demonstrates how to generate face embeddings for known faces using the FaceNet model from DeepFace. The script loads the FaceNet model, processes images of known faces from a specified directory, generates embeddings for each face, and saves the embeddings to a pickle file for later use in face recognition tasks. The generated embeddings can be used to compare and recognize faces in images or video frames.


# import os
# import pickle
# from deepface import DeepFace

# # Folder containing images of known faces
# known_faces_dir = "known_faces"

# # Initialize an empty dictionary for storing embeddings
# embeddings_db = {}

# # Generate embeddings for each image in the folder using the ArcFace model
# for filename in os.listdir(known_faces_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         person_name = os.path.splitext(filename)[0]  # Use the file name as the person's name
#         image_path = os.path.join(known_faces_dir, filename)

#         try:
#             # Generate face embedding using ArcFace (instead of Facenet)
#             embedding = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
#             embeddings_db[person_name] = embedding
#             print(f"Processed {person_name}: {embedding[:5]}...")  # Print first 5 values of the embedding
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# # Save the embeddings to a file
# with open("face_embeddings.pkl", "wb") as f:
#     pickle.dump(embeddings_db, f)

# print("Saved face embeddings to face_embeddings.pkl")
