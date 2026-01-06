## hardik jangir: 22b3901
## archishman : 22b2405


## the video link for the demo:  https://drive.google.com/file/d/171dNHXwYJxIV4NB6TBJpw3M1FCYS_05x/view?usp=sharing


## the video link for the code explaination:   https://drive.google.com/file/d/1dKJFB9svGhfQk4gRgzV9NYpx6hybUYcC/view?usp=sharing

## also i made changes in the trusted users folder and removed my images, used a dummy image
from deepface import DeepFace
import cv2, os, pickle, numpy as np

trusted_embeddings = {}
IMG_SIZE = (160, 160)  # standard FaceNet input

def augment_face(face_img):
    """Return list of augmentations for a face crop"""
    aug_faces = []
    
    # Flip
    aug_faces.append(cv2.flip(face_img, 1))  # horizontal
    aug_faces.append(cv2.flip(face_img, 0))  # vertical
    
    # Brightness
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    
    # brighter
    hsv_bright = hsv.copy()
    hsv_bright[:,:,2] = cv2.add(hsv_bright[:,:,2], 30)
    aug_faces.append(cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR))
    
    # darker
    hsv_dark = hsv.copy()
    hsv_dark[:,:,2] = cv2.subtract(hsv_dark[:,:,2], 30)
    aug_faces.append(cv2.cvtColor(hsv_dark, cv2.COLOR_HSV2BGR))
    
    return aug_faces

for user in os.listdir("trusted_users"):
    user_path = os.path.join("trusted_users", user)
    if not os.path.isdir(user_path):
        continue
    
    embeddings_list = []
    for img_file in os.listdir(user_path):
        img_path = os.path.join(user_path, img_file)
        try:
            # Detect face
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend="retinaface",
                enforce_detection=True
            )

            for face in face_objs:
                bbox = face["facial_area"]
                (x,y,w,h) = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])

                img = cv2.imread(img_path)
                if img is None: continue

                # Draw bbox for visualization
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.imshow("Detected Face", img)
                cv2.waitKey(300)

                # Crop + resize
                cropped = img[y:y+h, x:x+w]
                cropped_resized = cv2.resize(cropped, IMG_SIZE)

                # Prepare augmented dataset (original + aug)
                face_variants = [cropped_resized] + augment_face(cropped_resized)

                # Extract embeddings for each variant
                for fimg in face_variants:
                    embedding = DeepFace.represent(
                        img_path=fimg,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                    embeddings_list.append(embedding)

        except Exception as e:
            print(f"Skipping {img_path}, error: {e}")
    
    if embeddings_list:
        # Take centroid
        mean_embedding = np.mean(embeddings_list, axis=0)
        trusted_embeddings[user] = mean_embedding
    
cv2.destroyAllWindows()

# Save embeddings
with open("trusted_embeddings.pkl", "wb") as f:
    pickle.dump(trusted_embeddings, f)

print("✅ Embeddings with augmentations + centroid saved")
