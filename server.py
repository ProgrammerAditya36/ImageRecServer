from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import face_recognition
import numpy as np
import cv2
from datetime import datetime
import json
import os
import base64
import firebase_admin
from firebase_admin import credentials, storage, firestore

app = FastAPI()

# Initialize Firebase admin SDK
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'baylinkimagerecognition.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()

# Create directories for storing faces and data


# JSON files to store face data
faces_data_file = "faces_data.json"
uploaded_faces_file = "uploaded_faces.json"

class FaceData(BaseModel):
    image: str
    timestamp: str

def load_json(file):
    if not os.path.exists(file):
        return []
    with open(file, 'r') as f:
        return json.load(f)

def save_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)

unique_faces_data = load_json(faces_data_file)
uploaded_faces_data = load_json(uploaded_faces_file)

def compare_faces(known_face_encodings, face_encoding, threshold=0.6):
    if len(known_face_encodings) == 0:
        return False
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return np.any(distances <= threshold)

def upload_to_firebase(filename, face_encoding, image_data):
    # Upload image to Firebase Storage
    if filename in uploaded_faces_data:
        return
    blob_name = filename
    blob = bucket.blob(f"faces/{blob_name}")
    # Convert image back to RGB before uploading
    img_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', img_rgb)
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    blob.make_public()
    image_url = blob.public_url

    # Upload face encoding and image URL to Firestore
    doc_ref = db.collection('faces').document(filename)
    doc_ref.set({
        'encoding': face_encoding.tolist(),
        'url': image_url,
        'timestamp': datetime.now().isoformat()
    })

    # Update uploaded faces list
    uploaded_faces_data.append(filename)
    save_json(uploaded_faces_file, uploaded_faces_data)

@app.post("/process_face")
async def process_face(face_data: FaceData):
    try:
        # Decode image
        image_data = base64.b64decode(face_data.image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get face encoding
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) == 0:
            return {"is_unique": False}
        
        face_encoding = face_encodings[0]

        # Check if face is unique
        known_face_encodings = [np.array(face_data["face_encoding"]) for face_data in unique_faces_data]
        is_unique = not compare_faces(known_face_encodings, face_encoding)

        if is_unique:
            filename = f"face_{face_data.timestamp}.jpg"
            blobname = uuid4().hex 
            face_data = {
                "face_encoding": face_encoding.tolist(),
                "timestamp": face_data.timestamp,
                "filename": filename
            }
            unique_faces_data.append(face_data)
            save_json(faces_data_file, unique_faces_data)

            # Upload to Firebase
            upload_to_firebase(blobname, face_encoding, image_data)

            return {"is_unique": True, "filename": filename}
        else:
            return {"is_unique": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"Hello": "World"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
