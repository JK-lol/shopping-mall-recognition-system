from typing import Union, Annotated
import uvicorn
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import io
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mysql.connector
from mysql.connector import Error

# Create the FastAPI application
app = FastAPI()

# Configure CORS
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models and define constants
mlp = pickle.load(open("mlp.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

# set image size
IMAGE_SIZE = (132, 99)
className = ["", "Dataran Pahlawan", "Mahkota Parade", "Hattens Square"]
label = {v + 1: k for v, k in enumerate(className)}
print(label)


# Create a connection to the MySQL database
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='shopping',
            user='root',
            password=''
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")

    return connection

# Create a class to store the prediction data
class PredictionData(BaseModel):
    image: Annotated[bytes, File()]
    predicted_label: str
    user_label: str

# Load PCA and MLP (pre-trained model)
@app.post("/predictImage")
async def predict_image(image: Annotated[bytes, File()]):
    # Process the input image
    img = Image.open(io.BytesIO(image))
    img = img.resize((132, 99))
    img = img.convert("L")
    img = ImageOps.exif_transpose(img)
    x = np.array([np.array(img)])

    # Reshape and transform the image data
    x_flat = x.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    x_flat_pca = pca.transform(x_flat)

    # Perform prediction using the pre-trained model
    y = mlp.predict(x_flat_pca)[0]
    response = {"Label": className[y]}

    return JSONResponse(content=jsonable_encoder(response))

# Store the prediction data in MySQL database
@app.post("/storePrediction")
async def store_prediction(user_label: str, predicted_label: str, image: Annotated[bytes, File()]):
    # Store the data in MySQL database
    connection = create_connection()
    if connection is not None:
        try:
            cursor = connection.cursor()
            # Prepare the SQL query
            query = "INSERT INTO predictions (predicted_label, user_label, image) VALUES (%s, %s, %s)"
            values = ( predicted_label, user_label, image)
            # Execute the query
            cursor.execute(query, values)
            connection.commit()
            cursor.close()
            response = {"message": "Data stored successfully"}
        except Error as e:
            response = {"error": f"Error storing data in MySQL database: {e}"}
        finally:
            if connection.is_connected():
                connection.close()
                print("MySQL database connection closed")
    else:
        response = {"error": "Error connecting to MySQL database"}

    return JSONResponse(content=jsonable_encoder(response))


if __name__ == '__main__':
    # Run the FastAPI application
    uvicorn.run("predict:app", host="localhost", port=8000, reload=True)
