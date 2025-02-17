import cv2
import time
import tensorflow as tf

# Load a pre-trained object detection model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to extract frames at a one-minute interval
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #interval = int(fps)  # One second interval
    interval = int(fps * 60)  # One minute interval

    success, image = cap.read()
    count = 0

    while success:
        if count % interval == 0:
            frame_path = f"{output_folder}/frame_{count // interval}.jpg"
            cv2.imwrite(frame_path, image)
            print(f"Frame {count // interval} extracted.")

            # Run the object detector on the extracted frame
            run_object_detector(frame_path)

        success, image = cap.read()
        count += 1

    cap.release()

# Function to run the object detector on a frame
def run_object_detector(frame_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(frame_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Run the detector
    predictions = model.predict(img_array)

    # Print the top prediction
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    print(f"Frame Analysis - {decoded_predictions[0][1]} ({decoded_predictions[0][2]:.2f})")

# Example usage
if __name__ == "__main__":
    video_path = "Cat_2Mins.mp4"  # Replace with the actual path to your cat video
    #output_folder = "./cat_frames"
    output_folder = "/content/"

    # Extract frames at one-minute intervals
    extract_frames(video_path, output_folder)
