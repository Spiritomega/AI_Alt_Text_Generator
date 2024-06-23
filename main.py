import os
import time
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import piexif
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processor and model for caption generation
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def add_metadata(image_path, subject, description, output_path):
    try:
        with Image.open(image_path) as image:
            # Load existing EXIF data
            exif_data = image.info.get("exif", b'')
            exif_dict = piexif.load(exif_data) if exif_data else {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None}

            # Add description and subject
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = subject.encode('utf-8')

            # Convert the exif_dict back to bytes and save the image with the new EXIF data
            exif_bytes = piexif.dump(exif_dict)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path, "jpeg", exif=exif_bytes)
            print(f"Metadata added and image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred while adding metadata to {image_path}: {e}")


def generate_filename(description):
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokenize(description.lower()) if word.isalnum() and word not in stop_words]
    cleaned_description = ' '.join(words)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([cleaned_description])
    word_scores = {word: score for word, score in zip(vectorizer.get_feature_names_out(), X.toarray()[0])}
    top_words = sorted(word_scores, key=word_scores.get, reverse=True)[:2]
    return '-'.join(top_words)


def resize_and_enhance_image(input_path, output_path, max_width, max_height, resize_percentage):
    # Open the image file using Pillow
    with Image.open(input_path) as img:
        # Get original dimensions
        original_width, original_height = img.size

        # Determine if the image needs to be resized
        if original_width > max_width or original_height > max_height:
            # Calculate new dimensions while maintaining the aspect ratio
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                new_width = min(max_width, int(original_width * resize_percentage / 100))
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(max_height, int(original_height * resize_percentage / 100))
                new_width = int(new_height * aspect_ratio)

            # Resize the image using high-quality downsampling with Pillow (Image.LANCZOS)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert the resized image to a numpy array compatible with OpenCV
            open_cv_image = np.array(resized_img)

            # Convert RGB to BGR (OpenCV uses BGR format)
            open_cv_image = open_cv_image[:, :, ::-1].copy()

            # Apply a minimal Gaussian blur using OpenCV
            blurred_img = cv2.GaussianBlur(open_cv_image, (3, 3), 0.1)  # Reduced sigma value for minimal blur

            # Convert back to PIL format
            resized_img = Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))

            # Enhance sharpness slightly using Pillow (ImageEnhance.Sharpness)
            enhancer = ImageEnhance.Sharpness(resized_img)
            sharpened_img = enhancer.enhance(1.25)  # Adjust sharpness factor as needed

            # Enhance contrast slightly to improve overall appearance
            enhanced_img = ImageEnhance.Contrast(sharpened_img).enhance(1.03)

            # Save the enhanced image
            enhanced_img.save(output_path)
            print(f"Image resized, enhanced, and saved to {output_path}")
        else:
            # Save the original image if no resizing is needed
            img.save(output_path)
            print(f"Image dimensions are within the limit. Saved original image to {output_path}")



# Paths and settings
input_dir = 'Images'
output_dir = 'output'
alt_dir = 'alt'
max_width = 2000
max_height = 2500
resize_percentage = 75

os.makedirs(output_dir, exist_ok=True)
os.makedirs(alt_dir, exist_ok=True)

# Resize and enhance images
for filename in os.listdir(input_dir):
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, filename)
    resize_and_enhance_image(input_file, output_file, max_width, max_height, resize_percentage)

# Generate captions and add metadata
for filename in os.listdir(output_dir):
    img_path = os.path.join(output_dir, filename)
    try:
        raw_image = Image.open(img_path)

        # Conditional image captioning
        text = "Portrait of"
        inputs = processor(raw_image, text, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)  # Adjust max_new_tokens as needed
        desc_words = processor.decode(out[0], skip_special_tokens=True)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)  # Adjust max_new_tokens as needed
        subject_words = processor.decode(out[0], skip_special_tokens=True).replace("arafed ", "")

        title_words = f"{generate_filename(desc_words)}-{filename.strip().replace(' ', '-')}"
        output_file = os.path.join(alt_dir, title_words)
        add_metadata(img_path, subject_words, desc_words, output_file)
    except Exception as e:
        print(f"An error occurred while processing {img_path}: {e}")

print("Process Completed")
