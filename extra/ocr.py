from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import logging

# Suppress unnecessary warnings and information
logging.getLogger("transformers").setLevel(logging.ERROR)
Image.warnings.simplefilter('ignore')

def perform_ocr(image_path):
   
    try:
        # Load processor and model with custom cache location
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', 
                                                 use_fast=True,
                                                 cache_dir=os.environ.get("MODEL_CACHE"))
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed',
                                                        cache_dir=os.environ.get("MODEL_CACHE"))
        
        # Load and process imagea
        image = Image.open(image_path).convert("RGB")
        
        # Process image for TrOCR
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate text from image
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    except Exception as e:
        return f"OCR processing failed: {str(e)}"

def main():
    # Hardcoded image path
    image_path = "image.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Perform OCR and print only the result
    result = perform_ocr(image_path)
    print(result)

if __name__ == "__main__":
    # Set up custom model storage
    model_dir = ".model"
    os.makedirs(model_dir, exist_ok=True)
    os.environ["MODEL_CACHE"] = model_dir
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    main()