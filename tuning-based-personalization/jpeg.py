from PIL import Image
import io
import os

def jpeg_defense_and_save(image_path, save_path, quality=75):
    """
    Apply JPEG compression to an image to defend against adversarial attacks and save the result.
    
    Args:
        image_path (str): Path to the original image.
        save_path (str): Path where the defended image will be saved.
        quality (int): The quality of the JPEG compression (1-100, higher means better quality).
    """
    # Load the original image
    image = Image.open(image_path)
    
    # Save image to a byte buffer in JPEG format with specified quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload the image from the byte buffer
    defended_image = Image.open(buffer)
    
    # Save the defended image to the specified path
    defended_image.save(save_path, format="JPEG", quality=quality)


if __name__ == "__main__":
    used_name = "47"
    save_file = f"Exp/attack/jpeg/{used_name}"
    top_dir = f"EXP/Exp/attack/eps12-255-gam/{used_name}"
    os.makedirs(save_file, exist_ok=True)
    top_list = os.listdir(top_dir)
    for ele in top_list:
        image_file = os.path.join(top_dir,ele)
        save_image = os.path.join(save_file,ele)
        jpeg_defense_and_save(image_file, save_image, quality=85)



