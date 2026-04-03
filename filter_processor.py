from PIL import Image, ImageDraw, ImageFont, ImageOps
import base64
import io

class FilterProcessor:
    @staticmethod
    def apply_bw_filter(image_path):
        """
        Apply black and white filter to image
        Returns: Image object in grayscale
        """
        try:
            img = Image.open(image_path)
            bw_img = ImageOps.grayscale(img)
            return bw_img
        except Exception as e:
            print(f"BW filter error: {e}")
            return None

    @staticmethod
    def apply_normal_filter(image_path):
        """
        Return image as-is (normal filter)
        """
        try:
            img = Image.open(image_path)
            return img
        except Exception as e:
            print(f"Normal filter error: {e}")
            return None

    @staticmethod
    def composite_strip_with_template(template_path, photo_paths, filter_type='normal'):
        """
        Composite 4 photos onto template with selected filter
        
        Args:
            template_path: path to template PNG
            photo_paths: list of 4 photo file paths
            filter_type: 'normal' or 'bw'
        
        Returns: PIL Image object (composited strip)
        """
        try:
            # Load template
            strip = Image.open(template_path).convert('RGB')
            
            # Template dimensions
            STRIP_WIDTH = strip.width
            STRIP_HEIGHT = strip.height
            
            TOP_BORDER = 60
            SIDE_BORDER = 60
            PHOTO_SPACING = 40
            BOTTOM_AREA = 100
            
            PHOTOS_PER_STRIP = 4
            available_height = STRIP_HEIGHT - TOP_BORDER - BOTTOM_AREA - (PHOTO_SPACING * (PHOTOS_PER_STRIP - 1))
            PHOTO_HEIGHT = available_height // PHOTOS_PER_STRIP
            PHOTO_WIDTH = STRIP_WIDTH - (SIDE_BORDER * 2)
            
            # Process each photo
            for i, photo_path in enumerate(photo_paths[:4]):
                # Open and apply filter
                photo = Image.open(photo_path).convert('RGB')
                
                if filter_type == 'bw':
                    photo = ImageOps.grayscale(photo).convert('RGB')
                
                # Crop to fit slot
                target_aspect = PHOTO_WIDTH / PHOTO_HEIGHT
                photo_aspect = photo.width / photo.height
                
                if photo_aspect > target_aspect:
                    new_width = int(photo.height * target_aspect)
                    left = (photo.width - new_width) // 2
                    photo = photo.crop((left, 0, left + new_width, photo.height))
                else:
                    new_height = int(photo.width / target_aspect)
                    top = (photo.height - new_height) // 2
                    photo = photo.crop((0, top, photo.width, top + new_height))
                
                photo = photo.resize((PHOTO_WIDTH, PHOTO_HEIGHT), Image.Resampling.LANCZOS)
                
                # Paste onto strip
                y_pos = TOP_BORDER + i * (PHOTO_HEIGHT + PHOTO_SPACING)
                strip.paste(photo, (SIDE_BORDER, y_pos))
            
            # Add branding (same as original)
            import datetime
            draw = ImageDraw.Draw(strip)
            now = datetime.datetime.now()
            branding_text = f"VisionBooth {now.strftime('%m/%d/%y')}"
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except:
                    font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), branding_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = (STRIP_WIDTH - text_width) // 2
            text_y = STRIP_HEIGHT - BOTTOM_AREA + (BOTTOM_AREA - text_height) // 2
            
            shadow_offset = 2
            draw.text((text_x + shadow_offset, text_y + shadow_offset), branding_text, fill=(200, 200, 200), font=font)
            draw.text((text_x, text_y), branding_text, fill=(50, 50, 50), font=font)
            
            return strip
            
        except Exception as e:
            print(f"Composite error: {e}")
            import traceback
            traceback.print_exc()
            return None