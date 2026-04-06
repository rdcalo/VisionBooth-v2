from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import datetime


class FilterProcessor:

    @staticmethod
    def _find_photo_slots(template: Image.Image, n_photos: int = 4):
        """
        Detect photo slot boundaries by finding horizontal bands of
        highly-saturated color (the decorative separators between photos).
        The gaps between those bands are the photo slots.
        """
        try:
            arr = np.array(template.convert('RGB')).astype(float)
            h, w = arr.shape[:2]

            cmax = np.max(arr, axis=2)
            cmin = np.min(arr, axis=2)
            saturation = (cmax - cmin) / (cmax + 1e-6)
            row_sat = saturation.mean(axis=1)

            SAT_THRESHOLD = 0.25
            is_border = row_sat > SAT_THRESHOLD

            # Group consecutive border rows into bands
            border_bands = []
            in_band = False
            band_start = 0
            for i, flag in enumerate(is_border):
                if flag and not in_band:
                    band_start = i
                    in_band = True
                elif not flag and in_band:
                    border_bands.append((band_start, i - 1))
                    in_band = False
            if in_band:
                border_bands.append((band_start, h - 1))

            if len(border_bands) < 2:
                raise ValueError(f"Only {len(border_bands)} border bands found")

            # Gaps between border bands = photo slots
            gap_regions = []
            if border_bands[0][0] > 5:
                gap_regions.append((0, border_bands[0][0] - 1))
            for i in range(len(border_bands) - 1):
                y1 = border_bands[i][1] + 1
                y2 = border_bands[i + 1][0] - 1
                if y2 - y1 > 10:
                    gap_regions.append((y1, y2))
            last_end = border_bands[-1][1]
            if h - last_end > 20:
                gap_regions.append((last_end + 1, h - 1))

            top_gaps   = sorted(gap_regions, key=lambda g: g[1] - g[0], reverse=True)[:n_photos]
            photo_gaps = sorted(top_gaps, key=lambda g: g[0])

            if len(photo_gaps) < n_photos:
                raise ValueError(f"Only {len(photo_gaps)} gaps, need {n_photos}")

            slots = [(0, g[0], w, g[1]) for g in photo_gaps]
            print(f"[slots] {slots}")
            return slots

        except Exception as e:
            print(f"[slot detection] {e} — falling back")
            return None

    @staticmethod
    def _fallback_slots(sw, sh, n=4):
        slot_h = sh // n
        return [(0, i * slot_h, sw, (i + 1) * slot_h) for i in range(n)]

    @staticmethod
    def composite_strip_with_template(template_path, photo_paths, filter_type='normal'):
        try:
            template = Image.open(template_path).convert('RGBA')
            sw, sh = template.size

            # ── Step 1: build a canvas filled with photos ─────────────────
            # Start with a plain white canvas the same size as the template
            canvas = Image.new('RGBA', (sw, sh), (255, 255, 255, 255))

            slots = FilterProcessor._find_photo_slots(
                template.convert('RGB'), len(photo_paths[:4])
            )
            if slots is None:
                slots = FilterProcessor._fallback_slots(sw, sh, len(photo_paths[:4]))

            for i, photo_path in enumerate(photo_paths[:4]):
                x1, y1, x2, y2 = slots[i]
                slot_w = x2 - x1
                slot_h = y2 - y1
                if slot_w <= 0 or slot_h <= 0:
                    continue

                photo = Image.open(photo_path).convert('RGB')

                if filter_type == 'bw':
                    photo = ImageOps.grayscale(photo).convert('RGB')

                # Centre-crop to slot aspect ratio
                target_aspect = slot_w / slot_h
                photo_aspect  = photo.width / photo.height

                if photo_aspect > target_aspect:
                    new_w = int(photo.height * target_aspect)
                    left  = (photo.width - new_w) // 2
                    photo = photo.crop((left, 0, left + new_w, photo.height))
                else:
                    new_h = int(photo.width / target_aspect)
                    top_c = (photo.height - new_h) // 2
                    photo = photo.crop((0, top_c, photo.width, top_c + new_h))

                photo = photo.resize((slot_w, slot_h), Image.Resampling.LANCZOS)
                # Paste photo onto canvas at detected slot position
                canvas.paste(photo.convert('RGBA'), (x1, y1))

            # ── Step 2: overlay the template ON TOP of the photos ─────────
            # This preserves the borders, decorations, and branding area
            # Template must be RGBA so its transparent areas show the photos
            canvas.paste(template, (0, 0), mask=template)

            # ── Step 3: branding text ─────────────────────────────────────
            result = canvas.convert('RGB')
            draw   = ImageDraw.Draw(result)
            branding = f"VisionBooth {datetime.datetime.now().strftime('%m/%d/%y')}"

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", 28)
                except Exception:
                    font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), branding, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            tx   = (sw - tw) // 2
            ty   = sh - th - 12
            draw.text((tx + 1, ty + 1), branding, fill=(180, 180, 180), font=font)
            draw.text((tx,     ty),     branding, fill=(50,  50,  50),  font=font)

            return result

        except Exception as e:
            print(f"Composite error: {e}")
            import traceback
            traceback.print_exc()
            return None