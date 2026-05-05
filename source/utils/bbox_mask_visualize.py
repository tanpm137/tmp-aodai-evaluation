from PIL import Image, ImageDraw
import random
import numpy as np

def visualize_boxes(image: Image.Image, bboxes: list[tuple]):

    image_rgba = image.convert("RGBA")
    mask_layer = Image.new("RGBA", image_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(mask_layer)

    for x1, y1, x2, y2 in bboxes:
        outline_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 255)
        fill_color = (outline_color[0], outline_color[1], outline_color[2], 64)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=3)

    final_image = Image.alpha_composite(image_rgba, mask_layer)
    return final_image.convert("RGB")


def visualize_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.5, color: tuple = (255, 0, 0)) -> Image.Image:

    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)

    overlay_np[mask == 255] = [color[0], color[1], color[2], int(alpha * 255)]

    overlay = Image.fromarray(overlay_np, "RGBA")
    result = Image.alpha_composite(image_rgba, overlay)
    return result.convert("RGB")
