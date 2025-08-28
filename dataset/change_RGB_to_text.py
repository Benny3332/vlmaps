from color_text_dict import (iscc_nbs_colors, d3_40_colors_rgb_list)
import numpy as np
def closest_iscc_nbs_color(rgb):
    """Find the closest ISCC-NBS color name for a given RGB value"""
    min_distance = float('inf')
    closest_name = "unknown"
    rgb = np.asarray(rgb, dtype=np.float64)
    best_sim = -1.0            
    final_rgb = None
    for name, target_rgb in iscc_nbs_colors.items():
        target_rgb = np.asarray(target_rgb, dtype=np.float64)
        # dot = np.dot(rgb, target_rgb)
        # norm_rgb = np.linalg.norm(rgb)
        # norm_tgt = np.linalg.norm(target_rgb)
        # if norm_rgb == 0 or norm_tgt == 0:
        #     continue

        # similarity = dot / (norm_rgb * norm_tgt)

        # if similarity > best_sim:
        #     best_sim = similarity
        #     closest_name = name
        #     final_rgb = target_rgb
            
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, target_rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_name = name
            final_rgb = target_rgb
    return closest_name, final_rgb

if __name__ == "__main__":
    rgb = [148, 144, 145]
    closest_name, final_rgb = closest_iscc_nbs_color(rgb)
    print(f"Closest ISCC-NBS color name: {closest_name}, target RGB: {final_rgb}")