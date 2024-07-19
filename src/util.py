import cv2
import numpy as np

def load_cv2_img(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def find_color_cube(image, color, tolerance=10):    
    lower_color = np.array([max(c - tolerance, 0) for c in color])
    upper_color = np.array([min(c + tolerance, 255) for c in color])
    
    mask = cv2.inRange(image, lower_color, upper_color)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    return cv2.boundingRect(largest_contour)

def mask_bordered_region(image, border_color):
    border_color_rgb = tuple(reversed(border_color))

    lower = np.array(border_color_rgb) - 5
    upper = np.array(border_color_rgb) + 5
    mask = cv2.inRange(image, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            cv2.drawContours(image, [approx], 0, border_color, -1)
            cv2.drawContours(image, [approx], 0, border_color_rgb, 2)

    return image

def create_color_mask(image, target_color, tolerance=50, conv=True):
    if conv:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert target color to a NumPy array
    target_color = np.array(target_color, dtype=np.uint8)
    
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Iterate over each pixel
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Get the color of the current pixel
            pixel_color = image[y, x]
            # Calculate the absolute difference between the pixel color and target color
            if np.all(np.abs(pixel_color - target_color) <= tolerance):
                mask[y, x] = 255  # Set to white if it matches the target color
    
    return mask

def mask_straight_lines(image, min_line_length=20, max_line_gap=10, thickness=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line is horizontal or vertical
            if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:  # Vertical or horizontal
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    
    return mask

def show_img(title, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()