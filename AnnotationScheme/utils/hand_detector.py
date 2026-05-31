from ultralytics import YOLO
import numpy as np

class HandDetector:
    """Wrapper for YOLO-based hand detection to provide prompts for SAM2."""
    
    def __init__(self, model_path='yolo11n-pose.pt'):
        # Note: yolo11n-pose.pt detects person keypoints. 
        # For dedicated hand detection, a custom model might be better.
        # But we'll start with this and filter for wrist/hand area.
        self.model = YOLO(model_path)
        
    def detect_hands(self, image, conf=0.5):
        """
        Detect hands in the image and return points for SAM2 prompts.
        Returns: list of (points, labels)
        """
        results = self.model(image, conf=conf, verbose=False)
        all_prompts = []
        
        for r in results:
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                try:
                    kpts = r.keypoints.data.cpu().numpy() # [num_persons, 17, 3]
                    for person in kpts:
                        # Wrist points
                        for idx in [9, 10]:
                            x, y, c = person[idx]
                            if c > conf:
                                # Add a point prompt (rounded to integers for UI/OpenCV compatibility)
                                ix, iy = int(round(float(x))), int(round(float(y)))
                                points = np.array([[ix, iy]], dtype=np.int32)
                                labels = np.array([1], dtype=np.int32)
                                all_prompts.append((points, labels))
                except Exception as e:
                    print(f" [!] Error extracting keypoints: {e}")
                            
            if hasattr(r, 'boxes') and r.boxes is not None:
                # If using a model that detects hands as boxes
                boxes = r.boxes.data.cpu().numpy()
                for box in boxes:
                    # If class is 'hand' (depends on the model)
                    # For person detection, we could use the box area around wrists
                    pass
                    
        return all_prompts
