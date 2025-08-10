import os
import json
from datetime import datetime

class FeedbackService:
    def __init__(self, feedback_file="data/feedback.json"):
        self.feedback_file = feedback_file
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        
        # Create file if it doesn't exist
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def load_all_feedback(self):
        """Load all feedback entries from the JSON file."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except json.JSONDecodeError:
            return []

    def save_feedback(self, feedback_entry: dict) -> bool:
        """Save a feedback entry to the JSON file."""
        try:
            data = self.load_all_feedback()
            feedback_entry["timestamp"] = datetime.utcnow().isoformat()
            data.append(feedback_entry)
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

    def save_response_rating(self, rating_entry: dict) -> bool:
        """Alias for saving ratings — keeps code compatible."""
        return self.save_feedback(rating_entry)
