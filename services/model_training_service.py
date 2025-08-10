import json
import os
from collections import defaultdict
from .feedback_service import FeedbackService

IMPROVED_PROMPTS_FILE = "data/improved_prompts.json"

def calculate_reward(rating, sentiment):
    # Example reward formula
    reward = (rating - 3) / 2  # scale 1-5 to -1..+1
    if sentiment.lower() == "positive":
        reward += 0.1
    elif sentiment.lower() == "negative":
        reward -= 0.1
    return max(min(reward, 1), -1)

def generate_improved_prompts(feedback_data):
    prompt_rewards = defaultdict(list)
    prompt_comments = defaultdict(list)

    for fb in feedback_data:
        reward = calculate_reward(fb.get("rating", 3), fb.get("sentiment", "Neutral"))
        prompt_key = fb.get("query", "").strip().lower()

        if prompt_key:
            prompt_rewards[prompt_key].append(reward)
            if fb.get("comment"):
                prompt_comments[prompt_key].append(fb["comment"].lower())

    avg_rewards = {k: sum(v) / len(v) for k, v in prompt_rewards.items()}
    bad_prompts = {k: r for k, r in avg_rewards.items() if r < 0.6}

    improved = {}
    for prompt, score in bad_prompts.items():
        comments = " ".join(prompt_comments.get(prompt, []))

        if "not clear" in comments or "confusing" in comments:
            improvement = (
                f"When answering queries about '{prompt}', use **clear, plain language**, "
                f"avoid technical jargon unless necessary, and provide examples."
            )
        elif "short" in comments or "incomplete" in comments:
            improvement = (
                f"For '{prompt}', provide a **complete, detailed, step-by-step answer**, "
                f"covering all possible cases and adding examples."
            )
        elif "wrong" in comments or "incorrect" in comments:
            improvement = (
                f"When answering '{prompt}', verify the facts before responding. "
                f"Cross-check context from the document and ensure the answer is accurate."
            )
        else:
            improvement = (
                f"For '{prompt}', give a detailed, logically structured explanation with headings, "
                f"numbered steps, and examples. Include context from the product manual."
            )

        improvement += (
            " Always base your answer strictly on the most relevant chunks from the document. "
            "If unsure, clearly state assumptions and guide the user to the right section."
        )

        improved[prompt] = improvement

    return improved

def train_model_from_feedback():
    feedback_service = FeedbackService()
    feedback_data = feedback_service.load_all_feedback()

    if not feedback_data:
        print("⚠️ No feedback found. Skipping training.")
        return False

    improved_prompts = generate_improved_prompts(feedback_data)

    # Save to file
    os.makedirs("data", exist_ok=True)
    with open(IMPROVED_PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(improved_prompts, f, indent=2, ensure_ascii=False)

    print(f"✅ Improved prompts saved to {IMPROVED_PROMPTS_FILE}")
    return True
