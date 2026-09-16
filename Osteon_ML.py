import pandas as pd
import onnxruntime as rt
import numpy as np
import time
import sys

MODEL_PATH = "model/random_forest_mastery_model.onnx"
QUESTIONS_CSV = "csv/questions_datasets.csv"

# Load ONNX Session
try:
    session = rt.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"[INFO] ONNX model successfully loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"[ERROR] Could not load ONNX model: {e}")
    sys.exit(1)

# Load Questions
try:
    questions_df = pd.read_csv(QUESTIONS_CSV)
    print(f"[INFO] Loaded {len(questions_df)} questions from '{QUESTIONS_CSV}'\n")
except FileNotFoundError:
    print(f"[ERROR] Could not find '{QUESTIONS_CSV}'.")
    sys.exit(1)

print("="*60)
print("     VR PHYSICAL THERAPY / ANATOMY ADAPTIVE LEARNING LOOP     ")
print("="*60)

current_mastery = 0.50  # Initial baseline
asked_question_ids = set()

def determine_target_difficulty(mastery):
    """Map current mastery score to question difficulty tier."""
    if mastery < 0.50:
        return "Easy"
    elif mastery <= 0.75:
        return "Medium"
    else:
        return "Hard"

step = 1

while True:
    # 1. Determine difficulty level based on current mastery
    target_difficulty = determine_target_difficulty(current_mastery)
    
    # 2. Filter unasked questions by target difficulty
    available_questions = questions_df[
        (questions_df['difficulty_level'].str.lower() == target_difficulty.lower()) & 
        (~questions_df['question_id'].isin(asked_question_ids))
    ]
    
    # Fallback if all questions in target difficulty are exhausted
    if available_questions.empty:
        available_questions = questions_df[~questions_df['question_id'].isin(asked_question_ids)]
        if available_questions.empty:
            print("\n[COMPLETE] All questions in dataset completed!")
            break

    # 3. Select next question
    q = available_questions.iloc[0]
    asked_question_ids.add(q['question_id'])
    
    print(f"\n--- Question {step} [{q['bone_name']} - {q['difficulty_level']}] (Target: {target_difficulty}) ---")
    print(f"Question: {q['question_text']}")
    print(f"A. {q['option_a']}")
    print(f"B. {q['option_b']}")
    print(f"C. {q['option_c']}")
    print(f"D. {q['option_d']}")
    
    start_time = time.time()
    user_choice = input("\nYour Answer (A/B/C/D) or 'exit': ").strip().upper()
    
    if user_choice == 'EXIT':
        print("\nSession ended by user.")
        break
        
    elapsed_time = round(time.time() - start_time, 2)
    is_correct = 1.0 if user_choice == str(q['correct_option']).strip().upper() else 0.0
    
    # Construct model input
    raw_features = np.array([[
        float(q['difficulty_score']),
        float(elapsed_time),
        float(is_correct),
        float(current_mastery),
        1.0 if str(q['bone_group']).lower() == 'appendicular' else 0.0,
        0.0 if str(q['difficulty_level']).lower() == 'easy' else (1.0 if str(q['difficulty_level']).lower() == 'medium' else 2.0)
    ]], dtype=np.float32)

    onnx_inputs = {input_name: raw_features}
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], onnx_inputs)[0]
    
    # Extract predicted DELTA shift from the model
    predicted_delta = float(prediction[0][0])
    
    # --- DYNAMIC RULE SAFETY NET ---
    if is_correct == 0.0:
        # If ML model fails to predict a negative delta, calculate a dynamic penalty
        if predicted_delta >= 0.0:
            # Drop more points if you fail an Easy question than a Hard question
            diff_level = str(q['difficulty_level']).lower()
            if diff_level == 'easy':
                predicted_delta = -0.12  # Bigger penalty for missing easy questions
            elif diff_level == 'medium':
                predicted_delta = -0.08
            else:
                predicted_delta = -0.04  # Smaller penalty for missing hard questions
    else:
        # Correct answer fallback
        if predicted_delta <= 0.0:
            predicted_delta = 0.05

    # Apply DELTA to current mastery and clamp bounds between 0.0 and 1.0
    new_mastery = float(np.clip(current_mastery + predicted_delta, 0.0, 1.0))
    
    delta = new_mastery - current_mastery
    direction = "Increased" if delta >= 0 else "Decreased"
    
    print("\n" + "-"*45)
    print(f"Result:            {'CORRECT (+)' if is_correct == 1.0 else 'INCORRECT (-)'}")
    print(f"Response Time:     {elapsed_time} seconds")
    print(f"Previous Mastery:  {current_mastery:.4f}")
    print(f"Predicted Mastery: {new_mastery:.4f}")
    print(f"Mastery Shift:     {direction} by {abs(delta):.4f} points")
    print("-" * 45)
    
    # Update state for next question
    current_mastery = new_mastery
    step += 1

print("\nAdaptive learning session finished.")