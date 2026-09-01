# generate_test_audio.py
import subprocess
import os

test_cases = [
    {"id": "greeting", "text": "Hello, how are you?"},
    {"id": "math_simple", "text": "What is 12 plus 15?"},
    {"id": "math_complex", "text": "Multiply 47 by 12"},
    {"id": "calendar_check", "text": "Check my calendar for August 15th"},
    {"id": "handoff_trigger", "text": "Can you help me find a free slot next week?"},
    {"id": "off_topic", "text": "Tell me a fun fact about space"},
]

os.makedirs("test_audio", exist_ok=True)

for case in test_cases:
    output_path = f"test_audio/{case['id']}.wav"
    subprocess.run(
        ["piper", "--model", "/home/rishu/en_US-lessac-medium.onnx", "--output_file", output_path],
        input=case["text"], text=True, capture_output=True
    )
    print(f"Generated {output_path}: \"{case['text']}\"")