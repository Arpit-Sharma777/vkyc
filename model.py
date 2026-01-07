from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# 1. Print Model Info (Architecture & Layers)
print("--- Model Info ---")
model.info()

# 2. Get Class Names
print("\n--- Class Names ---")
print(model.names)

# 3. Check specific training arguments (if available)
# This often reveals image size, epochs, and dataset used.
if hasattr(model, 'ckpt') and model.ckpt:
    print("\n--- Training Metadata ---")
    print(f"Epochs trained: {model.ckpt.get('epoch', 'Unknown')}")
    print(f"Best fitness score: {model.ckpt.get('best_fitness', 'Unknown')}")
    print(f"Date created: {model.ckpt.get('date', 'Unknown')}")