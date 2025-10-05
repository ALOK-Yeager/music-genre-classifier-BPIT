from datasets import load_dataset

# Load the 'eval' subset of the ccmusic-database/music_genre dataset
ds = load_dataset("ccmusic-database/music_genre", name="eval")

# Print the structure of one training example
print("Dataset structure:")
print(ds)
print("\n" + "="*50 + "\n")

# Print the keys of the first training example
print("Keys in the first training example:")
print(ds["train"][0].keys())
print("\n" + "="*50 + "\n")

# Print detailed information about the first example
print("First training example details:")
for key in ds["train"][0].keys():
    value = ds["train"][0][key]
    if hasattr(value, 'shape'):
        print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
    else:
        print(f"{key}: {type(value).__name__} = {value}")
