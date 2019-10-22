# Details for calculating the SOA Scores (Semantic Object Accuracy)

1. To load the captions: load a caption file, get the captions, and generate images
```python
import pickle
import my_model

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

# iterate over the captions and generate three images each
for caption in captions:
    current_caption = caption["caption"]
    for idx in range(3):
        my_generated_image = my_model(current_caption)
        save("images/label_01_bicycle/my_generated_image)    
```

class labels -> make sure they're right



how to load and process a caption file



instructions about calculating iou values

    1. TODO save dicts
    2. TODO make sure you use the same labeling as we do
    3. run ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
