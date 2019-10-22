# Details for calculating the SOA Scores (Semantic Object Accuracy)

## Work with the caption files
To load the captions: load a caption file, get the captions, and generate images:
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
        save("images/label_01_bicycle/my_generated_image.png)    
```

Alternatively, if you're working with the ``captions.pickle`` file from the [AttnGAN](https://github.com/taoxugit/AttnGAN) and their dataloader you can use the provided ``idx`` to load the file directly from the file:
```python
import pickle
import my_model

# load the AttnGAN captions file
with open(captions.pickle, "rb") as f:
    attngan_captions = pickle.load(f)
test_captions = attngan_captions[1]

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

# iterate over the captions and generate three images each
for caption in captions:
    current_caption_idx = caption["idx"]
    # new_ix is the index for the filenames
    new_ix = [current_caption_idx[0]]
    # new_sent_ix is the index to the exact caption, e.g. use it for 
    # caps, cap_len = get_caption(new_sent_ix)
    new_sent_ix = [current_caption_idx[0]*5+current_caption_idx[i][1]]
    for idx in range(3):
        ...
```

For the file ``label_00_person.pkl`` we randomly sample 30,000 captions and generate one image each:
```python
import pickle
import random
import my_model

# load the caption file
with open(label_00_person.pkl, "rb") as f:
    captions = pickle.load(f)

caption_subset = random.sample(captions, 30000)

# iterate over the captions and generate three images each
for caption in caption_subset:
    current_caption = caption["caption"]
    my_generated_image = my_model(current_caption)
    save("images/label_00_person/my_generated_image.png)    
```

## Calculating IoU Scores
For the IoU scores it is important that you use the same label mappings as we (we use the standard mapping). Our labels can be found in ``data/coco.names`` where each label is mapped to the line it is on, i.e. ``person=0, bicycle=1, ...``



instructions about calculating iou values

    1. TODO save dicts
    2. TODO make sure you use the same labeling as we do
    3. run ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
