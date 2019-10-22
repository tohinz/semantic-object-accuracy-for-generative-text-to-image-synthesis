# Details for calculating the SOA Scores

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
        save("images/label_01_bicycle/my_generated_image_{}.png".format(idx))  
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

In order to calculate the IoU scores you need to save the "ground truth" information, i.e. the bounding boxes you give your model as input, so we can compare them with the bounding boxes from the detection network.
We expect the information about the bounding boxes as a pickle file which is a dictionary of the form
```python
output_dict = {"name_of_the_generated_image": [[], [label_int], [bbox]],
               ...}
# for example:
output_dict = {"my_generated_image": [[], [1, 1], [[0.1, 0.1, 0.3, 0.5], [0.6, 0.2, 0.2, 0.4]]]}
```
Here, ``label_int`` is a list of the integer labels you use as conditioning (e.g. ``person=0, bicycle=1, ...``) and ``bbox`` 
is a list of the bounding boxes ``[x, y, width, height]`` where the values are normalized to be between ``[0,1]`` and the coordinate system starts at the top left corner of the image, i.e. a bounding box of ``[0, 0, 0.5, 0.5]`` covers the top left quarter of the image.
The ``output_dict`` should be saved in the same folder as the images for which it was created.

```python
import pickle
import my_model

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

# this is the dictionary we use to save the bounding boxes
output_dict = {}

# iterate over the captions and generate three images each
for caption in captions:
    current_caption = caption["caption"]
    for idx in range(3):
        my_generated_image = my_model(current_caption)
        save("images/label_01_bicycle/my_generated_image_{}.png".format(idx))    
        # label_int is a list of the integer values for labels you used as input to the network
        # bbox is a list with the corresponding bounding boxes [x, y, width, height]
        # e.g. label_int = [1, 1]
        #      bbox = [[0.1, 0.1, 0.3, 0.5], [0.6, 0.2, 0.2, 0.4]]
        output_dict["my_generated_image_{}.png".format(idx)] = [[], label_int, bbox]
        
with open("images/label_01_bicycle/ground_truth_label_01_bicycle.pkl", "wb") as f:
    pickle.dump(output_dict, f)
```

Finally, you should have the 80 folders with images as before, but now each folder should also contain a ``.pkl`` file with the ground truth information of the given layout.
Run the same command as before but with the ``--iou`` flag: ``python calculate_soa.py --images path/to/folder/created-in-first-step --output path/to/folder/where-results-are-saved --gpu 0 --iou``
