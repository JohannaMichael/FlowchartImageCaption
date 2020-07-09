import json
import os
import skimage.io

flowcharts_symbols_array = []

ROOT_DIR = os.path.abspath("./")
dataset_dir = os.path.join(ROOT_DIR, "FlowchartDataMRCNN/TrainingImages")

annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]
# Add images
for a in annotations:
    single_flowchart_symbol_array = []
    flowchart_symbols = [r['region_attributes'] for r in a['regions']]

    # load_mask() needs the image size to convert polygons to masks.
    # Unfortunately, VIA doesn't include it in JSON, so we must read
    # the image. This is only managable since the dataset is tiny.
    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    for i, p in enumerate(flowchart_symbols):
        print(i)
        print(p)
        # "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
        if p['flowchart_symbols'] == 'terminal_start':
            single_flowchart_symbol_array.insert(i, 1)
        elif p['flowchart_symbols'] == 'flowline':
            single_flowchart_symbol_array.insert(i, 2)
        elif p['flowchart_symbols'] == 'input':
            single_flowchart_symbol_array.insert(i, 3)
        elif p['flowchart_symbols'] == 'decision':
            single_flowchart_symbol_array.insert(i, 4)
        elif p['flowchart_symbols'] == 'process':
            single_flowchart_symbol_array.insert(i, 5)
        elif p['flowchart_symbols'] == 'terminal_end':
            single_flowchart_symbol_array.insert(i, 6)
        elif p['flowchart_symbols'] == 'process_end':
            single_flowchart_symbol_array.insert(i, 7)
        elif p['flowchart_symbols'] == 'process_start':
            single_flowchart_symbol_array.insert(i, 8)
        elif p['flowchart_symbols'] == 'connector':
            single_flowchart_symbol_array.insert(i, 9)
        elif p['flowchart_symbols'] == 'document':
            single_flowchart_symbol_array.insert(i, 10)
        elif p['flowchart_symbols'] == 'terminal':
            single_flowchart_symbol_array.insert(i, 11)
    flowcharts_symbols_array.append(single_flowchart_symbol_array)

print(flowcharts_symbols_array)
