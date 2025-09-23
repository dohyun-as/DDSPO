import argparse
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.utils.data

import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
# zsc.tqdm = lambda it, *args, **kwargs: it

from accelerate import Accelerator
import pickle

##############################################################################
# Utilities
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./")

    # Additional mmcv/mmdet style options: e.g. "--options threshold=0.4 max_objects=8"
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()

    # Convert "--options" to dictionary
    args.options = dict(opt.split("=", 1) for opt in args.options)

    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
    return args

def timed(fn):
    """Decorator for timing a function call."""
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

def load_models(config_path, model_path, detector_name, clip_arch, device):
    """Initialize the Mask2Former (or other) detector and CLIP model."""
    # Load object detector
    ckpt_path = os.path.join(model_path, f"{detector_name}.pth")
    object_detector = init_detector(config_path, ckpt_path, device=device)

    # Load CLIP
    clip_model, _, transform = open_clip.create_model_and_transforms(
        clip_arch, pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer(clip_arch)

    # Read COCO names (or your 80-class names) from a local file
    with open("evaluation/geneval/evaluation/object_names.txt") as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, clip_model, transform, tokenizer, classnames

class ImageCrops(torch.utils.data.Dataset):
    """
    A dataset that returns cropped or masked images for zero-shot classification
    of color. If 'bgcolor' is not "original", a background color is applied.
    """
    def __init__(
        self,
        image: Image.Image,
        objects,
        crop: bool = True,
        bgcolor: str = "#999",
        transform=None
    ):
        self._image = image.convert("RGB")
        self._bgcolor = bgcolor
        self._crop = crop
        self._transform = transform

        if self._bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=self._bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            # composite the original image over the blank background using the mask
            image = Image.composite(
                self._image,
                self._blank,
                Image.fromarray(mask)
            )
        else:
            image = self._image

        if self._crop:
            # box is [x1,y1,x2,y2,...]; use the first 4 to crop
            image = image.crop(box[:4])

        if self._transform is not None:
            image = self._transform(image)

        return (image, 0)

def color_classification(
    image, bboxes, classname,
    clip_model, transform, tokenizer,
    colors, device,
    crop=True, bgcolor="#999"
):
    """
    Classify color of each object using zero-shot classification.
    """
    # Build a zero-shot classifier for a given class on all known colors
    prompts = [
        f"a photo of a {{c}} {classname}",
        f"a photo of a {{c}}-colored {classname}",
        f"a photo of a {{c}} object"
    ]
    if classname not in color_classification._cache:
        clf = zsc.zero_shot_classifier(
            clip_model, tokenizer, colors, prompts, device
        )
        color_classification._cache[classname] = clf
    else:
        clf = color_classification._cache[classname]

    dataset = ImageCrops(
        image, bboxes,
        crop=crop,
        bgcolor=bgcolor,
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4)

    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, device)
        return [colors[index.item()] for index in pred.argmax(dim=1)]

# Attach a "static" cache so we don't rebuild the classifier for the same class
color_classification._cache = {}

def compute_iou(box_a, box_b):
    """IoU between two bounding boxes, each in [x1,y1,x2,y2,score]."""
    def area_fn(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    # Intersection
    inter_box = [
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3])
    ]
    i_area = area_fn(inter_box)
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    if u_area == 0:
        return 0
    return i_area / u_area

def relative_position(obj_a, obj_b, pos_threshold=0.1):
    """
    Return the set of directional relations (e.g. "left of", "right of", etc.)
    that A has relative to B, factoring in some threshold of bounding-box size.
    """
    box_a, box_b = obj_a[0], obj_b[0]  # each is [x1,y1,x2,y2,score]
    # [x1,y1,x2,y2] => center + dimensions
    axc = 0.5*(box_a[0] + box_a[2])
    ayc = 0.5*(box_a[1] + box_a[3])
    aw = abs(box_a[2] - box_a[0])
    ah = abs(box_a[3] - box_a[1])

    bxc = 0.5*(box_b[0] + box_b[2])
    byc = 0.5*(box_b[1] + box_b[3])
    bw = abs(box_b[2] - box_b[0])
    bh = abs(box_b[3] - box_b[1])

    offset = np.array([axc - bxc, ayc - byc], dtype=np.float32)
    dim_sum = np.array([aw + bw, ah + bh], dtype=np.float32)
    # "shrink" offset by threshold * dim_sum, to require clearly-lateral positions
    revised_offset = np.maximum(np.abs(offset) - pos_threshold * dim_sum, 0) * np.sign(offset)

    if np.all(np.abs(revised_offset) < 1e-3):
        # no strong directional difference
        return set()

    dx, dy = revised_offset
    norm = np.linalg.norm([dx, dy]) + 1e-8
    dx, dy = dx/norm, dy/norm

    rels = set()
    if dx < -0.5:
        rels.add("left of")
    elif dx > 0.5:
        rels.add("right of")
    if dy < -0.5:
        rels.add("above")
    elif dy > 0.5:
        rels.add("below")
    return rels

def evaluate(
    image, objects, metadata,
    clip_model, transform, tokenizer,
    colors, device,
    threshold_cfg
):
    """
    Evaluate an image against 'metadata' constraints:
    - 'include' clauses must all be satisfied (AND)
    - 'exclude' clauses must all be satisfied (AND)
    """
    correct = True
    reason = []
    matched_groups = []

    # Evaluate "include" clauses
    for req in metadata.get('include', []):
        classname = req['class']
        req_count = req['count']
        matched = True

        # Objects for this class
        found_objects = objects.get(classname, [])[:req_count]
        if len(found_objects) < req_count:
            correct = matched = False
            reason.append(
                f"expected {classname}>={req_count}, found {len(found_objects)}"
            )
        else:
            # Color check
            if 'color' in req:
                c = req['color']
                colors_pred = color_classification(
                    image, found_objects, classname,
                    clip_model, transform, tokenizer,
                    colors, device,
                    crop=bool(threshold_cfg.get('crop', True)),
                    bgcolor=threshold_cfg.get('bgcolor', "#999")
                )
                if colors_pred.count(c) < req_count:
                    correct = matched = False
                    reason.append(
                        f"expected {c} {classname}>={req_count}, found {colors_pred.count(c)} {c}; " +
                        "counts: " + ", ".join(
                            f"{col}={colors_pred.count(col)}" for col in colors
                            if col in colors_pred
                        )
                    )

            # Position check
            if 'position' in req and matched:
                # e.g., "position": ["above", 1], meaning "above group index 1"
                expected_rel, target_group_idx = req['position']
                if (target_group_idx >= len(matched_groups)) or (matched_groups[target_group_idx] is None):
                    correct = matched = False
                    reason.append(f"no valid target group for {classname} to be {expected_rel}")
                else:
                    for obj_a in found_objects:
                        for obj_b in matched_groups[target_group_idx]:
                            rels = relative_position(obj_a, obj_b, threshold_cfg['pos_threshold'])
                            if expected_rel not in rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} to be {expected_rel} target; found {rels}"
                                )
                                break
                        if not matched:
                            break

        # If matched, store in matched_groups
        matched_groups.append(found_objects if matched else None)

    # Evaluate "exclude" clauses
    for req in metadata.get('exclude', []):
        classname = req['class']
        req_count = req['count']  # i.e. "fewer than 'count'"
        nfound = len(objects.get(classname, []))
        if nfound >= req_count:
            correct = False
            reason.append(
                f"expected {classname}<{req_count}, found {nfound}"
            )

    return correct, "\n".join(reason)

def evaluate_image(
    filepath,
    metadata,
    object_detector,
    classnames,
    clip_model,
    transform,
    tokenizer,
    colors,
    device,
    threshold_cfg
):
    """
    Run detection -> run NMS -> evaluate clauses -> return final result
    """
    # Inference
    result = inference_detector(object_detector, filepath)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None

    # Load image (with orientation fix)
    image = ImageOps.exif_transpose(Image.open(filepath))

    # Filter and store objects by class
    # If "tag" is "counting", we might apply a stricter threshold
    conf_thresh = (
        threshold_cfg['counting_threshold']
        if metadata['tag'] == 'counting' else
        threshold_cfg['score_threshold']
    )

    detected = {}
    for cls_idx, classname in enumerate(classnames):
        # bounding boxes for this class
        cls_bboxes = bbox[cls_idx]
        # Sort by descending confidence
        order = np.argsort(cls_bboxes[:, 4])[::-1]
        # Filter by confidence threshold
        order = order[cls_bboxes[order, 4] > conf_thresh]

        max_objects = threshold_cfg['max_objects']
        order = order[:max_objects].tolist()

        # NMS pass (if desired)
        keeped = []
        while order:
            best = order.pop(0)
            keeped.append(best)
            order = [
                idx for idx in order
                if threshold_cfg['nms_threshold'] == 1.0
                   or compute_iou(cls_bboxes[best], cls_bboxes[idx]) < threshold_cfg['nms_threshold']
            ]
        if not keeped:
            continue

        final_objs = []
        for b in keeped:
            mask = segm[cls_idx][b] if segm is not None else None
            final_objs.append((cls_bboxes[b], mask))

        if final_objs:
            detected[classname] = final_objs

    # Evaluate
    is_correct, reason = evaluate(
        image, detected, metadata,
        clip_model, transform, tokenizer,
        colors, device,
        threshold_cfg
    )
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        }),
    }

def gather_list_of_dicts(accelerator, data_list):
    """
    Helper to gather a list of dictionaries from all processes using `accelerator`.
    Since `accelerator.gather` only handles tensors, we pickle to bytes and then gather.
    Returns a single, merged list (on every process). If you only want it on the main
    process, you can adjust accordingly.
    """
    import torch

    # Serialize the data
    pickled = pickle.dumps(data_list)
    data_tensor = torch.ByteTensor(list(pickled)).to(accelerator.device)
    length = torch.tensor([data_tensor.shape[0]], dtype=torch.long, device=accelerator.device)

    # Collect lengths from all ranks
    lengths = accelerator.gather(length)
    max_length = int(lengths.max().item())

    # Pad each rank's data_tensor to max_length
    padded_data = torch.zeros((1, max_length), dtype=torch.uint8, device=accelerator.device)
    padded_data[0, :data_tensor.shape[0]] = data_tensor

    # Gather across all ranks (shape => [world_size, max_length])
    gathered = accelerator.gather(padded_data)
    
    # if accelerator.num_processes == 1:
    #     gathered = gathered.unsqueeze(0)

    # On each rank, reconstruct the data from each rank
    all_data = []
    for i, l in enumerate(lengths):
        part = gathered[i, :l]
        # Convert to list, then to bytes
        recon = bytes(part.tolist())
        # Unpickle
        partial_list = pickle.loads(recon)
        all_data.extend(partial_list)

    return all_data

@timed
def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    # Hyper-params from command-line
    # Some defaults; override if specified in --options
    detector_name  = args.options.get('model', 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco')
    clip_arch      = args.options.get('clip_model', 'ViT-L-14')
    score_threshold = float(args.options.get('threshold', 0.3))
    counting_threshold = float(args.options.get('counting_threshold', 0.9))
    max_objects    = int(args.options.get('max_objects', 16))
    nms_threshold  = float(args.options.get('max_overlap', 1.0))
    pos_threshold  = float(args.options.get('position_threshold', 0.1))
    crop_flag      = (args.options.get('crop', '1') == '1')
    bgcolor        = args.options.get('bgcolor', '#999')

    threshold_cfg = {
        'score_threshold': score_threshold,
        'counting_threshold': counting_threshold,
        'max_objects': max_objects,
        'nms_threshold': nms_threshold,
        'pos_threshold': pos_threshold,
        'crop': crop_flag,
        'bgcolor': bgcolor
    }

    # Load models (locally in each process)
    object_detector, clip_model, transform, tokenizer, classnames = load_models(
        config_path=args.model_config,
        model_path=args.model_path,
        detector_name=detector_name,
        clip_arch=clip_arch,
        device=device
    )

    # Colors for color classification
    COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]

    # Identify which subfolders belong to this rank
    all_subfolders = os.listdir(args.imagedir)
    my_subfolders = []
    for sf in all_subfolders:
        match = re.match(r"(\d{5})", sf)
        if match:
            index = int(match.group(1))
            if index % accelerator.num_processes == accelerator.process_index:
                folder_path = os.path.join(args.imagedir, sf)
                if os.path.isdir(folder_path):
                    my_subfolders.append(sf)

    partial_results = []
    for subfolder in my_subfolders:
        folderpath = os.path.join(args.imagedir, subfolder)
        metafile = os.path.join(folderpath, "metadata.jsonl")
        if not os.path.isfile(metafile):
            continue
        with open(metafile, "r") as fp:
            metadata = json.load(fp)

        # Evaluate each image in the "samples" subdir
        sample_dir = os.path.join(folderpath, "samples")
        if not os.path.isdir(sample_dir):
            continue

        for imagename in os.listdir(sample_dir):
            if not re.match(r"\d+\.png", imagename):
                continue
            imagepath = os.path.join(sample_dir, imagename)
            if not os.path.isfile(imagepath):
                continue

            result = evaluate_image(
                imagepath,
                metadata,
                object_detector,
                classnames,
                clip_model,
                transform,
                tokenizer,
                COLORS,
                device,
                threshold_cfg
            )
            partial_results.append(result)

    # Gather partial_results from all ranks to each process (you could also only do it to rank 0).
    # For large result sets, you may prefer to gather only on rank 0. Here's how to gather on each rank:
    full_results = gather_list_of_dicts(accelerator, partial_results)

    # Only rank 0 (main process) writes out the results to file
    if accelerator.is_main_process:
        if os.path.dirname(args.outfile):
            os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

        df = pd.DataFrame(full_results)
        with open(args.outfile, "w") as fp:
            df.to_json(fp, orient="records", lines=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
