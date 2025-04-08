import argparse
import json
import os
import yaml
import numpy as np

# Load classnames

with open("object_names.txt") as cls_file:
    classnames = [line.strip() for line in cls_file]

with open("texture_combiation.json") as f:
    texture_dict = json.load(f)

colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
positions = ["left of", "right of", "above", "below"]
shape_adjectives = ["long", "tall", "short", "big", "small"]
spatial_prepositions = ["on the side of", "next to", "near", "on the left of",
                        "on the right of", "on the bottom of", "on the top of"]
numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

prompt_counter = 0
# Proper a vs an

def with_article(name: str):
    return f"an {name}" if name[0] in "aeiou" else f"a {name}"

def unique_id():
    global prompt_counter
    prompt_counter += 1
    return f"{prompt_counter:08d}"

# Proper plural

def make_plural(name: str):
    return f"{name}es" if name.endswith("s") else f"{name}s"

# Generates single object samples

def generate_single_object_sample(rng):
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    return dict(
        id=unique_id(),
        tag="single_object",
        prompt=f"a photo of {with_article(classnames[idx_a])}",
        neg_prompts=[
            f"a photo of {with_article(classnames[idx_b])}"
        ]
    )
    # if size and size > len(classnames):
    #     size = len(classnames)
    #     print(f"Not enough distinct classes, generating only {size} samples")
    # size = size or 1
    # idxs = rng.choice(len(classnames), size=size, replace=False)
    # samples = []
    # for idx in idxs:
    #     neg_idx = rng.choice([i for i in range(len(classnames)) if i != idx])
    #     samples.append(dict(
    #         id=unique_id(),
    #         tag="single_object",
    #         prompt=f"a photo of {with_article(classnames[idx])}",
    #         neg_prompts=[f"a photo of {with_article(classnames[neg_idx])}"]
    #     ))
    # return samples[0] if size == 1 else samples


# Generate two object samples

def generate_two_object_sample(rng):
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    alt_a = rng.choice([i for i in range(len(classnames)) if i != idx_a])
    alt_b = rng.choice([i for i in range(len(classnames)) if i != idx_b])
    return dict(
        id=unique_id(),
        tag="two_object",
        prompt=f"a photo of {with_article(classnames[idx_a])} and {with_article(classnames[idx_b])}",
        neg_prompts=[
            f"a photo of {with_article(classnames[alt_a])} and {with_article(classnames[idx_b])}",
            f"a photo of {with_article(classnames[idx_a])} and {with_article(classnames[alt_b])}"
        ]
    )

# Generate counting samples



def generate_counting_sample(rng, max_count=4):
    idx = rng.choice(len(classnames))
    num = int(rng.integers(2, max_count, endpoint=True))
    negs = [f"a photo of {numbers[i]} {make_plural(classnames[idx])}" for i in range(max_count + 1) if i != num and i <= 10]
    return dict(
        id=unique_id(),
        tag="counting",
        prompt=f"a photo of {numbers[num]} {make_plural(classnames[idx])}",
        neg_prompts=negs
    )

# Generate color samples


def generate_color_sample(rng):
    idx = rng.choice(len(classnames) - 1) + 1
    idx = (idx + classnames.index("person")) % len(classnames)
    color = rng.choice(colors)
    alt_color = rng.choice([c for c in colors if c != color])
    return dict(
        id=unique_id(),
        tag="colors",
        prompt=f"a photo of {with_article(color)} {classnames[idx]}",
        neg_prompts=[
            f"a photo of {with_article(alt_color)} {classnames[idx]}"
        ]
    )


# Generate position samples



def generate_position_sample(rng):
    idx_a, idx_b = rng.choice(len(classnames), size=2, replace=False)
    position = rng.choice(positions)
    alt_pos = rng.choice([p for p in positions if p != position])
    alt_object = rng.choice([i for i in range(len(classnames)) if i != idx_a and i != idx_b])
    return dict(
        id=unique_id(),
        tag="position",
        prompt=f"a photo of {with_article(classnames[idx_a])} {position} {with_article(classnames[idx_b])}",
        neg_prompts=[
            f"a photo of {with_article(classnames[idx_a])} {alt_pos} {with_article(classnames[idx_b])}",
            f"a photo of {with_article(classnames[alt_object])} {position} {with_article(classnames[idx_b])}"
        ]
    )
# Generate color attribution samples

def generate_color_attribution_sample(rng):
    idxs = rng.choice(len(classnames) - 1, size=2, replace=False) + 1
    idx_a, idx_b = (idxs + classnames.index("person")) % len(classnames)
    cidx_a, cidx_b = rng.choice(len(colors), size=2, replace=False)
    alt_cidx = rng.choice([i for i in range(len(colors)) if i != cidx_a])
    return dict(
        id=unique_id(),
        tag="color_attr",
        prompt=f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}",
        neg_prompts=[
            f"a photo of {with_article(colors[alt_cidx])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}",
            f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[alt_cidx])} {classnames[idx_b]}"
        ]
    )


def generate_color_t2i_sample(rng):
    color_a, color_b = rng.choice(colors, size=2, replace=False)
    noun_a, noun_b = rng.choice(classnames, size=2, replace=False)
    neg_color = rng.choice([c for c in colors if c != color_a])
    return dict(
        id=unique_id(),
        tag="color_t2i",
        prompt=f"a {color_a} {noun_a} and a {color_b} {noun_b}",
        neg_prompts=[
            f"a {neg_color} {noun_a} and a {color_b} {noun_b}",
            f"a {color_a} {noun_a} and a {neg_color} {noun_b}"
        ]
    )

def generate_shape_t2i_sample(rng):
    shape_a, shape_b = rng.choice(shape_adjectives, size=2, replace=False)
    noun_a, noun_b = rng.choice(classnames, size=2, replace=False)

    alt_shape_a = rng.choice([s for s in shape_adjectives if s != shape_a])
    alt_shape_b = rng.choice([s for s in shape_adjectives if s != shape_b])
    # alt_noun_a = rng.choice([n for n in classnames if n != noun_a])
    # alt_noun_b = rng.choice([n for n in classnames if n != noun_b])

    return dict(
        id=unique_id(),
        tag="shape_t2i",
        prompt=f"a {shape_a} {noun_a} and a {shape_b} {noun_b}",
        neg_prompts=[
            f"a {alt_shape_a} {noun_a} and a {shape_b} {noun_b}",
            # f"a {shape_a} {alt_noun_a} and a {shape_b} {noun_b}",
            f"a {shape_a} {noun_a} and a {alt_shape_b} {noun_b}"
            # f"a {shape_a} {noun_a} and a {shape_b} {alt_noun_b}"
        ]
    )



def generate_texture_t2i_sample(rng):
    texture_keys = list(texture_dict.keys())
    tex_a, tex_b = rng.choice(texture_keys, size=2, replace=False)
    obj_a = rng.choice(texture_dict[tex_a])
    obj_b = rng.choice(texture_dict[tex_b])

    # Negative textures for a and b
    alt_tex_a = rng.choice([t for t in texture_keys if t != tex_a])
    alt_tex_b = rng.choice([t for t in texture_keys if t != tex_b])

    # Negative objects for a and b (matching correct texture)
    # alt_obj_a = rng.choice([o for o in texture_dict[tex_a] if o != obj_a]) if len(texture_dict[tex_a]) > 1 else obj_a
    # alt_obj_b = rng.choice([o for o in texture_dict[tex_b] if o != obj_b]) if len(texture_dict[tex_b]) > 1 else obj_b

    return dict(
        id=unique_id(),
        tag="texture_t2i",
        prompt=f"a {tex_a.lower()} {obj_a} and a {tex_b.lower()} {obj_b}",
        neg_prompts=[
            f"a {alt_tex_a.lower()} {obj_a} and a {tex_b.lower()} {obj_b}",  # change texture a
            # f"a {tex_a.lower()} {alt_obj_a} and a {tex_b.lower()} {obj_b}",  # change obj a
            f"a {tex_a.lower()} {obj_a} and a {alt_tex_b.lower()} {obj_b}"  # change texture b
            # f"a {tex_a.lower()} {obj_a} and a {tex_b.lower()} {alt_obj_b}"   # change obj b
        ]
    )
    
def generate_spatial_t2i_sample(rng):
    noun_a, noun_b = rng.choice(classnames, size=2, replace=False)
    rel = rng.choice(spatial_prepositions)
    alt_rel = rng.choice([r for r in spatial_prepositions if r != rel])
    return dict(
        id=unique_id(),
        tag="spatial_t2i",
        prompt=f"a {noun_a} {rel} a {noun_b}",
        neg_prompts=[
            f"a {noun_b} {rel} a {noun_a}",
            f"a {noun_a} {alt_rel} a {noun_b}"
        ]
    )

def generate_complex_caption_sample(rng):
    adjective_domains = {
        "shape": shape_adjectives,
        "color": colors,
        "texture": list(texture_dict.keys())
    }
    domain_keys = list(adjective_domains.keys())
    
    adj_type_a, adj_type_b = rng.choice(domain_keys, size=2, replace=False)
    adj_a = rng.choice(adjective_domains[adj_type_a])
    adj_b = rng.choice(adjective_domains[adj_type_b])

    if adj_type_a == "texture":
        noun_a = rng.choice(texture_dict[adj_a])
    else:
        noun_a = rng.choice(classnames)

    if adj_type_b == "texture":
        noun_b = rng.choice(texture_dict[adj_b])
    else:
        noun_b = rng.choice(classnames)

    rel = rng.choice(spatial_prepositions)
    alt_rel = rng.choice([r for r in spatial_prepositions if r != rel])

    alt_adj_a = rng.choice([x for x in adjective_domains[adj_type_a] if x != adj_a])
    alt_adj_b = rng.choice([x for x in adjective_domains[adj_type_b] if x != adj_b])

    return {
        "id": unique_id(),
        "tag": "complex_caption",
        "prompt": f"a {adj_a} {noun_a} {rel} a {adj_b} {noun_b}",
        "neg_prompts": [
            f"a {alt_adj_a} {noun_a} {rel} a {adj_b} {noun_b}",
            f"a {adj_a} {noun_a} {rel} a {alt_adj_b} {noun_b}",
            f"a {adj_a} {noun_a} {alt_rel} a {adj_b} {noun_b}"
        ]
    }
    
# Generate evaluation suite

def generate_suite(rng: np.random.Generator, n: int = 100, output_path: str = ""):
    samples = []
    # Generate single object samples for all COCO classnames
    for _ in range(n):
        samples.append(generate_single_object_sample(rng))

    #GenEval #########################################
    # Generate two object samples (~100)
    for _ in range(n):
        samples.append(generate_two_object_sample(rng))
    # Generate counting samples
    for _ in range(n):
        samples.append(generate_counting_sample(rng, max_count=4))
    # Generate color samples
    for _ in range(n):
        samples.append(generate_color_sample(rng))
    # Generate position samples
    for _ in range(n):
        samples.append(generate_position_sample(rng))
    # Generate color attribution samples
    for _ in range(n):
        samples.append(generate_color_attribution_sample(rng))
        
    # Gnerate T2I samples
    for _ in range(n):
        samples.append(generate_color_t2i_sample(rng))
    for _ in range(n):
        samples.append(generate_texture_t2i_sample(rng))
    for _ in range(n):
        samples.append(generate_shape_t2i_sample(rng))
    for _ in range(n):
        samples.append(generate_spatial_t2i_sample(rng))
    for _ in range(n):
        samples.append(generate_complex_caption_sample(rng))
    ####################################################


    #T2I_compbench #########################################


    # De-duplicate
    unique_samples, used_samples = [], set()
    for sample in samples:
        sample_text = yaml.safe_dump(sample)
        if sample_text not in used_samples:
            unique_samples.append(sample)
            used_samples.add(sample_text)

    # Write to files
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "generation_prompts.txt"), "w") as fp:
        for sample in unique_samples:
            print(sample['prompt'], file=fp)
    with open(os.path.join(output_path, "prompts_metadata.jsonl"), "w") as fp:
        for sample in unique_samples:
            print(json.dumps(sample), file=fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43, help="generation seed (default: 43)")
    parser.add_argument("--num-prompts", "-n", type=int, default=100, help="number of prompts per task (default: 100)")
    parser.add_argument("--output-path", "-o", type=str, default="prompts", help="output folder for prompts and metadata (default: 'prompts/')")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    generate_suite(rng, args.num_prompts, args.output_path)

