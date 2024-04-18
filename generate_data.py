#%%

import os
from pathlib import Path

import numpy as np
import pandas as pd
import spriteworld.factor_distributions as distribution
from matplotlib import pyplot as plt
from PIL import Image
from skimage.io import imread_collection
from spriteworld.renderers import PILRenderer, color_maps
from spriteworld.sprite import Sprite

#%%

rainbow_colors = {
	"red": distribution.Mixture([distribution.Continuous("c0", 0.95, 1.0), distribution.Continuous("c0", 0.0, 0.05)]),
	"orange": distribution.Continuous("c0", 0.05, 0.10),
	"yellow": distribution.Continuous("c0", 0.10, 0.18),
	"green": distribution.Continuous("c0", 0.27, 0.37),
	"blue": distribution.Continuous("c0", 0.55, 0.65),
	"indigo": distribution.Continuous("c0", 0.68, 0.75),
	"violet": distribution.Continuous("c0", 0.75, 0.85),
}
shapes_colors = {k: v for k, v in rainbow_colors.items() if k in ["red", "green", "blue"]}
blackbird_colors = {k: v for k, v in rainbow_colors.items() if k in ["red", "yellow", "green", "blue"]}

blackbird_positions = {
	"top_left": distribution.Product(
		[distribution.Continuous("y", 0.65, 0.85), distribution.Continuous("x", 0.15, 0.35)]),
	"top_right": distribution.Product(
		[distribution.Continuous("y", 0.65, 0.85), distribution.Continuous("x", 0.65, 0.85)]),
	"bottom_right": distribution.Product(
		[distribution.Continuous("y", 0.15, 0.35), distribution.Continuous("x", 0.65, 0.85)]),
	"bottom_left": distribution.Product(
		[distribution.Continuous("y", 0.15, 0.35), distribution.Continuous("x", 0.15, 0.35)]),
}

positions = {
	"top": distribution.Product([distribution.Continuous("y", 0.56, 0.74), distribution.Discrete("x", [0.5])]),
	"center": distribution.Product([distribution.Continuous("y", 0.38, 0.56), distribution.Discrete("x", [0.5])]),
	"bottom": distribution.Product([distribution.Continuous("y", 0.2, 0.38), distribution.Discrete("x", [0.5])]),
}

shapes = {
	"triangle",
	"square",
	"circle",
}

sizes = {
	"small": distribution.Continuous("scale", 0.1, 0.17),
	"medium": distribution.Continuous("scale", 0.17, 0.23),
	"large": distribution.Continuous("scale", 0.23, 0.3),
}


saturation = distribution.Continuous("c1", 0.5, 1.0)
brightness = distribution.Continuous("c2", 0.9, 1.0)

renderer = PILRenderer(anti_aliasing=2, color_to_rgb=color_maps.hsv_to_rgb)

#%%

def distribute_three(colors: np.ndarray) -> bool:
	return np.unique(colors).size == 3


def progression(positions: np.ndarray) -> bool:
	length = len(blackbird_positions)
	return (positions[0] + 1) % length == positions[1] and (positions[1] + 1) % length == positions[2]

def sample_color(correct: bool):
	if correct:
		return np.random.choice(list(blackbird_colors.keys()), 3, replace=False)
	else:
		x = np.random.choice(list(blackbird_colors.keys()), 3, replace=True)
		while distribute_three(x):
			x = np.random.choice(list(blackbird_colors.keys()), 3, replace=True)
		return x

def sample_position(correct: bool):
	if correct:
		start = np.random.randint(0, len(blackbird_positions))
		return np.array([list(blackbird_positions.keys())[i % len(blackbird_positions)] for i in range(start, start + 3)])
	else:
		length = len(blackbird_positions)
		x = np.random.randint(0, length, 3)
		while progression(x):
			x = np.random.randint(0, length, 3)
		return np.array([list(blackbird_positions.keys())[i] for i in x])

#%%

def create_blackbird_dataset(name: str, samples: int, type: str):
	Path(name).mkdir(parents=True, exist_ok=True)

	puzzles = pd.DataFrame(index=range(samples),
						   columns=["correct"] + [f"{type}_{i}" for i in range(9) for type in ["color", "position"]])
	color_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	position_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	images = pd.DataFrame(index=range(samples * 9), columns=["color", "position"])
 

	for i in range(samples):
		correct_puzzle = np.random.rand() < 0.5 if type == "balanced" else True

		match type:
			case "balanced":
				colors = np.stack([sample_color(correct_puzzle) for _ in range(3)], axis=0).flatten()
				color_relations.loc[i * 3:(i + 1) * 3 - 1, "correct"] = correct_puzzle

				positions = np.stack([sample_position(correct_puzzle) for _ in range(3)], axis=1).flatten()
				position_relations.loc[i * 3:(i + 1) * 3 - 1, "correct"] = correct_puzzle
			case "independent":
				probability = 0.22
				colors = []
				for j in range(3):
					color_correct = np.random.rand() >= probability
		
					colors.append(sample_color(color_correct))
					color_relations.loc[i * 3 + j] = color_correct
					correct_puzzle = correct_puzzle and color_correct
				colors = np.stack(colors, axis=0).flatten()
	
				positions = []
				for j in range(3):
					position_correct = np.random.rand() >= probability

					positions.append(sample_position(position_correct))
					position_relations.loc[i * 3 + j] = position_correct
					correct_puzzle = correct_puzzle and position_correct
				positions = np.stack(positions, axis=1).flatten()

		puzzles.loc[i, ["color_" + str(j) for j in range(9)]] = colors
		puzzles.loc[i, ["position_" + str(j) for j in range(9)]] = positions
		images.loc[i * 9:(i + 1) * 9 - 1,] = np.stack([colors, positions], axis=1)
  
		puzzles.loc[i, "correct"] = correct_puzzle

		for j in range(9):
			instance = distribution.Product([
				blackbird_colors[colors[j]],
				distribution.Continuous("scale", 0.1, 0.3),
				blackbird_positions[positions[j]],
				saturation,
				brightness,
			]).sample()
			instance["shape"] = "circle"

			sprite = Sprite(**instance)

			image = renderer.render([sprite])
			Image.fromarray(image).save(os.path.join(name, f"{i * 9 + j}.png"))

	puzzles.to_csv(os.path.join(name, "blackbird.csv"), index=False)
	color_relations.to_csv(os.path.join(name, "distribute_three.csv"), index=False)
	position_relations.to_csv(os.path.join(name, "progression.csv"), index=False)
	images.to_csv(os.path.join(name, "product_concepts.csv"), index=False)

#%%

def create_dataset(name: str, samples: int, type: str):
	Path(name).mkdir(parents=True, exist_ok=True)

	match type:
		case "shapes":
			colors = shapes_colors
			table = pd.DataFrame(columns=["color", "shape", "size", "position"])
		case "rainbow":
			colors = rainbow_colors
			table = pd.DataFrame(columns=["color"])
	
	labels = distribution.Product([distribution.Discrete("color", list(colors.keys())),
								   distribution.Discrete("shape", list(shapes)),
								   distribution.Discrete("size", list(sizes.keys())),
								   distribution.Discrete("position", list(positions.keys()))])

	for i in range(samples):
		label = labels.sample()
		table.loc[i] = label

		instance = distribution.Product([
			colors[label["color"]],
			sizes[label["size"]],
			positions[label["position"]],
			saturation,
			brightness]
		).sample()
		instance["shape"] = label["shape"]

		sprite = Sprite(**instance)

		image = renderer.render([sprite])
		Image.fromarray(image).save(os.path.join(name, f"{i}.png"))

	table.to_csv(os.path.join(name, "product_concepts.csv"), index=False)

#%%

def create_datasets(path: str):
	print("Creating shapes training dataset")
	create_dataset(path + "/shapes/train", 3000, "shapes")
	print("Creating shapes validation dataset")
	create_dataset(path + "/shapes/val", 300, "shapes")
	print("Creating shapes test dataset")
	create_dataset(path + "/shapes/test", 300, "shapes")
	
	print("Creating rainbow training dataset")
	create_dataset(path + "/rainbow/train", 3000, "rainbow")
	print("Creating rainbow validation dataset")
	create_dataset(path + "/rainbow/val", 300, "rainbow")
	print("Creating rainbow test dataset")
	create_dataset(path + "/rainbow/test", 300, "rainbow")
	
	print("Creating balanced training dataset")
	create_blackbird_dataset(path + "/balanced/train", 3000, "balanced")
	print("Creating balanced validation dataset")
	create_blackbird_dataset(path + "/balanced/val", 300, "balanced")
	print("Creating balanced test dataset")
	create_blackbird_dataset(path + "/balanced/test", 300, "balanced")
 
	print("Creating independent training dataset")
	create_blackbird_dataset(path + "/independent/train", 3000, "independent")
	print("Creating independent validation dataset")
	create_blackbird_dataset(path + "/independent/val", 300, "independent")
	print("Creating independent test dataset")
	create_blackbird_dataset(path + "/independent/test", 300, "independent")
 
def display_puzzle(path: str, i: int):
	images = imread_collection([os.path.join(path, f"{i}.png") for i in range(i * 9, (i + 1) * 9)],
							   conserve_memory=False)
	fig, ax = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={"wspace": 0.02, "hspace": 0.02}, squeeze=True)
	for i in range(3):
		for j in range(3):
			ax[i, j].imshow(images[i * 3 + j], aspect="auto")
			ax[i, j].axis("off")
	# plt.savefig("graphs/" + name, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
	plt.show()
 
def inspect_blackbird_dataset(path: str):
	puzzles = pd.read_csv(path + "/blackbird.csv")
	distribute_three = pd.read_csv(path + "/distribute_three.csv")
	progression = pd.read_csv(path + "/progression.csv")
 
	mean_puzzles = np.mean(puzzles["correct"])
	mean_distribute_three = np.mean(distribute_three["correct"])
	mean_progression = np.mean(progression["correct"])
 
	print(f"Dataset: {mean_puzzles} correct puzzles, {mean_distribute_three} correct distribute three relations, {mean_progression} correct progression relations")
 
	display_puzzle(path, 0)

#%%

if __name__ == "__main__":
	directory = "data"
	create_datasets(directory)
 
	print("Balanced dataset:")
	inspect_blackbird_dataset(directory + "/balanced/train")
 
	print("Independent dataset:")
	inspect_blackbird_dataset(directory + "/independent/train")


# %%
