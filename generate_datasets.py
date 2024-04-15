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

color = {
	"red": distribution.Mixture([distribution.Continuous("c0", 0.95, 1.0), distribution.Continuous("c0", 0.0, 0.05)]),
	"yellow": distribution.Continuous("c0", 0.10, 0.18),
	"green": distribution.Continuous("c0", 0.27, 0.37),
	"blue": distribution.Continuous("c0", 0.55, 0.65),
}

saturation = distribution.Continuous("c1", 0.5, 1.0)
brightness = distribution.Continuous("c2", 0.9, 1.0)

position = {
	"top_left": distribution.Product(
		[distribution.Continuous("y", 0.65, 0.85), distribution.Continuous("x", 0.15, 0.35)]),
	"top_right": distribution.Product(
		[distribution.Continuous("y", 0.65, 0.85), distribution.Continuous("x", 0.65, 0.85)]),
	"bottom_right": distribution.Product(
		[distribution.Continuous("y", 0.15, 0.35), distribution.Continuous("x", 0.65, 0.85)]),
	"bottom_left": distribution.Product(
		[distribution.Continuous("y", 0.15, 0.35), distribution.Continuous("x", 0.15, 0.35)]),
}

renderer = PILRenderer(anti_aliasing=2, color_to_rgb=color_maps.hsv_to_rgb)

#%%

def distribute_three(colors: np.ndarray) -> bool:
	return np.unique(colors).size == 3


def progression(positions: np.ndarray) -> bool:
	length = len(position)
	return (positions[0] + 1) % length == positions[1] and (positions[1] + 1) % length == positions[2]

#%%

def sample_color(correct: bool):
	if correct:
		return np.random.choice(list(color.keys()), 3, replace=False)
	else:
		x = np.random.choice(list(color.keys()), 3, replace=True)
		while distribute_three(x):
			x = np.random.choice(list(color.keys()), 3, replace=True)
		return x

def sample_position(correct: bool):
	if correct:
		start = np.random.randint(0, len(position))
		return np.array([list(position.keys())[i % len(position)] for i in range(start, start + 3)])
	else:
		length = len(position)
		x = np.random.randint(0, length, 3)
		while progression(x):
			x = np.random.randint(0, length, 3)
		return np.array([list(position.keys())[i] for i in x])

#%%

def create_balanced_dataset(name: str, samples: int):
	Path(name).mkdir(parents=True, exist_ok=True)

	puzzles = pd.DataFrame(index=range(samples),
						   columns=["correct"] + [f"{type}_{i}" for i in range(9) for type in ["color", "position"]])
	color_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	position_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	images = pd.DataFrame(index=range(samples * 9), columns=["color", "position"])

	for i in range(samples):
		correct_puzzle = np.random.rand() < 0.5

		puzzles.loc[i]["correct"] = correct_puzzle

		colors = np.stack([sample_color(correct_puzzle) for _ in range(3)], axis=0).flatten()
		positions = np.stack([sample_position(correct_puzzle) for _ in range(3)], axis=1).flatten()
  
		color_relations.loc[i * 3:(i + 1) * 3 - 1] = correct_puzzle
		position_relations.loc[i * 3:(i + 1) * 3 - 1] = correct_puzzle


		puzzles.loc[i][["color_" + str(j) for j in range(9)]] = colors
		puzzles.loc[i][["position_" + str(j) for j in range(9)]] = positions
		images.loc[i * 9:(i + 1) * 9 - 1] = np.stack([colors, positions], axis=1)

		for j in range(9):
			instance = distribution.Product([
				color[colors[j]],
				distribution.Continuous("scale", 0.1, 0.3),
				position[positions[j]],
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
	
def create_independent_dataset(name: str, samples: int):
	Path(name).mkdir(parents=True, exist_ok=True)

	puzzles = pd.DataFrame(index=range(samples),
						   columns=["correct"] + [f"{type}_{i}" for i in range(9) for type in
												  ["color", "position"]])
	color_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	position_relations = pd.DataFrame(index=range(samples * 3), columns=["correct"])
	images = pd.DataFrame(index=range(samples * 9), columns=["color", "position"])

	probability = 0.22

	for i in range(samples):
		correct_puzzle = True

		colors = []
		for j in range(3):
			color_correct = np.random.rand() >= probability
   
			colors.append(sample_color(color_correct))
			color_relations.loc[i * 3 + j] = color_correct

		positions = []
		for j in range(3):
			position_correct = np.random.rand() >= probability

			positions.append(sample_position(position_correct))
			position_relations.loc[i * 3 + j] = position_correct

		colors = np.stack(colors, axis=0).flatten()
		positions = np.stack(positions, axis=1).flatten()

		puzzles.loc[i][["color_" + str(j) for j in range(9)]] = colors
		puzzles.loc[i][["position_" + str(j) for j in range(9)]] = positions
		images.loc[i * 9:(i + 1) * 9 - 1] = np.stack([colors, positions], axis=1)

		puzzles.loc[i]["correct"] = correct_puzzle

		for j in range(9):
			instance = distribution.Product([
				color[colors[j]],
				distribution.Continuous("scale", 0.1, 0.3),
				position[positions[j]],
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


def create_datasets(path: str):
	create_balanced_dataset(path + "/train", 3000)
	create_balanced_dataset(path + "/val", 300)
	create_balanced_dataset(path + "/test", 300)

	create_independent_dataset(path + "/train", 3000)
	create_independent_dataset(path + "/val", 300)
	create_independent_dataset(path + "/test", 300)
 
 
def inspect_dataset(path: str):
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
	create_datasets("data/blackbird")
 
	print("Balanced dataset:")
	inspect_dataset("data/blackbird/balanced/train")
 
	print("Independent dataset:")
	inspect_dataset("data/blackbird/independent/train")

