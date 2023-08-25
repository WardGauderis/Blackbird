import os
from pathlib import Path

import spriteworld.factor_distributions as distribution
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread_collection
from spriteworld.renderers import PILRenderer, color_maps
from spriteworld.sprite import Sprite
import pandas as pd

import numpy as np

color = {
	"red": distribution.Mixture([distribution.Continuous("c0", 0.95, 1.0), distribution.Continuous("c0", 0.0, 0.05)]),
	# "orange": distribution.Continuous("c0", 0.05, 0.10),
	"yellow": distribution.Continuous("c0", 0.10, 0.18),
	"green": distribution.Continuous("c0", 0.27, 0.37),
	"blue": distribution.Continuous("c0", 0.55, 0.65),
	# "indigo": distribution.Continuous("c0", 0.68, 0.75),
	# "violet": distribution.Continuous("c0", 0.75, 0.85),
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


def check_color(colors: np.ndarray):
	return np.unique(colors).size == 3


def check_position(positions: np.ndarray):
	length = len(position)
	return (positions[0] + 1) % length == positions[1] and (positions[1] + 1) % length == positions[2]


def sample_color_correct():
	return np.random.choice(list(color.keys()), 3, replace=False)


def sample_color_incorrect():
	x = np.random.choice(list(color.keys()), 3, replace=True)
	while check_color(x):
		x = np.random.choice(list(color.keys()), 3, replace=True)
	return x


def sample_position_correct():
	start = np.random.randint(0, len(position))
	return np.array([list(position.keys())[i % len(position)] for i in range(start, start + 3)])


def sample_position_incorrect():
	length = len(position)
	x = np.random.randint(0, length, 3)
	while check_position(x):
		x = np.random.randint(0, length, 3)
	return np.array([list(position.keys())[i] for i in x])


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

		if correct_puzzle:
			colors = np.stack([sample_color_correct() for _ in range(3)], axis=0).flatten()
			positions = np.stack([sample_position_correct() for _ in range(3)], axis=1).flatten()
			color_relations.loc[i * 3:(i + 1) * 3 - 1] = True
			position_relations.loc[i * 3:(i + 1) * 3 - 1] = True
		else:
			colors = np.stack([sample_color_incorrect() for _ in range(3)], axis=0).flatten()
			positions = np.stack([sample_position_incorrect() for _ in range(3)], axis=1).flatten()
			color_relations.loc[i * 3:(i + 1) * 3 - 1] = False
			position_relations.loc[i * 3:(i + 1) * 3 - 1] = False

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

	puzzles.to_csv(os.path.join(name, "puzzles.csv"), index=False)
	color_relations.to_csv(os.path.join(name, "color_relations.csv"), index=False)
	position_relations.to_csv(os.path.join(name, "position_relations.csv"), index=False)
	images.to_csv(os.path.join(name, "images.csv"), index=False)


def generate_balanced_datasets():
	Path("datasets/balanced").mkdir(parents=True, exist_ok=True)

	create_balanced_dataset("datasets/balanced/train", 3000)
	create_balanced_dataset("datasets/balanced/val", 300)
	create_balanced_dataset("datasets/balanced/test", 300)
	
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
			if np.random.rand() >= probability:
				colors.append(sample_color_correct())
				color_relations.loc[i * 3 + j] = True
			else:
				correct_puzzle = False
				colors.append(sample_color_incorrect())
				color_relations.loc[i * 3 + j] = False

		positions = []
		for j in range(3):
			if np.random.rand() >= probability:
				positions.append(sample_position_correct())
				position_relations.loc[i * 3 + j] = True
			else:
				correct_puzzle = False
				positions.append(sample_position_incorrect())
				position_relations.loc[i * 3 + j] = False

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

	puzzles.to_csv(os.path.join(name, "puzzles.csv"), index=False)
	color_relations.to_csv(os.path.join(name, "color_relations.csv"), index=False)
	position_relations.to_csv(os.path.join(name, "position_relations.csv"), index=False)
	images.to_csv(os.path.join(name, "images.csv"), index=False)


def generate_independent_datasets():
	Path("datasets/independent").mkdir(parents=True, exist_ok=True)

	create_independent_dataset("datasets/independent/train", 3000)
	create_independent_dataset("datasets/independent/val", 300)
	create_independent_dataset("datasets/independent/test", 300)


def display_puzzle(path: str, i: int, name: str):
	images = imread_collection([os.path.join(path, f"{i}.png") for i in range(i * 9, (i + 1) * 9)],
	                           conserve_memory=False)
	fig, ax = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={"wspace": 0.02, "hspace": 0.02}, squeeze=True)
	for i in range(3):
		for j in range(3):
			ax[i, j].imshow(images[i * 3 + j], aspect="auto")
			ax[i, j].axis("off")
	plt.savefig("graphs/" + name, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
	plt.show()





if __name__ == "__main__":
	generate_balanced_datasets()
	generate_independent_datasets()

	puzzles = pd.read_csv("datasets/balanced/train/puzzles.csv")
	color_relations = pd.read_csv("datasets/balanced/train/color_relations.csv")
	position_relations = pd.read_csv("datasets/balanced/train/position_relations.csv")
	mean_puzzles = np.mean(puzzles["correct"])
	mean_color_relations = np.mean(color_relations["correct"])
	mean_position_relations = np.mean(position_relations["correct"])
	print(mean_puzzles, mean_color_relations, mean_position_relations)

	puzzles = pd.read_csv("datasets/independent/train/puzzles.csv")
	color_relations = pd.read_csv("datasets/independent/train/color_relations.csv")
	position_relations = pd.read_csv("datasets/independent/train/position_relations.csv")
	mean_puzzles = np.mean(puzzles["correct"])
	mean_color_relations = np.mean(color_relations["correct"])
	mean_position_relations = np.mean(position_relations["correct"])
	print(mean_puzzles, mean_color_relations, mean_position_relations)

