"""File for running tests on the Rasch Model
"""
import numpy as np
import random
import string
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import pandas as pd
from estimation.RaschModel import PL1RaschEstimator, PL2RaschEstimator, PL3RaschEstimator


class TestCase1:
	"""Test case developed from quantdev.ssri.psu.edu/tutorials/introduction-irt-modeling 's data and output
	"""
	@staticmethod
	def load_data():
		# To get data go download it from quantdev.ssri.psu.edu/tutorials/introduction-irt-modeling
		with open("ouirt.txt", "r") as f:
			rows = []
			for line in f.readlines():
				current_row = []
				for no in line.split(" "):
					try:
						current_row.append(int(no))
					except ValueError:
						continue
				rows.append(current_row)
		return pd.DataFrame(rows)

	def run(self):
		results = self.load_data()
		tester = PL2RaschEstimator()
		tester.fit(results, epochs=10000)
		model_params = tester.get_model_descriptors()
		for params in model_params:
			print(params)


NUMBER_OF_STUDENTS = 500
NUMBER_OF_QUESTIONS = 10


class HelperFunction:
	"""Static class container for helpful functions
	"""
	@staticmethod
	def generate_random_names():
		names = []
		for _ in range(0, NUMBER_OF_STUDENTS):
			x = random.choice(string.ascii_letters)
			y = random.choice(string.ascii_letters)
			z = random.choice(string.ascii_letters)
			names.append(f"{x}{y}{z}")
		return names

	@staticmethod
	def generate_test_dataset():
		results = tf.Variable(
			tf.cast(np.random.randint(0, 2, size=[NUMBER_OF_STUDENTS, NUMBER_OF_QUESTIONS]), "float32"))
		return results


test_run = TestCase1()
test_run.run()