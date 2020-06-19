"""
Implementation of Rasch Model
# Rasch, G. 1993. Probabilistic models for some intelligence
# and attainment tests. ERIC.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class RaschModel(nn.Module):
	"""TODO:
	3) Copy over relevant documentation
	"""

	def __init__(self):
		super(RaschModel, self).__init__()
		self.student_abilities = None
		self.question_difficulties = None

	def initialize_weights(self, no_students, no_questions):
		self.student_abilities = nn.Parameter(torch.rand([no_students, 1]))
		self.question_difficulties = nn.Parameter(torch.rand([1, no_questions]))

	def forward(self, students, questions):
		student_values = self.student_abilities[students]
		difficulty_values = self.question_difficulties[:, questions]
		return torch.sigmoid(student_values - difficulty_values)

	@staticmethod
	def _process_training_data(training_data: [torch.Tensor, pd.DataFrame, np.Array]):
		if isinstance(training_data, torch.Tensor):
			assert training_data.shape == 2, f"Expected data to be 2d. Got {len(training_data.shape)}d instead."
			return training_data
		elif isinstance(training_data, pd.DataFrame):
			assert training_data.shape == 2, f"Expected data to be 2d. Got {len(training_data.shape)}d instead."
			return torch.tensor(training_data.values.tolist()).float()
		elif isinstance(training_data, np.ndarray):
			assert training_data.shape == 2, f"Expected data to be 2d. Got {len(training_data.shape)}d instead."
			return torch.tensor(training_data.tolist()).float()

	def get_model_descriptors(self):
		return self.student_abilities, self.question_difficulties

	def fit(self, testing_data, epochs, learning_rate, regularization_term, verbose=False):
		optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=regularization_term)
		crtiertion = torch.nn.BCELoss(reduction="sum")
		students = list(range(testing_data.shape[0]))
		questions = list(range(testing_data.shape[1]))
		targets = self._process_training_data(testing_data)
		for i in range(epochs):
			self.zero_grad()
			preds = self.forward(students, questions)
			loss = crtiertion(preds, targets)
			loss.backward()
			optimizer.step()
			if i % 1000 == 0:
				if verbose:
					print(f"====== ITERATION {i} ========")
					print(f"Loss: {loss.item()}")
					print("=============================")


class IRT2PL(RaschModel):
	def __init__(self):
		super(IRT2PL, self).__init__()
		self.question_discriminator_values = None

	def initialize_weights(self, no_students, no_questions):
		super(IRT2PL, self).initialize_weights(no_students, no_questions)
		self.question_discriminator_values = nn.Parameter(torch.rand([1, no_questions]))

	def forward(self, students, questions):
		student_values = self.student_abilities[students]
		difficulty_values = self.question_difficulties[:, questions]
		discriminator_values = self.question_difficulties[:, questions]
		return torch.sigmoid(torch.mul(discriminator_values, student_values - difficulty_values))

	def fit(self, testing_data, epochs, learning_rate, regularization_term, verbose=False):
		super(IRT2PL, self).fit(testing_data, epochs, learning_rate, regularization_term, verbose)
