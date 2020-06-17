"""
Implementation of Rasch Model
# Rasch, G. 1993. Probabilistic models for some intelligence
# and attainment tests. ERIC.
"""
import pandas as pd
import numpy as np
import torch
from typing import Union, Tuple


class RaschModel:
	"""Implementation of the Rasch model in Tensorflow 2.0
	The Rasch model estimates a particular students "ability" and a question's "difficultly" in the context of academic
	tests. The model is P(X_I_J | S_I, Q_J) where X_I_J is weather student I got
	question J correct (0, 1), S_I is student I's ability, and Q_J is question J's difficultly. The probability function
	is given by P(X|S,Q) = e^X(S-Q) / 1 + e^X(S-Q)
	"""

	def __init__(self, learning_rate=.01):
		"""
		Initializes a new RaschModel.
		:param learning_rate: The Learning rate for the model
		:attr student_abilities: The estimated abilities for students (can't get size without the input dataset)
		:attr question_difficulties: The estimated difficulties for questions (can't get size without the input dataset)
		:attr results: Placeholder for the tests data provided in fit
		"""
		self.student_abilities = None
		self.questions_difficulties = None
		self.results = None
		self.learning_rate = learning_rate

	@staticmethod
	def _calc_loss(predictions: tf.Tensor, truths: tf.Tensor) -> tf.Tensor:
		"""Returns the loss for the given predictions and truths
		:param predictions: A Tensor representing the current predictions from student abilities and question diff.
		:param truths: A Tensor representing the current truth values for which questions were correctly answered
		:return: A [1, 1] Tensor containing the loss value as given by the negative log likelihood function
		"""
		return torch.sum(truths * torch.log(predictions) + (1 - truths) * torch.log(1 - predictions))

	@staticmethod
	def predict(batch_students: Union[torch.Tensor, np.array], batch_questions: Union[torch.Tensor, np.array],
	            **kwargs) -> torch.Tensor:
		"""Returns the current prediction for given students and questions
		NOTE: This will return a #S by #Q matrix when #S != #Q
		:param batch_students: An array-like object representing the batch of student abilities
		:param batch_questions: An array-like object representing the batch of question difficulties
		:param kwargs: In the case of the base Rasch model this will be unused, modifications to the Rasch model will
					   implement these parameters.
		:return: A sigmoid result of (S_I - Q_J) done element wise
		"""
		return torch.sigmoid(batch_students - batch_questions)

	def _calc_deriv_student_ability(self, predictions: torch.Tensor) -> torch.Tensor:
		"""Calculates the derivative in terms of the student abilities
		Citation:“22.1 The Rasch Model.” Bayesian Reasoning and Machine Learning,
			by David Barber, University Cambridge Press, 2012, pp. 403–404.
		:param predictions: A Tensor representing the predictions based on the current parameters of the model
		:return: A Sx1 matrix representing the current gradient in terms of the student abilities
		"""
		return torch.sum(self.results - predictions)

	def _calc_deriv_question_difficulty(self, predictions):
		"""Calculates the derivative in terms of the question difficulties
		Citation:“22.1 The Rasch Model.” Bayesian Reasoning and Machine Learning,
			by David Barber, University Cambridge Press, 2012, pp. 403–404.
		:param predictions: A Tensor representing the predictions based on the current parameters of the model
		:return: A 1xQ matrix representing the current gradient in terms of the question difficulties
		"""
		return -torch.sum(self.results - predictions)

	def _train(self, calc_loss=False, **kwargs):
		"""Computes and applies gradients
		:param calc_loss: boolean indicating if the loss is supposed to be calculated and returned
		:param kwargs: This parameter is to make modified Rasch model implementation easier. These can vary from model
						to model, for a good example see the JAGRaschModel in eduTech.grading.JAG
		:return: The current loss at the time of calculation or None
		"""
		predictions = self.predict(self.student_abilities, self.questions_difficulties, **kwargs)

		grad_student_ability = self.learning_rate * self._calc_deriv_student_ability(predictions)
		# Gradient Normalization (reduces overstepping)
		grad_student_ability = grad_student_ability / grad_student_ability.shape[0]

		grad_question_difficulty = self.learning_rate * self._calc_deriv_question_difficulty(predictions)
		# Gradient Normalization (reduces overstepping)
		grad_question_difficulty = grad_question_difficulty / grad_question_difficulty.shape[1]

		self.student_abilities.assign_add(grad_student_ability)
		self.questions_difficulties.assign_add(grad_question_difficulty)

		if calc_loss:
			return self._calc_loss(predictions, self.results)
		else:
			return None

	def fit(self, testing_data: pd.DataFrame, epochs=100) -> None:
		"""Trains the model on the given data for the number of epochs
		:param testing_data: A Dataframe that's #S x #Q containing 1 for correct answers and 0 for incorrect
		:param epochs: The number of iterations to train for
		:return: None
		"""
		self.student_abilities = torch.zeros([testing_data.shape[0], 1])
		self.questions_difficulties = torch.zeros([1, testing_data.shape[1]])
		self.results = torch.tensor(testing_data.values)

		loss = None
		for _ in range(0, epochs):
			loss = self._train(calc_loss=True)
			if _ % 100 == 0:
				print(f"LOSS AFTER {_} ITERATIONS: {loss}")

		print(f"Finished fitting with a final loss of {loss}")

	def get_model_descriptors(self) -> Tuple[np.array, np.array]:
		"""Gets the student ability and question difficulty in a numpy array
		:return: (student_abilities, question_difficulties)
		"""
		return self.student_abilities.numpy(), self.questions_difficulties.numpy()
