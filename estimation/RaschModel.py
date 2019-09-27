"""
Implementation of Rasch Model
# Rasch, G. 1993. Probabilistic models for some intelligence
# and attainment tests. ERIC.
# TODO: Regularization Hyperparameter & Learning Rate Hyperparameter
# TODO: Differentiate between LP1, LP2, LP3
"""
import pandas as pd
import numpy as np
import tensorflow as tf
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
		Initializes a new RaschModel. All attributes are set to None save for self.optimizer.
		:param learning_rate: The Learning rate for the model
		:attr student_abilities: The estimated abilities for students (can't get size without the input dataset)
		:attr question_difficulties: The estimated difficulties for questions (can't get size without the input dataset)
		:attr results: The tests data provided in fit
		:attr optimizer: The optimization algorithm provided by Tensorflow
		"""
		self.student_abilities = None
		self.questions_difficulties = None
		self.results = None
		self.optimizer = tf.optimizers.SGD()

	@staticmethod
	def _calc_loss(predictions: tf.Tensor, truths: tf.Tensor) -> tf.Tensor:
		"""Returns the loss for the given predictions and truths
		:param predictions: A EagerTensor representing the current predictions from student abilities and question diff.
		:param truths: A EagerTensor representing the current truth values for which questions were correctly answered
		:return: A [1, 1] Tensor containing the loss value as given by the negative log likelihood function
		"""
		return -tf.reduce_sum(truths * tf.math.log(predictions) + (1 - truths) * tf.math.log(1 - predictions))

	@staticmethod
	def predict(batch_students: Union[tf.Tensor, np.array], batch_questions: Union[tf.Tensor, np.array]) -> tf.Tensor:
		"""Returns the current prediction for given students and questions
		NOTE: This will return a #S by #Q matrix when #S != #Q
		:param batch_students: An array-like object representing the batch of student abilities
		:param batch_questions: An array-like object representing the batch of question difficulties
		:return: A sigmoid result of (SI - QJ) done element wise
		"""
		return tf.exp(batch_students - batch_questions) / (1 + tf.exp(batch_students - batch_questions))

	# return batch_students - batch_questions

	def _train(self):
		"""Computes and applies gradients via Tensorflow 2.0
		:return: None
		"""
		with tf.GradientTape() as tape:
			loss = self._calc_loss(self.predict(self.student_abilities, self.questions_difficulties), self.results)

		gradients = tape.gradient(loss, self.questions_difficulties)
		self.optimizer.apply_gradients([(gradients, self.questions_difficulties)])
		return loss

	def fit(self, testing_data: pd.DataFrame, epochs=1000) -> None:
		"""Trains the model on the given data for the number of epochs
		:param testing_data: A Dataframe that's #S x #Q containing 1 for correct answers and 0 for incorrect
		:param epochs: The number of iterations to train for
		:return: None
		"""
		self.student_abilities = tf.cast(tf.Variable(tf.zeros(shape=[testing_data.shape[0], 1])), "float32")
		self.student_abilities.assign((np.sum(testing_data, axis=1) / 10).values.reshape([testing_data.shape[0], 1]))
		self.questions_difficulties = tf.Variable(tf.random.normal(shape=[1, testing_data.shape[1]]))
		self.results = tf.cast(tf.Variable(testing_data.values), "float32")

		loss = None
		for _ in range(0, epochs):
			loss = self._train()
			if _ % 1000 == 0:
				print(f"LOSS AFTER {_} ITERATIONS: {loss}")

		print(f"Finished fitting with a final loss of {loss}")

	def get_model_descriptors(self) -> Tuple[np.array, np.array]:
		"""Gets the student ability and question difficulty in a numpy array
		:return: (student_abilities, question_difficulties)
		"""
		return self.student_abilities.numpy(), self.questions_difficulties.numpy()