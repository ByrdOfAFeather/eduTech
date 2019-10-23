"""
Implementation of JAG (Joint-Assessment-Grading) based on Christoph Studer & Igor Labutov's work
https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14786/13874
"""
import tensorflow as tf
import pandas as pd
from irt.rasch import RaschModel


class JAGRachModel(RaschModel):
	def __init__(self, learning_rate):
		RaschModel.__init__(self, learning_rate)

	@staticmethod
	def predict(batch_students, batch_questions, **kwargs):
		try:
			results = kwargs["results"]
		except KeyError:
			raise Exception("Results must be provided to the JAG Rasch Model!")
		return tf.sigmoid(-results * (batch_students - batch_questions))

	def _train(self, calc_loss=False, **kwargs):
		return super(JAGRachModel, self)._train(calc_loss, results=self.results)


def initialization(test_data):
	s = tf.Variable(tf.random.normal(shape=[test_data.shape[0], 1]))
	q = tf.Variable(tf.random.normal(shape=[1, test_data.shape[1]]))

	pass


def e_step():
	pass


def m_step():
	pass

