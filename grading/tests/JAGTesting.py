import pandas as pd
from grading.JAG import JAGRachModel


def load_test_data():
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


# The learning rate and epochs are arbitrary and chosen to match the likelihood of
# the Penn State tutorial, in reality the R implementation use Marginal Maximum likelihood estimation
# which optimizes the model in a significantly different way.
test_model = JAGRachModel(learning_rate=.001)
test_data = load_test_data()
test_model.fit(test_data, epochs=250)
student_abilities, question_difficulties = test_model.get_model_descriptors()
print("======= Question Difficulties =======")
print(question_difficulties)
