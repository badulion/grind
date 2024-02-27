from pysr import PySRRegressor



model = PySRRegressor.from_file("hall_of_fame_2024-02-26_153058.137.pkl")
torch_model = model.pytorch()

