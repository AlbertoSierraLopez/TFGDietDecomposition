from data_loader import DataLoader

dataset = "RAW_recipes.csv"

print("Leyendo dataset:", dataset)

data_loader = DataLoader(path_csv="../datasets/"+dataset, cap=50)
data_loader.display_dataframe()

