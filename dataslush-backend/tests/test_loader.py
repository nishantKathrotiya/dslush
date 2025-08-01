from app.data.loader import load_all_data

data = load_all_data()
print(data['talents'].head(2))      # Confirm bio, skills cleaned
print(data['creators'].head(1))     # Confirm HTML is stripped
