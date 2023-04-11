import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import shortest_path

# Load the crime dataset into a Pandas DataFrame
df = pd.read_csv('/content/boston.csv',encoding='ISO-8859-1')

# Extract relevant features and target variable
features = ['STREET', 'Lat', 'Long']
target = 'OFFENSE_CODE'

X = df[features]
y = df[target]

# Train a machine learning model to predict crime rates based on the extracted features
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

def find_safest_path(start, end):
    # Convert the start and end points to latitude and longitude
    start_lat, start_long = geocode_address(start)
    end_lat, end_long = geocode_address(end)

    # Find the street segments between the start and end points using a mapping API (e.g., Google Maps API)
    street_segments = find_street_segments(start_lat, start_long, end_lat, end_long)

    # Predict the crime rates for each street segment using the trained model
    X_pred = pd.DataFrame(street_segments, columns=['STREET', 'Lat', 'Long'])
    y_pred = rf.predict(X_pred)

    # Create an adjacency matrix for the street segments
    n = len(street_segments)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = y_pred[i] + y_pred[j]

    # Find the safest path using the Dijkstra algorithm
    shortest_dist, prev = shortest_path(dist_matrix, directed=False, method='D', return_predecessors=True)
    path = []
    current = np.argmin(shortest_dist)
    while current != -1:
        path.append(street_segments[current][0])
        current = prev[current]

    # Reverse the path and return it as a list of street names
    return path[::-1]

# Driver code to test the function
start = '123 Main St, Boston, MA'
end = '456 Elm St, Boston, MA'
safest_path = find_safest_path(start, end)
print('Safest path:', safest_path)
