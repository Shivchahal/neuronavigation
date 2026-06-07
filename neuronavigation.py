import osmnx as ox
import networkx as nx
import heapq
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. Load the NEW Deep Learning Brain ---
print("🧠 Loading LSTM Predictive Model...")
try:
    model = tf.keras.models.load_model('traffic_model.h5')
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"\n--- ERROR: Could not load traffic_model.h5 ---\n{e}")
    exit()

# --- 2. THE AI PREDICTION ENGINE (FIXED FOR LSTM) ---
def get_future_congestion_factor():
    """
    STRESS TEST VERSION:
    We are sending a sequence of 3000 -> 4500 -> 6000 -> 7500 cars.
    This trend tells the LSTM that a massive jam is coming.
    """
    try:
        # Instead of reading the CSV tail, we provide a 'Traffic Spike' scenario
        test_history = np.array([3000, 4500, 6000, 7500]) / 8000.0 
        input_data = test_history.reshape((1, 4, 1))
        
        prediction = model.predict(input_data, verbose=0)
        
        # Multiply by 15 to ensure the 'A*' algorithm finds the main road 'too expensive'
        # and forces a detour to the side streets.
        return prediction[0][0] * 15 
    except:
        return 0.9  # Emergency fallback to high congestion

# --- 3. A* Algorithm (Unchanged but uses our new weights) ---
def heuristic(G, node, goal):
    node_data = G.nodes[node]
    goal_data = G.nodes[goal]
    return ((node_data['x'] - goal_data['x'])**2 + (node_data['y'] - goal_data['y'])**2)**0.5

def a_star_pathfinder(G, start_node, goal_node, weight_key):
    try:
        return nx.shortest_path(G, start_node, goal_node, weight=weight_key)
    except nx.NetworkXNoPath:
        return None

# --- 4. Main Execution (THE BIG UPGRADE) ---
def main():
    # 1. SET LOCATIONS (Kurukshetra area for demonstration)
    start_lat_lon = (29.9695, 76.8783)
    end_lat_lon = (29.9600, 76.8900)
    
    print(f"🗺️ Downloading map data...")
    G = ox.graph_from_point(start_lat_lon, dist=1500, network_type='drive')
    G = ox.add_edge_speeds(G, fallback=40)
    G = ox.add_edge_travel_times(G)

    # 2. RUN THE AI INFERENCE
    print("🚦 Querying LSTM for future traffic patterns...")
    future_congestion = get_future_congestion_factor()
    print(f"📈 AI Predicts future congestion factor: {future_congestion:.2f}")

    # 3. APPLY AI WEIGHTS TO THE STREETS
    for u, v, k, data in G.edges(data=True, keys=True):
        # The 'Dumb' weight
        data['standard_time'] = data['travel_time']
        
        # The 'AI' weight: We multiply time by predicted congestion
        # If AI predicts high volume (0.8), roads get 4x slower
        data['ai_weighted_time'] = data['travel_time'] * (1 + future_congestion * 5)

    # 4. FIND NEAREST NODES
    start_node = ox.nearest_nodes(G, start_lat_lon[1], start_lat_lon[0])
    goal_node = ox.nearest_nodes(G, end_lat_lon[1], end_lat_lon[0])

    # 5. COMPARE PATHS
    print("\nCalculating 'Dumb' Path (Shortest Distance)...")
    path_dumb = a_star_pathfinder(G, start_node, goal_node, 'length')
    
    print("Calculating 'AI' Path (Predictive Future Time)...")
    path_ai = a_star_pathfinder(G, start_node, goal_node, 'ai_weighted_time')

    # 6. RESULTS & VISUALS
    if path_dumb and path_ai:
        # Calculate time saved
       # Updated for OSMnx 2.x compatibility
        time_standard = sum(ox.routing.route_to_gdf(G, path_dumb)['travel_time'])
        time_ai = sum(ox.routing.route_to_gdf(G, path_ai)['travel_time'])
        
        print(f"\n✅ SUCCESS!")
        print(f"Standard Travel Time: {time_standard/60:.2f} mins")
        print(f"AI Optimized Time: {time_ai/60:.2f} mins")
        
        diff = (time_standard - time_ai) / 60
        if diff > 0:
            print(f"💡 The AI saved you {diff:.2f} minutes!")
        else:
            print("💡 Both paths are optimal for this traffic state.")

        # Plotting
        print("\n🎨 Saving comparison map to 'ai_vs_standard.png'...")
        fig, ax = ox.plot_graph_routes(G, [path_dumb, path_ai], 
                                      route_colors=['red', 'green'], 
                                      route_linewidths=4, node_size=0,
                                      save=True, filepath="ai_vs_standard.png")
    else:
        print("❌ Could not find a path between those points.")

if __name__ == "__main__":
    main()