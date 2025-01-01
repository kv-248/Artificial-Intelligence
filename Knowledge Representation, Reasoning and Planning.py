# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping

    routes_to_tripcount = defaultdict(int)
    for _ , row in df_trips.iterrows():
        trip_id = row['trip_id']
        route_id = row['route_id']
        
        # Map trip ID to route ID and update route count
        trip_to_route[trip_id] = route_id
        routes_to_tripcount[route_id] += 1

    # Map route_id to a list of stops in order of their sequence

    i = 0
    prev_trip_id = -1 
    while i < len(df_stop_times):
        row = df_stop_times.iloc[i]  # Access the row at index i
        new_trip_id = row['trip_id'] 
        route_id = trip_to_route.get(new_trip_id)  # Retrieve route ID for the trip ID
        
        # Skip if we're seeing a new trip on an already-processed route
        if new_trip_id != prev_trip_id and route_id in route_to_stops:
            i += len(route_to_stops[route_id])
            prev_trip_id = new_trip_id 
            continue 
        stop_id = row['stop_id']
        route_to_stops[route_id].append(stop_id)
        prev_trip_id = new_trip_id 
        i += 1  

    # Count trips per stop
    for route_id, stop_ids in route_to_stops.items():
        frequency_of_route = routes_to_tripcount.get(route_id,0) 
        for stop_id in stop_ids:
            stop_trip_count[stop_id] += frequency_of_route


    # Ensure each route only has unique stops

    for route_id in route_to_stops:
        route_to_stops[route_id] = list(dict.fromkeys(route_to_stops[route_id]))
    

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')

    # Create fare rules for routes
    for idx, row in merged_fare_df.iterrows():
        route_id = row['route_id']
        if route_id:
            if route_id not in fare_rules:
                fare_rules[route_id] = []
            fare_rules[route_id].append({
                'fare_id': row['fare_id'],
                'origin_id': row.get('origin_id', None),
                'destination_id': row.get('destination_id', None),
                'price': row['price']
            })

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_counts = df_trips['route_id'].value_counts()

    top_routes = [(route_id, count)
                  for route_id, count in route_trip_counts.head(5).items()]

    return top_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    top_stops = sorted(stop_trip_count.items(),
                      key=lambda x: x[1],
                      reverse=True)[:5]

    return [(stop_id, count) for stop_id, count in top_stops]

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # Create a dictionary to count routes per stop using route_to_stops
    stop_route_count = defaultdict(int)

    # Use route_to_stops directly to count routes per stop
    for stops in route_to_stops.values():
        for stop_id in stops:
            stop_route_count[stop_id] += 1

    # Get top 5 stops sorted by number of routes
    top_stops = sorted(stop_route_count.items(),
                      key=lambda x: x[1],
                      reverse=True)[:5]

    return top_stops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    pair_routes = defaultdict(set)  # To store stop pairs and their routes
    pair_frequencies = {}  # To store combined trip frequencies for pairs

    # Helper function to calculate the combined frequency of a stop pair
    def calculate_combined_frequency(stop1, stop2):
        return stop_trip_count.get(stop1, 0) + stop_trip_count.get(stop2, 0)

    # Iterate through all routes to find stop pairs and calculate their frequencies
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_pair = tuple(sorted([stops[i], stops[i + 1]]))
            pair_routes[stop_pair].add(route_id)
            combined_freq = calculate_combined_frequency(stop_pair[0], stop_pair[1])
            pair_frequencies[stop_pair] = combined_freq

    # Identify stop pairs that are served by exactly one route
    single_route_pairs = [
        (pair, next(iter(routes)))  # Select one route from the set
        for pair, routes in pair_routes.items()
        if len(routes) == 1
    ]

    # Sort stop pairs by their combined frequency (in descending order) and select the top 5
    top_pairs = sorted(single_route_pairs, key=lambda x: pair_frequencies[x[0]], reverse=True)[:5]

    return top_pairs

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """

    G = nx.Graph()

    # Dictionary to store node positions
    pos = {}

    # Set to store all unique stops
    all_stops = set()

    # Add nodes and edges from route_to_stops
    for route_id, stops in route_to_stops.items():
        # Add stops to the set of all stops
        all_stops.update(stops)

        # Add edges between consecutive stops in the route
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route_id=route_id)


    pos = nx.spring_layout(G, seed=42)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Calculate node sizes based on degree (number of connections)
    node_sizes = [3 + 2 * G.degree(node) for node in G.nodes()]

    # Calculate node colors based on number of routes passing through
    node_colors = []
    for node in G.nodes():
        routes_count = sum(1 for stops in route_to_stops.values() if node in stops)
        node_colors.append(routes_count)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Number of Routes',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Create hover text
    node_text = []
    for node in G.nodes():
        routes_through = [route_id for route_id, stops in route_to_stops.items()
                        if node in stops]
        text = f"Stop ID: {node}<br>"
        text += f"Connected Stops: {G.degree(node)}<br>"
        text += f"Routes: {len(routes_through)}"
        node_text.append(text)

    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Delhi Transit Network',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[
                           dict(
                               text="Transit Network Visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002
                           )
                       ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))

    # Show the figure in the default web browser
    fig.show()

    # Optional: Save the figure as HTML
    fig.write_html("delhi_transit_network.html")
    

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []

    global route_to_stops  

    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops and start_stop != end_stop:
            direct_routes.append(route_id)


    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    global route_to_stops  # Use the global route_to_stops dictionary

    # Clear previous terms
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog

    # Define Datalog predicates
    DirectRoute(R, X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y))

    # Define OptimalRoute predicate for forward chaining
    OptimalRoute(R1, X, Y, Z, R2) <= (DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1 != R2))
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    result = DirectRoute(R, start, end)
    # Extract route IDs from the result and sort them
    direct_routes = sorted([r[0] for r in result])

    return direct_routes

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    result = OptimalRoute(R1, start_stop_id, end_stop_id, stop_id_to_include, R2)

    # Extract route information from the result
    optimal_routes = [(r1, stop_id_to_include, r2) for r1, r2 in result]


    return optimal_routes

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Query Datalog for optimal routes using backward chaining
    result = OptimalRoute(R1, end_stop_id, start_stop_id, stop_id_to_include, R2)

    # Extract route information from the result
    optimal_routes = [(r1, stop_id_to_include, r2) for r1, r2 in result]

    return optimal_routes


pyDatalog.create_terms('State, Action, At, DirectRoute , Route, Transfer, ValidPath')
pyDatalog.create_terms('CurrentStop, NextStop, R, R1, R2, X, Y, Z')

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    global route_to_stops  # Use the global route_to_stops dictionary

    # Clear previous terms
    pyDatalog.clear()  # Clear previous terms

    # create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog

    # Define Datalog predicates
    DirectRoute(R, X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y))

    # Define OptimalRoute predicate for forward chaining
    OptimalRoute(R1, X, Y, Z, R2) <= (DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1 != R2)) 



    
    # Action 1: Board a route
    Action.board_route(R, X, Y) <= (
        DirectRoute(R, X, Y)  # Direct route exists
    )
   
    # Action 2: Transfer between routes
    Action.transfer_route(R1, R2, Z) <= (
        DirectRoute(R1, start_stop_id, Z) &  # First route to transfer stop
        DirectRoute(R2, Z, end_stop_id) &     # Second route from transfer stop
        (R1 != R2)                         # Different routes
    )

    # Define valid paths in terms of Actions
    ValidPath(R1, Z, R2) <= (
        Action.board_route(R1, start_stop_id, Z) &
        Action.board_route(R2, Z, end_stop_id) &
        Action.transfer_route(R1, R2, Z) &
        (Z == stop_id_to_include)
    )
    # Query for valid paths and process results
    result = ValidPath(R1, Z, R2)


    
    # Convert result to list of tuples, handling all returned values
    optimal_routes = []
    for solution in result:
        r1, z, r2 = solution
        optimal_routes.append((r1, stop_id_to_include, r2))
    # Print planning steps for each found path
    # for route in optimal_routes:
    #     print("\nPlanning Steps:")
    #     print(f"1. Start at stop {start_stop_id}")
    #     print(f"2. Board route {route[0]}")
    #     print(f"3. Travel to transfer stop {route[1]}")
    #     print(f"4. Transfer to route {route[2]}")
    #     print(f"5. Travel to destination {end_stop_id}")

    return optimal_routes

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    filtered_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return filtered_df

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}

    route_fares = pruned_df.groupby('route_id')['price'].min().to_dict()
    w = 0 
    for route_id, stops in route_to_stops.items():
        route_summary[route_id] = {'min_price': route_fares.get(route_id, float('inf')), 'stops': set(stops)}

    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    # Queue for BFS, storing (current_stop, current_route, path, total_fare, transfers)
    queue = deque([(start_stop_id, None, [], 0, 0)])
    visited = set()  # Track visited (stop, route, transfers)

    while queue:
        current_stop, current_route, path, total_fare, transfers = queue.popleft()

        # If the end stop is reached and fare constraints are met
        if current_stop == end_stop_id and total_fare <= initial_fare:
            return path

        # Skip if transfers exceed the maximum allowed
        if transfers > max_transfers:
            continue

        # Explore routes containing the current stop
        for route_id, route_data in route_summary.items():
            if current_stop not in route_data['stops']:
                continue

            # Calculate the new transfer count if switching routes
            new_transfers = transfers + (1 if route_id != current_route else 0)

            # Avoid revisiting a stop on the same route with fewer transfers
            if (current_stop, route_id, transfers) in visited:
                continue
            visited.add((current_stop, route_id, transfers))

            # Use the minimum fare for this route
            new_total_fare = total_fare + route_data['min_price']
            if new_total_fare > initial_fare:
                continue  # Skip paths that exceed the fare limit

            # Explore each neighboring stop on this route
            for neighbor_stop in route_data['stops']:
                if neighbor_stop != current_stop:  # Skip self-loops
                    # Create the new path
                    new_path = path + [(route_id, neighbor_stop)]

                    # Add the next state to the queue
                    queue.append((
                        neighbor_stop,           # Move to destination stop
                        route_id,                # Continue on the same route
                        new_path,                # Updated path
                        new_total_fare,          # Updated fare
                        new_transfers            # Updated transfer count
                    ))
    return []  # Return an empty list if no valid path is found
