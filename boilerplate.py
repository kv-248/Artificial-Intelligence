#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
import time

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    train_data = pd.read_csv("train_data.csv")
    validation_data = pd.read_csv("validation_data.csv")

    return train_data , validation_data
    

def make_network(df):
    """
    Define and fit the initial Bayesian Network using bnlearn.

    Parameters:
        df (pd.DataFrame): Training dataset containing the features.

    Returns:
        DAG (dict): A dictionary containing the Bayesian Network structure and parameters.
    """
    # Define edges based on the desired Bayesian Network structure

    edges =  [
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Start_Stop_ID', 'Distance')
    ,  ('Start_Stop_ID', 'Zones_Crossed'),
       ('Start_Stop_ID', 'Route_Type')
     , ('Start_Stop_ID', 'Fare_Category'),
       ('End_Stop_ID', 'Distance'), 
       ('End_Stop_ID', 'Zones_Crossed'), 
       ('End_Stop_ID', 'Route_Type'), 
       ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Zones_Crossed'),
       ('Distance', 'Route_Type'),
       ('Distance', 'Fare_Category')
    ,  ('Zones_Crossed', 'Route_Type'), 
      ('Zones_Crossed', 'Fare_Category'), 
       ('Route_Type', 'Fare_Category')
    ]

    start_time = time.time()
    # Create the DAG structure
    DAG = bn.make_DAG(edges)

    # Fit the Bayesian Network using the data
    model = bn.parameter_learning.fit(DAG, df)

    # Retrieve the edge properties for customization
    edge_properties = bn.get_edge_properties(DAG)

    # Adjust the edge size (thickness/weight)
    for edge in edge_properties:
        edge_properties[edge]['weight'] = 0.2  # Example: Thinner edges

    # Visualize the network with adjusted edge and font properties
    bn.plot(
        model=DAG,
        edge_properties=edge_properties,
        params_static={
            'font_size': 8,   # Adjust font size for better readability
            'figsize': (15, 10),  # Set figure size
        },
    )

    return model
    

def make_pruned_network(df):
    """
    Define and fit a pruned Bayesian Network using statistical tests (e.g., chi2, mutual information).
    
    Parameters:
        df (pd.DataFrame): The dataset to fit the Bayesian Network.
    
    Returns:
        pruned_model (dict): The pruned Bayesian Network model.
    """
    # Perform structure learning to define the initial Bayesian Network
    # Load the initial model
    with open("base_model.pkl", "rb") as file:
        initial_model = pickle.load(file)

    # Perform independence tests for pruning
    pruned_model = bn.independence_test(
        initial_model,  # Extract the actual DAG
        df,
        test='chi_square', 
        prune=True
    )
    pruned_model = bn.parameter_learning.fit(pruned_model, df)

    # Visualize the pruned network (optional)
    bn.plot(pruned_model, params_static={'font_size': 8, 'figsize': (15, 10)})
    return pruned_model


def make_optimized_network(df):

    with open('base_model.pkl', 'rb') as file:
        base_model = pickle.load(file)

    DAG = base_model['model']

    print("Applying Hill Climbing for structure learning...")
    refined_DAG = bn.structure_learning.fit(df, methodtype='hc', verbose=3)


    print("Applying parameter learning on the refined structure...")
    refined_model = bn.parameter_learning.fit(refined_DAG, df)

    bn.plot(refined_model, params_static={'font_size': 8, 'figsize': (15, 10)})

    return refined_model


def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname , 'wb') as file:
        pickle.dump(model, file)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")


############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    
    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done") 

if __name__ == "__main__":
    main()

