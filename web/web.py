import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge
import base64
from PIL import Image
from itertools import product

# Set page config
st.set_page_config(page_title="Reaction Optimization Platform", layout="wide")

# Set background
def set_bg_image():
    with open('./figure/background.jpg', "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stDataFrame {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .centered {{
            display: flex;
            justify-content: center;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_bg_image()

# Initialize session state
if 'known_reactions' not in st.session_state:
    st.session_state.known_reactions = pd.DataFrame()
if 'chemical_space' not in st.session_state:
    st.session_state.chemical_space = pd.DataFrame()
if 'descriptor_type' not in st.session_state:
    st.session_state.descriptor_type = "Morgan Fingerprint"
if 'condition_names' not in st.session_state:
    st.session_state.condition_names = []
if 'show_output' not in st.session_state:
    st.session_state.show_output = False
if 'last_method_used' not in st.session_state:
    st.session_state.last_method_used = None

# Page 1: Introduction
def introduction_page():
    st.markdown("<h1 style='text-align: center; color: black;'>Reaction Optimization Platform</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Shu-Wen Li, Shan Chen, Shuo-Qing Zhang, Lutz Ackermann,* and Xin Hong*</h3>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: black;'>
    This is a reaction optimization platform. Simply input five or more known reactions (by index or molecular components), 
    and the system will analyze the data using orthogonal experimental design principles to predict high-yield reaction conditions, 
    as shown in the schematic below.
    </p>
    """, unsafe_allow_html=True)
    
    try:
        workflow_img = Image.open('./figure/workflow.jpg')
        st.image(workflow_img, use_container_width=True)
    except FileNotFoundError:
        st.warning("Workflow image not found at './figure/workflow.jpg'")

# Helper function for molecular fingerprints
def smi_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    else:
        return np.zeros(1024)

# Page 2: Prediction
def prediction_page():
    st.markdown("<h1 style='text-align: center; color: black;'>Recommended Reaction Conditions</h1>", 
                unsafe_allow_html=True)
    
    # Part 1: Descriptor selection
    st.subheader("1. Descriptor and Model Selection")
    st.session_state.descriptor_type = st.selectbox("Select descriptor type:", ["Morgan Fingerprint", "Chemprop"])
    
    # Model selection - use the returned value directly
    model_type = st.selectbox("Select model type:", 
                            ["Decision Tree", "Extra Trees","Gradient Boosting", 
                             "Kernel Ridge","K-Nearest Neighbors", "Linear SVR",
                             "Random Forest", "Ridge", "SVR"],
                            key="model_type")
    
    # Part 2: Create Chemical Space
    st.subheader("2. Generate Chemical Space")
    
    # Method 1: Upload CSV file
    st.markdown("**Method 1: Upload CSV File**")
    uploaded_file = st.file_uploader("Upload reaction data (CSV):", type=["csv"], key="file_uploader",
                                   help="You can use the test_file.csv provided in the package for testing purposes")
    st.caption("ℹ️ A test file (test_file.csv) is provided in the package for testing")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Automatically detect condition columns
            possible_yield_cols = ['yield', 'Yield', 'output']
            possible_index_cols = ['index', 'Index', 'id', 'ID','entry','Entry']
            
            condition_cols = [col for col in df.columns 
                            if col not in possible_yield_cols 
                            and col not in possible_index_cols]
            
            selected_cols = st.multiselect("Select condition columns:", 
                                         df.columns.tolist(), 
                                         default=condition_cols,
                                         key="csv_columns")
            
            if selected_cols:
                # Get known reactions
                yield_col = next((col for col in possible_yield_cols if col in df.columns), None)
                if yield_col:
                    known_reactions = df[df[yield_col].notna()][selected_cols + [yield_col]]
                    st.session_state.known_reactions = known_reactions
                    
                    # Generate chemical space from unique combinations
                    unique_values = {col: df[col].dropna().unique().tolist() for col in selected_cols}
                    chemical_space = pd.DataFrame(list(product(*unique_values.values())), 
                                                columns=unique_values.keys())
                    st.session_state.chemical_space = chemical_space
                    
                    st.session_state.last_method_used = "file"
                    st.session_state.show_output = True
                    st.success(f"Chemical space created with {len(chemical_space)} combinations!")
                else:
                    st.error("No yield column found in the uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Method 2: Manual input
    st.markdown("**Method 2: Manual Input**")
    
    num_conditions = st.number_input("Number of reaction conditions:", 
                                   min_value=1, max_value=20, value=3,
                                   key="num_conditions")
    
    # Initialize condition names if not already set or if number changed
    if ('condition_names' not in st.session_state or 
        len(st.session_state.condition_names) != num_conditions):
        st.session_state.condition_names = [f"Condition_{i+1}" for i in range(num_conditions)]
        st.session_state.condition_values = ["" for _ in range(num_conditions)]
    
    # Dynamic input fields for condition names
    cols = st.columns(num_conditions + 1)
    for i in range(num_conditions):
        with cols[i]:
            st.session_state.condition_names[i] = st.text_input(
                f"Condition {i+1} name:", 
                value=st.session_state.condition_names[i],
                key=f"cond_name_{i}"
            )
    
    with cols[-1]:
        st.write("Yield")
    
    # Input fields for new reaction
    new_cols = st.columns(num_conditions + 1)
    current_values = ["" for _ in range(num_conditions)]
    for i in range(num_conditions):
        with new_cols[i]:
            current_values[i] = st.text_input(
                f"{st.session_state.condition_names[i]} SMILES/Value:", 
                key=f"cond_value_{i}"
            )
    
    with new_cols[-1]:
        yield_value = st.number_input("Yield value:", min_value=0.0, max_value=100.0, value=50.0,
                                    key="yield_value")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Manual Reaction"):
            if all(current_values) and all(st.session_state.condition_names):
                new_reaction = {
                    name: value 
                    for name, value in zip(st.session_state.condition_names, current_values)
                }
                new_reaction['yield'] = yield_value
                
                # Add to known reactions
                if st.session_state.known_reactions.empty:
                    st.session_state.known_reactions = pd.DataFrame([new_reaction])
                else:
                    # Check if columns match existing data
                    if set(new_reaction.keys()) == set(st.session_state.known_reactions.columns):
                        st.session_state.known_reactions = pd.concat([
                            st.session_state.known_reactions,
                            pd.DataFrame([new_reaction])
                        ], ignore_index=True)
                    else:
                        st.error("Condition names don't match existing data structure")
                        return
                
                # Update chemical space with unique combinations
                unique_values = {}
                for i, name in enumerate(st.session_state.condition_names):
                    if name in st.session_state.chemical_space.columns:
                        unique_vals = st.session_state.chemical_space[name].unique().tolist()
                        if current_values[i] not in unique_vals:
                            unique_vals.append(current_values[i])
                        unique_values[name] = unique_vals
                    else:
                        unique_values[name] = [current_values[i]]
                
                # Generate all combinations
                chemical_space = pd.DataFrame(list(product(*unique_values.values())), 
                                            columns=unique_values.keys())
                st.session_state.chemical_space = chemical_space
                
                st.session_state.last_method_used = "manual"
                st.session_state.show_output = True
                st.success("Reaction added successfully!")
            else:
                st.error("Please fill all condition names and values")
    
    with col2:
        if st.button("Remove Last Reaction") and not st.session_state.known_reactions.empty:
            st.session_state.known_reactions = st.session_state.known_reactions.iloc[:-1]
            
            # Rebuild chemical space from remaining reactions
            if not st.session_state.known_reactions.empty:
                condition_cols = [col for col in st.session_state.known_reactions.columns if col != 'yield']
                unique_values = {
                    col: st.session_state.known_reactions[col].unique().tolist() 
                    for col in condition_cols
                }
                st.session_state.chemical_space = pd.DataFrame(
                    list(product(*unique_values.values())),
                    columns=unique_values.keys()
                )
            
            st.session_state.last_method_used = "manual"
            st.session_state.show_output = True
            st.success("Last reaction removed!")
    
    # Display output only for the last used method
    if st.session_state.show_output and st.session_state.last_method_used:
        if (st.session_state.last_method_used == "file" and uploaded_file is not None) or \
           (st.session_state.last_method_used == "manual" and not st.session_state.known_reactions.empty):
            
            st.subheader("Performed Reactions:")
            st.dataframe(st.session_state.known_reactions)
            
            st.subheader("Generated Full Chemical Space:")
            st.write(f"Size: {len(st.session_state.chemical_space)}")
            st.dataframe(st.session_state.chemical_space)
    
    # Part 3: Recommendation
    st.markdown("---")
    st.subheader("3. Recommended Experiments ")
    
    if not st.session_state.known_reactions.empty:
        # Get condition columns (exclude yield)
        condition_cols = [col for col in st.session_state.known_reactions.columns 
                         if col != 'yield']
        num_conditions = len(condition_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            num_recommendations = st.number_input("Number of experiments to recommend:", 
                                                min_value=1, max_value=20, value=5,
                                                key="num_recommendations")
        with col2:
            ortho_strength = st.number_input("Orthogonal constraint strength:", 
                                           min_value=1, max_value=num_conditions, 
                                           value=min(2, num_conditions),
                                           key="ortho_strength")
        
        if st.button("Recommend Experiments", type="primary", key="recommend_button"):
            with st.spinner('Training model and making predictions...'):
                try:
                    # Prepare training data
                    X_train = []
                    y_train = st.session_state.known_reactions['yield'].values
                    
                    if st.session_state.descriptor_type == "Morgan Fingerprint":
                        for _, row in st.session_state.known_reactions.iterrows():
                            fps = [smi_to_fp(row[col]) for col in condition_cols]
                            combined = np.concatenate(fps)
                            X_train.append(combined)
                    else:  # chemprop
                        for _, row in st.session_state.known_reactions.iterrows():
                            desc = [len(str(row[col])) for col in condition_cols]  
                            X_train.append(desc)
                    
                    X_train = np.array(X_train)
                    
                    # Train selected model
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingRegressor(random_state=42)
                    elif model_type == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)
                    elif model_type == "Extra Trees":
                        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                    elif model_type == "K-Nearest Neighbors":
                        model = KNeighborsRegressor()
                    elif model_type == "Kernel Ridge":
                        model = KernelRidge()
                    elif model_type == "Linear SVR":
                        model = LinearSVR(random_state=42)
                    elif model_type == "Ridge":
                        model = Ridge(random_state=42)
                    elif model_type == "SVR":
                        model = SVR()
                    
                    model.fit(X_train, y_train)
                    
                    # Prepare prediction data (unseen combinations)
                    known_combinations = st.session_state.known_reactions[condition_cols].apply(tuple, axis=1).tolist()
                    all_combinations = st.session_state.chemical_space[condition_cols].apply(tuple, axis=1).tolist()
                    unseen_combinations = [comb for comb in all_combinations if comb not in known_combinations]
                    
                    if not unseen_combinations:
                        st.info("No unseen reaction combinations available for recommendation.")
                    else:
                        # Predict yields for unseen combinations
                        X_pred = []
                        for comb in unseen_combinations:
                            if st.session_state.descriptor_type == "Morgan Fingerprint":
                                fps = [smi_to_fp(smi) for smi in comb]
                                combined = np.concatenate(fps)
                                X_pred.append(combined)
                            else:  # chemprop
                                desc = [len(str(smi)) for smi in comb]  
                                X_pred.append(desc)
                        
                        X_pred = np.array(X_pred)
                        pred_yields = model.predict(X_pred)
                        
                        # Create results dataframe
                        results = pd.DataFrame(unseen_combinations, columns=condition_cols)
                        results['Predicted_Yield'] = pred_yields
                        
                        # Sort by predicted yield
                        results = results.sort_values('Predicted_Yield', ascending=False)
                        
                        # Apply orthogonal constraint
                        recommended = []
                        known_set = set(known_combinations)
                        
                        for _, row in results.iterrows():
                            if len(recommended) >= num_recommendations:
                                break
                            
                            current_comb = tuple(row[condition_cols])
                            valid = True
                            
                            for known in known_set:
                                diff_count = sum(1 for a, b in zip(current_comb, known) if a != b)
                                
                                if diff_count < ortho_strength:
                                    valid = False
                                    break
                            
                            if valid:
                                recommended.append(row)
                                known_set.add(current_comb)
                        
                        if recommended:
                            recommendations = pd.DataFrame(recommended)
                            st.success(f"Top {len(recommendations)} Recommended Experiments:")
                            st.dataframe(recommendations)
                        else:
                            st.warning("No recommendations found that satisfy the orthogonal constraints. Try reducing the constraint strength.")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please add known reactions first to enable recommendations")

# Page selection
pages = {
    "Introduction": introduction_page,
    "Prediction": prediction_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()