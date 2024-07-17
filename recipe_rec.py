import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

recipes_ingtag = load_data('data/recipes_ingtag.parquet')
recipes_steps = load_data('data/recipes_steps.parquet')
recipes_feat = load_data('data/recipes_feat.parquet')

st.image('images/header.jpg', use_column_width=True)
st.title('Content Recipe Search & Recommendations')

tag_options = ['60-minutes-or-less', '30-minutes-or-less', '15-minutes-or-less', 'meat', 'poultry', 'vegetables', 
               'fruit', 'pasta-rice-and-grains', 'dietary', 'healthy', 'low-carb', 'low-sodium', 'low-saturated-fat', 'low-calorie', 
               'low-cholesterol', 'low-fat', 'low-sugar', 'beginner-cook', 'sweet', 'savory']

tags_selected = st.multiselect('Please select any tags for a meal you are interested in.', tag_options)

def all_tags_present(item_tags, selected):
    return all(string in item_tags for string in selected)

recipes_rec = recipes_ingtag.copy()
if tags_selected:
    recipes_rec['tag_match'] = recipes_rec['tags'].apply(all_tags_present, selected=tags_selected)
        
ing_selected = st.text_input('Please enter any ingredients you have on hand separated by commas.', help='Use lowercase and plural form when appropriate')
ing_selected = ing_selected.split(',')

def check_ingredients_df(ingredients_col):
    ingredients = ing_selected
    ings_matched = ingredients_col
    
    # Join all ingredients into a single lowercase string
    ingredients_str = ' '.join(str(ing).lower() for ing in ings_matched)
    
    # Check each item in all_ings
    for item in ingredients:
        item = item.strip('s')
        if item not in ingredients_str:
            return False
    
    # If we've made it through all items without returning False, return True
    return True

search = st.button('Search for matching recipes')
if search:
    recipes_rec = recipes_ingtag.copy()
    if tags_selected:
        recipes_rec['tag_match'] = recipes_rec['tags'].apply(all_tags_present, selected=tags_selected)
        recipes_rec = recipes_rec[recipes_rec['tag_match'] == True]
    if ing_selected:
        recipes_rec['ing_match'] = recipes_rec['ingredients'].apply(check_ingredients_df)
        recipes_rec = recipes_rec[recipes_rec['ing_match'] == True]
    recipes_id = recipes_rec['id'].values
    recipes_steps_rec = recipes_steps[recipes_steps['id'].isin(recipes_id)][['id', 'name', 'description']]
    st.write(recipes_steps_rec)
id_num = st.number_input('Does one of these recipes catch your eye? Enter the id number here.', value=0)
get_recipe = st.button('Get Recipe')
if get_recipe:
    recipe = recipes_steps[recipes_steps['id'] == id_num]
    rec_name = recipe['name'].values[0]
    rec_steps = recipe['steps'].values[0]
    rec_ingredients = recipe['ingredients'].values[0]
    link_name = rec_name.replace(' ', '-')
    link_url = f'https://www.food.com/recipe/{link_name}-{id_num}'
    st.write(f"Link to Recipe: {link_url}")
    st.write(f"Recipe Name: {rec_name.title()}")
    st.write(f"Ingredients: {rec_ingredients}")
    num = 1
    for step in rec_steps:
        st.write(f"Step {num}: {step.capitalize()}")
        num += 1
        
else:
    st.write('No recipe selected yet')

# Get similar recipes
sim = st.button('Take a look at some similar recipes')
if sim:
    rec_feat = recipes_feat[recipes_feat['id'] == id_num]
    rec_feat = rec_feat.drop(columns=['id']).values.reshape(1, -1)

    cosine_sim = cosine_similarity(rec_feat, recipes_feat.drop(columns=['id']))
    sim_scores = list(zip(recipes_feat['id'].values, cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    rec_indices = [i[0] for i in sim_scores]
    
    recs = recipes_steps[recipes_steps['id'].isin(rec_indices)][['id', 'name', 'description']]
    st.write(recs)  