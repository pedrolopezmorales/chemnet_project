import pandas as pd
from pyvis.network import Network
import re
from country_list import countries_for_language
import pubchempy as pcp
import os
import json
from django.conf import settings
import ast

CSV_PATH = os.path.join(settings.BASE_DIR, 'data', 'esandt_papers_2024_with_inchikeys.csv')

import requests
import time
from functools import lru_cache

main = pd.read_csv(CSV_PATH)


countries = dict(countries_for_language('en'))
country_names = list(countries.values())
countries.update({'U.K.':'United Kingdom',
                  'Brasil':'Brazil',
                  'Republic of Korea':'South Korea',
                  'Czech Republic':'Czechia',
                  'United State':'United States',
                  'The Netherlands' : 'Netherlands',
                  'Slovak Republic' : 'Slovakia',
                  'Korea':'Korea',
                  'Lao PDR':'Laos',
                  'England':'England',
                  'Chinese':'China'
                 })
university_keys = ['institute of','university','instituto','Universidad','Universita','Universit']

#function for categorizing funding sources

@lru_cache(maxsize=1000)
def categorize_funding_source(entity_name):
    if not entity_name or pd.isna(entity_name):
        return 'Unknown'
    
    result = categorize_funding_source_keywords(entity_name)
    if result != 'Unknown':
        return result

    result = check_opencorporates_api(entity_name)
    if result != 'Unknown':
        return result
    
    result = check_wikipedia_api(entity_name)
    if result != 'Unknown':
        return result

    result = check_government_databases(entity_name)
    if result != 'Unknown':
        return result
    
    return 'Unknown'

@lru_cache(maxsize=1000)
def check_opencorporates_api(entity_name):
    try:
        url = "https://api.opencorporates.com/v0.4/companies/search"
        params = {
            'q': entity_name,
            'format': 'json',
            'limit': 3,
            'order': 'score'
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            companies = data.get('results', {}).get('companies', [])

            for company_data in companies:
                company = company_data.get('company', {})
                name = company.get('name', '').lower()
                entity_lower = entity_name.lower()

                if entity_lower in name or name in entity_lower:
                    company_type = company.get('company_type', '').lower()
                    status = company.get('current_status', '').lower()

                    if 'active' in status:
                        if any(corp_type in company_type for corp_type in ['corporation', 'inc', 'llc', 'ltd', 'limited', 'company']):
                            return 'Company'
                        elif any(np_type in company_type for np_type in ['non-profit', 'nonprofit', 'foundation']):
                            return 'Foundation'
        time.sleep(0.2)
    except Exception as e:
        print(f"OpenCorporates API error for {entity_name}: {e}")
    return 'Unknown'

@lru_cache(maxsize=1000)
def check_wikipedia_api(entity_name):
    try: 
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        entity_encoded = entity_name.replace(' ', '_')
        response = requests.get(f"{search_url}{entity_encoded}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            extract = data.get('extract', '').lower()
            title = data.get('title', '').lower()

            if any(term in extract for term in [
                'government agency', 'federal agency', 'department of', 
                'ministry of', 'government department', 'public agency',
                'federal government', 'united states government', 'government'
            ]):
                return 'Government'
            if any(term in extract for term in [
                'university', 'college', 'institute of technology',
                'academic institution', 'higher education', 'medical school'
            ]):
                return 'University'
            if any(term in extract for term in [
                'foundation', 'charitable foundation', 'non-profit',
                'nonprofit', 'charity', 'philanthropic', 'endowment'
            ]):
                return 'Foundation'
            if any(term in extract for term in [
                'corporation', 'company', 'inc.', 'pharmaceutical company',
                'biotechnology company', 'multinational corporation',
                'publicly traded', 'private company'
            ]):
                return 'Company'
        time.sleep(0.2)
    except Exception as e:
        print(f"Wikipedia API error for {entity_name}: {e}")
    return 'Unknown'

def check_government_databases(entity_name):
    entity_lower = entity_name.lower().strip()

    us_government = {
        'national science foundation': 'Government',
        'national institutes of health': 'Government',
        'department of energy': 'Government',
        'department of defense': 'Government',
        'environmental protection agency': 'Government',
        'nasa': 'Government',
        'nih': 'Government',
        'nsf': 'Government',
        'doe': 'Government',
        'dod': 'Government',
        'epa': 'Government',
        'cdc': 'Government',
        'fda': 'Government'
    }
    for agency, classification in us_government.items():
        if agency in entity_lower:
            return classification
    government_patterns = [
        r'\b(u\.?s\.?|united states)\s+(department|agency|office)\b',
        r'\bnational\s+(institute|center|laboratory)\b',
        r'\bministry\s+of\b'
    ]
    
    for pattern in government_patterns:
        if re.search(pattern, entity_lower):
            return 'Government'
    
    return 'Unknown'

def categorize_funding_source_keywords(entity_name):
    entity_lower = entity_name.lower().strip()
    
    for key in university_keys:
        if key.lower() in entity_lower:
            return 'University'
    if any(term in entity_lower for term in ['foundation', 'trust', 'endowment', 'charity']):
            return 'Foundation'
    if any(term in entity_lower for term in ['inc.', 'corp.', 'corporation', 'llc', 'ltd', 'company', 'pharmaceutical', 'biotech', 'technologies']):
            return 'Company'
    if any(term in entity_lower for term in ['department', 'government']):
            return 'Government'
    return 'Unknown'
#graphing funding source function to get category color
def get_category_color(category):
    color_map = {
        'Government': '#FF6B6B',      # Red
        'University': '#96CEB4',      # Light Green
        'Foundation': '#4ECDC4',      # Teal
        'Company': '#FFEAA7',         # Yellow
        'Unknown': '#DDD6FE'          # Light Purple
    }
    return color_map.get(category, "#DDD6FE")

def add_classification_to_funding_sources(funding_sources_list):
    classified_sources = []
    for source in funding_sources_list:
        if source and not pd.isna(source):
            category = categorize_funding_source(source.strip())
            classified_sources.append(f"{source.strip()} [{category}]")
        else:
            classified_sources.append(source)
    return classified_sources
def extract_name_and_class(classified_source):
    if '[' in classified_source and ']' in classified_source:
        parts = classified_source.rsplit('[',1)
        name=parts[0].strip()
        category=parts[1].replace(']','').strip()
        return name, category
    return classified_source, 'Unknown'

def classify_companies_series(companies_list):
    classified_companies=[]
    for company in companies_list:
        if company and not pd.isna(company):
            category = categorize_funding_source(company.strip())
            classified_company = f"{company.strip()} [{category}]"
            classified_companies.append(classified_company)
        else:
            classified_companies.append(company)
    return classified_companies

# Modifying Database by removing certain columns

comparing_companies = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract'], axis = 1)

# Creating new dataframe that had a list of companies, a list of chemicals, and a list of affiliations per row

companies = comparing_companies['Funding Sources'].str.split(r'[;]').explode().str.strip().tolist()
no_dup_comp = list(set(companies))
affiliations = comparing_companies['Affiliations'].str.split(r'[|]').explode().str.strip().tolist()
no_dup_aff = list(set(affiliations))
new_no_dup_aff = []
for aff in no_dup_aff:
    if isinstance(aff,str) and aff != '':
        if ',' in aff:
            attributes = aff.split(',')
            ext_delimiter = False
            for attr in attributes:
                if ';' in attr:
                    poss_delimiter = attr.split(';')
                    first = poss_delimiter[0].lower().strip()
                    for country in country_names:
                        if country.lower() in first:
                            ext_delimiter = True
                    if ext_delimiter == False:
                        for abb in countries:
                            if abb.lower() == first:
                                ext_delimiter = True
            if ext_delimiter == True:
                new_no_dup_aff.extend([a.strip() for a in aff.split(';')])
            else:
                new_no_dup_aff.append(aff)
    else:
        continue
def match_items_against_master(df, column, master_list, delimiters=r'[;]'):
    """
    Given a DataFrame and a master list, return a new column of matched items.
    
    Parameters:
    - df: pandas DataFrame
    - column: name of the column to search (expects strings or lists)
    - master_list: list of values to match against
    
    Returns:
    - A new Series with lists of matched values per row
    """
    split_series = df[column].str.split(delimiters).apply(
        lambda x: [i.strip() for i in x] if isinstance(x, list) else []
    )

    def match_items(row_items):
        # Use a set to avoid duplicates
        return [item for item in master_list if item in row_items]

    return split_series.apply(match_items)
# function specifically for affiliations series because it decided to be really confusing
def match_items_against_master_aff(df, column, master_list):
    """
    Splits affiliation strings intelligently using '|' always, and ';' only when it follows a country.
    Matches split items against a master list.
    
    Parameters:
    - df: pandas DataFrame
    - column: column name (expects strings)
    - master_list: list of values to match
    - country_names: list of full country names
    - abbreviations: list of country abbreviations
    
    Returns:
    - A Series of lists with matched values
    """
    def split_affiliations(aff_string):
        if not isinstance(aff_string, str) or aff_string.strip() == '':
            return []

        parts = aff_string.split('|')
        final_parts = []

        for part in parts:
            if ';' in part:
                attrs = part.split(',')
                should_split = False

                for attr in attrs:
                    if ';' in attr:
                        first = attr.split(';')[0].strip().lower()
                        for country in country_names:
                            if country.lower() in first:
                                should_split = True
                        if not should_split:
                            for abbr in countries:
                                if abbr.lower() == first:
                                    should_split = True

                if should_split:
                    final_parts.extend([a.strip() for a in part.split(';')])
                else:
                    final_parts.append(part.strip())
            else:
                final_parts.append(part.strip())

        return final_parts

    split_series = df[column].apply(split_affiliations)

    return split_series.apply(lambda items: [item for item in items if item in master_list])
def split_researchers(r_string):
    if pd.isna(r_string):
        return []
    parts = [part.strip() for part in r_string.split(',')]
    return [f"{parts[i]}, {parts[i+1]}" for i in range(0, len(parts)-1, 2)]
def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.lower()  # lowercase
    name = re.sub(r'[-]', ' ', name)  # replace hyphens with spaces
    name = re.sub(r'\s+', ' ', name)  # collapse multiple spaces
    name = name.strip()
    return name
comparing_companies['Matched Companies'] = match_items_against_master(comparing_companies,'Funding Sources', no_dup_comp)
comparing_companies['Matched Chemicals'] = comparing_companies['Chemicals with InChIKey'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
comparing_companies['Matched Affiliations'] = match_items_against_master_aff(comparing_companies,'Affiliations', new_no_dup_aff)
comparing_companies['Researchers'] = comparing_companies['Authors'].apply(split_researchers)
comparing_companies['Aff'] = comparing_companies['Affiliations'].apply(
    lambda x: [item.strip() for item in x.split('|')] if isinstance(x, str) and x.strip() != '' else []
)
comparing_companies = comparing_companies.drop(['Affiliations','Funding Sources','Chemicals with InChIKey','Authors'],axis=1)

# having one company per row, with a list of affiliations and chemicals associated alongside them 

match_chem = []
for idx, (companies, chemicals) in comparing_companies[['Matched Companies', 'Matched Chemicals']].iterrows():
    for company in companies:
        for chemical in chemicals:
            match_chem.append({'Company': company, 'Chemical': chemical})

match_chem_df = pd.DataFrame(match_chem)
chemicals_per_company = (
    match_chem_df
    .groupby('Company')['Chemical']
    .unique()
    .reset_index()
    .rename(columns={'Chemical': 'Chemicals'})
)
matched_aff = []
for idx, (companies, affiliations) in comparing_companies[['Matched Companies', 'Matched Affiliations']].iterrows():
    for company in companies:
        for affiliation in affiliations:
            matched_aff.append({'Company': company, 'Affiliations': affiliation})

matched_aff_df = pd.DataFrame(matched_aff)

aff_per_company = (
    matched_aff_df
    .groupby('Company')['Affiliations']
    .unique()
    .reset_index()
)

comparing_companies['Names'] = comparing_companies['Researchers'].apply(
    lambda name_list: [normalize_name(name) for name in name_list]
)

comparing_companies['ResearcherAffPairs'] = comparing_companies.apply(
    lambda row: list(zip(row['Names'], row['Aff'])),
    axis=1
)

re_comp = comparing_companies.explode('ResearcherAffPairs')

re_comp[['Researcher', 'Aff']] = pd.DataFrame(
    re_comp['ResearcherAffPairs'].tolist(),
    index=re_comp.index
)

re_comp = re_comp.explode('Matched Companies')


re_comp = re_comp.rename(columns={'Matched Companies': 'Company'})

final_recomp = re_comp[['Company','Researcher', 'Aff']].reset_index(drop=True)

def normalize_comma_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip().lower()

    # Split into last and first name based on comma
    if ',' in name:
        last, first = [part.strip() for part in name.split(',', 1)]
    else:
        return name.title()  # fallback if format is unexpected

    # Capitalize each word, but leave fully uppercase words alone (e.g., acronyms)
    def smart_title(part):
        return ' '.join([
            word.capitalize() if not word.isupper() else word
            for word in part.split()
        ])

    return f"{smart_title(last)}, {smart_title(first)}"


final_recomp['Researcher'] = final_recomp['Researcher'].apply(normalize_comma_name)

res_per_comp = (
    final_recomp.groupby('Company')
      .agg({
          'Researcher': list,
          'Aff': list
      })
      .reset_index()
      .rename(columns={
          'Researcher': 'Researchers',
          'Aff': 'Affs'
      })
)

company_assoc = pd.merge(aff_per_company, chemicals_per_company, on ='Company')
company_assoc = pd.merge(company_assoc , res_per_comp, on = 'Company')


def extract_university_comp(affil, university_keys):
    if pd.isna(affil) or affil is None:
        return None
    affil = str(affil)
    if ',' in affil:
        found = False
        attributes = [a.strip() for a in affil.split(',')]
        for attr in attributes:
            for key in university_keys:
                if (key.lower() in attr.lower()) and (not any(char.isdigit() for char in attr)):
                    if ';' in attr:
                        att = [a.strip() for a in affil.split(';')]
                        for at in att:
                            if key.lower() in at.lower():
                                found = True
                                uni = at.strip()
                    else:
                        found = True 
                        uni = attr.strip()
        if found == True:
            return uni
        else:
            return None
def extract_uni_affil(affils, university_keys):
    universities = []
    for affil in affils:
        if extract_university_comp(affil,university_keys) != None:
            universities.append(extract_university_comp(affil,university_keys))
    return universities
company_assoc['Universities'] = company_assoc['Affiliations'].apply(lambda x: extract_uni_affil(x, university_keys))

# Creating a function that works for plotting a network graph with company at the middle

def extract_country_list(affiliation_list):
    countries_lists = []
    for aff in affiliation_list:
        attributes = [a.strip() for a in aff.split(',')]
        if attributes:
            last = attributes[-1].lower()
            match = next((country for country in country_names if country.lower() in last), 'Not Recognized')
            if 'chinese' in last:
                match = 'China'
            if match == 'Georgia':
                if 'Georgia Institute of Technology'.lower() in last:
                    match = 'United States'
            if match == 'Jersey':
                if 'New Jersey'.lower() in last:
                    match = 'United States'
            if match == 'Not Recognized':
                for abb in countries:
                    if abb.lower() == last:
                        match = countries[abb]
            countries_lists.append(match)
        else:
            countries_lists.append(None)  # in case the list is empty
    return countries_lists

company_assoc['Countries'] = company_assoc['Affiliations'].apply(extract_country_list)

organic_suffixes = [
    'ane', 'ene', 'yne','ol', 'diol', 'triol','al','one', 'anone','oic acid', 'carboxylic acid', 'anoate', 'oate','amide','amine','nitrile',    
    'thiol','ether','phenone','acid anhydride','imine'                      
]
carbon_prefixes = [
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl',
    'phenyl', 'benzyl', 'aryl','cyclo', 'bicyclo', 'spiro','iso', 'neo', 'sec', 'tert','alkyl', 'alkenyl', 'alkynyl',
]
aromatic_roots = ['benz', 'phen', 'tolu', 'naphth', 'styren']
def parse_chemical_entry(entry):
    match = re.search(r'^(.*)\s+\(([^()]*)\)\s*$', entry)
    if match:
        name, inchikey = match.groups()
        if inchikey.strip() != 'Error':
            return name.strip(), inchikey.strip()
        else:
            return name.strip(), 'Not Found'
    return entry.strip(), None
def is_likely_organic(name):
    name = name.lower()
    return (
        any(name.endswith(suffix) for suffix in organic_suffixes) or
        any(prefix in name for prefix in carbon_prefixes)
    )
def is_organic(name):
    try:
        compound = pcp.get_compounds(name, 'name')[0]
        formula = compound.molecular_formula
        return 'C' in formula and 'H' in formula
    except:
        return None  # Not found

def show_company_network_pyvis(company_name, category='Affiliations', chemical_group='All', sep_country=False, output_file=None):
    if output_file is None:
        # Generate unique filename based on ALL parameters
        safe_company = company_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_category = category.replace(' ', '_')
        
        if category == 'Chemicals':
            if chemical_group == 'All':
                output_file = f"networkviewer/static/network_{safe_company}_{safe_category}_all.html"
            elif chemical_group == 'Organic':
                output_file = f"networkviewer/static/network_{safe_company}_{safe_category}_organic.html"
        elif category == 'Affiliations':
            if sep_country:
                output_file = f"networkviewer/static/network_{safe_company}_{safe_category}_by_country.html"
            else:
                output_file = f"networkviewer/static/network_{safe_company}_{safe_category}_combined.html"
        else:
            # For Universities, Researchers, etc.
            output_file = f"networkviewer/static/network_{safe_company}_{safe_category}.html"
    # Filter for the selected company
    row = company_assoc[company_assoc['Company'] == company_name]
    if row.empty:
        print(f"Company '{company_name}' not found.")
        return False
    data = row.iloc[0][category]
    if sep_country == True and category == 'Affiliations':
        data = row.iloc[0][[category,'Countries']]
    if category == 'Chemicals':
        parsed_chems = (parse_chemical_entry(c) for c in data)
    if category == 'Researchers':
        res_list = row.iloc[0]['Researchers']
        aff_list = row.iloc[0]['Affs']
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add company node
    net.add_node(company_name, label=company_name, color="red", shape="ellipse", size=55)

    # Add affiliation nodes and edges
    if category == 'Chemicals':
        if chemical_group == 'All':
            # Add nodes first (keep existing logic)
            total_inchikeys = []
            for name, inchikey in parsed_chems:
                if inchikey:
                    if inchikey != 'Not Found':
                        if inchikey not in total_inchikeys:
                            net.add_node(
                                name,
                                label=name,
                                title=f"InChIKey: {inchikey}",
                                color='lightgreen',
                                shape='ellipse'
                            )
                            total_inchikeys.append(inchikey)
                        else:
                            total_inchikeys.append(inchikey)
                    else:
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        net.add_edge(company_name, name)
            
            # REPLACE InChIKey counting with study counting
            study_counts = {}
            for node in net.nodes:
                if node['id'] != company_name:  # Skip the company node itself
                    node_title = node.get('title', '')
                    if 'InChIKey:' in node_title:
                        inchikey = node_title.replace('InChIKey:', '').strip()
                        if inchikey and inchikey != 'Not Found':
                            # Count studies mentioning this InChIKey
                            studies = main[
                                (main['Funding Sources'].str.contains(company_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(inchikey, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name for chemicals without InChIKey
                            chemical_name = node.get('label', '')
                            studies = main[
                                (main['Funding Sources'].str.contains(company_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(chemical_name, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        
                        study_counts[node['id']] = study_count
                        net.add_edge(
                            company_name, 
                            node['id'], 
                            width=max(1, study_count),  # Minimum width of 1 for visibility
                            title=f"Studies: {study_count}"
                        )
        elif chemical_group =='Organic':
            added_inchikeys = []
            for name, inchikey in parsed_chems:
                if inchikey:
                    if ((is_organic(name) or is_likely_organic(name)) and (inchikey not in added_inchikeys)) and (inchikey != 'Not Found'):
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        added_inchikeys.append(inchikey)
                    elif inchikey in added_inchikeys:
                        added_inchikeys.append(inchikey)
                    elif (is_organic(name)) and (inchikey == 'Not Found'):
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        net.add_edge(company_name, name)
            
            # REPLACE InChIKey counting with study counting for organic chemicals
            study_counts = {}
            for node in net.nodes:
                if node['id'] != company_name:  # Skip the company node itself
                    node_title = node.get('title', '')
                    if 'InChIKey:' in node_title:
                        inchikey = node_title.replace('InChIKey:', '').strip()
                        if inchikey and inchikey != 'Not Found':
                            # Count studies mentioning this InChIKey
                            studies = main[
                                (main['Funding Sources'].str.contains(company_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(inchikey, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = main[
                                (main['Funding Sources'].str.contains(company_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(chemical_name, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        
                        study_counts[node['id']] = study_count
                        net.add_edge(
                            company_name, 
                            node['id'], 
                            width=max(1, study_count),
                            title=f"Studies: {study_count}"
                        )
    elif category == 'Affiliations':
        if sep_country == False:
            total_affil = []
            for affil in data:
                if affil not in total_affil:
                    found = False
                    short_label = affil
                    if ',' in affil:
                        attributes = affil.split(',')
                        for attr in attributes:
                            for key in university_keys:
                                if key.lower() in attr.lower():
                                    short_label = attr.strip()
                                    found = True
                    if (found == False) and (',' in affil):
                        short_label = attributes[0]
                    net.add_node(affil, label=short_label, title=affil, color="lightblue", shape="ellipse",size=15)
                    total_affil.append(affil)
                else:
                    total_affil.append(affil)
            study_counts = {}
            for node in net.nodes:
                if node['id'] != company_name:  # Skip the company node itself
                    affiliation = node.get('title', '')  # Full affiliation is in title
                    if affiliation:
                        # Count studies mentioning this affiliation
                        studies = main[
                            (main['Funding Sources'].str.contains(company_name, na=False)) &
                            (main['Affiliations'].str.contains(affiliation, na=False, regex=False))
                        ]
                        study_count = len(studies.drop_duplicates(subset=['DOI']))
                        
                        study_counts[node['id']] = study_count
                        net.add_edge(
                            company_name,
                            node['id'], 
                            width=max(1, study_count), 
                            title=f"Studies: {study_count}"
                        )
        elif sep_country == True:
            total_affil=[]
            aff_counts={}
            country_affil_counts = {}

            for affil, country in zip(data['Affiliations'], data['Countries']):
                total_affil.append(affil)
                aff_counts[affil] = aff_counts.get(affil,0)+1
                country_affil_counts[country] = country_affil_counts.get(country, 0) + 1

            for country in country_affil_counts:
                net.add_node(country, label=country, color='lightgreen', shape='box', size=20)
        
                # FIX: Use affiliation count instead of total study count
                affiliation_count = country_affil_counts[country]
                
                # Scale the edge width proportionally (max 10 for visual balance)
                scaled_width = min(affiliation_count, 10)
                
                net.add_edge(
                    company_name, 
                    country, 
                    width=max(1, scaled_width), 
                    title=f"Affiliations: {affiliation_count}"  # Show affiliation count instead
                )
                
            for affil,country in zip(data['Affiliations'], data['Countries']):
                found = False
                short_label = affil
                if ',' in affil:
                    attributes = affil.split(',')
                    for attr in attributes:
                        for key in university_keys:
                            if key.lower() in attr.lower():
                                short_label = attr.strip()
                                found = True
                    if not found:
                        short_label = attributes[0].strip()
                if not any(node['id'] ==  affil for node in net.nodes):
                    net.add_node(affil, label=short_label, title=affil, color='lightblue',shape='ellipse',size=15)
                
                # REPLACE affiliation counting with study counting
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Affiliations'].str.contains(affil, na=False, regex=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                
                net.add_edge(
                    country, 
                    affil,
                    width=max(1, study_count), 
                    title=f"Studies: {study_count}"
                )
    elif category == 'Universities':
        total_uni = []
        for uni in data:
            if uni not in total_uni:
                net.add_node(uni, label=uni, title=uni, color="lightblue", shape="ellipse",size=15)
                total_uni.append(uni)
            else:
                total_uni.append(uni)
        study_counts = {}
        for node in net.nodes:
            if node['id'] != company_name:  # Skip the company node itself
                university = node.get('title', '')  # University name is in title
                if university:
                    # Count studies mentioning this university
                    studies = main[
                        (main['Funding Sources'].str.contains(company_name, na=False)) &
                        (main['Affiliations'].str.contains(university, na=False))
                    ]
                    study_count = len(studies.drop_duplicates(subset=['DOI']))
                    
                    study_counts[node['id']] = study_count
                    net.add_edge(
                        company_name,
                        node['id'], 
                        width=max(1, study_count), 
                        title=f"Studies: {study_count}"
                    )
    elif category == 'Researchers':
        total_res = []
        for res, aff in zip(res_list, aff_list):
            if (res + '|' + aff[:20]) not in total_res:
                net.add_node(res,label=res,title=aff, color='lightblue',shape='ellipse',size = 15)
                total_res.append(res+'|'+aff[:20])
            else:
                total_res.append(res+'|'+aff[:20])
        study_counts = {}
        for node in net.nodes:
            if node['id'] != company_name:  # Skip the company node itself
                researcher = node.get('label', '')  # Researcher name is the label
                if researcher:
                    # Count studies mentioning this researcher
                    studies = main[
                        (main['Funding Sources'].str.contains(company_name, na=False)) &
                        (main['Authors'].str.contains(researcher, na=False))
                    ]
                    study_count = len(studies.drop_duplicates(subset=['DOI']))
                    
                    study_counts[node['id']] = study_count
                    net.add_edge(
                        company_name,
                        node['id'], 
                        width=max(1, study_count), 
                        title=f"Studies: {study_count}"
                    )
    num_nodes = len(net.nodes)

    net.options.interaction = {
    "zoomView": True,          
    "dragView": True,        
    "zoomSpeed": 0.00000000000000000000000000000000000000000000000000000000001,            
    "minZoom": 0.1,           
    "maxZoom": 4.0,           
    "wheelSensitivity": 0,    
    "hideEdgesOnDrag": False,
    "hideEdgesOnZoom": False,
    "keyboard": {
        "enabled": False,
        "bindToWindow": False
        }
    }
    if num_nodes > 100:
        net.options.physics.barnesHut.gravitationalConstant = -2000
        net.options.physics.barnesHut.springLength = 200
        net.options.physics.barnesHut.springConstant = 0.0005
    elif num_nodes > 40:
        net.options.physics.barnesHut.gravitationalConstant = -1200
        net.options.physics.barnesHut.springLength = 120
        net.options.physics.barnesHut.springConstant = 0.001
    else:
        net.options.physics.barnesHut.gravitationalConstant = -500
        net.options.physics.barnesHut.springLength = 60
        net.options.physics.barnesHut.springConstant = 0.002

    net.options.physics.minVelocity = 0.75
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    company_study_map = {}
    if category == 'Chemicals':
        parsed_chems = (parse_chemical_entry(c) for c in data)
        for name, inchikey in parsed_chems:
            key = name  # node label
            if inchikey and inchikey != 'Not Found':
                # Search by InChIKey if available
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Chemicals with InChIKey'].str.contains(inchikey, na=False))
                ]
            else:
                # Fallback to chemical name search
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Chemicals with InChIKey'].str.contains(name, na=False))
                ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection"
            company_study_map[key] = study_info
    elif category == 'Affiliations':
        if sep_country:
            # data is a DataFrame with columns 'Affiliations' and 'Countries'
            affiliations = data['Affiliations']
            countries = data['Countries']

            # Map studies for affiliation nodes (use full affiliation string as key)
            for affil in affiliations:
                affil_str = str(affil)
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Affiliations'].str.contains(affil_str, na=False, regex=False))
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[affil_str] = study_info


            # Optionally, map studies for country nodes (use country name as key)
            for country in countries:
                country_str = str(country)
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Affiliations'].str.contains(country_str, na=False, regex=False))
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[country_str] = study_info
        else:
            for affil in data:
                affil_str = str(affil)
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Affiliations'].str.contains(affil_str, na=False, regex=False))
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[affil_str] = study_info
    elif category =='Universities':
        for uni in data:
            studies = main[
                (main['Funding Sources'].str.contains(company_name, na = False)) &
                (main['Affiliations'].str.contains(uni, na = False))
            ]
            study_info = '<br>'.join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection"
            company_study_map[uni] = study_info
    elif category == 'Researchers':
        res_list = row.iloc[0]['Researchers']
        for res in res_list:
            studies = main[(
                main['Funding Sources'].str.contains(company_name, na = False)) &
                (main['Authors'].str.contains(res, na = False))
            ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection"
            company_study_map[res] = study_info
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    if category == 'Affiliations':
        js_lookup = "node.title"
    else:
        js_lookup = "node.label"

    injection = f"""
    <style>
        .zoom-controls {{
            margin: 10px 0;
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .zoom-btn {{
            padding: 10px 16px;
            margin: 4px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }}
        .zoom-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .zoom-in {{ background: #007bff; color: white; }}
        .zoom-out {{ background: #6c757d; color: white; }}
        .zoom-reset {{ background: #28a745; color: white; }}
    </style>
    <div class="zoom-controls">
        <button class="zoom-btn zoom-in" onclick="zoomIn()">🔍+ Zoom In</button>
        <button class="zoom-btn zoom-out" onclick="zoomOut()">🔍- Zoom Out</button>
        <button class="zoom-btn zoom-reset" onclick="resetZoom()">🎯 Reset View</button>
    </div>
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
        // Configure zoom options to disable scroll zoom
        network.setOptions({{
            interaction: {{
                zoomView: true,
                dragView: true,
                wheelSensitivity: 0,  // DISABLE scroll zoom
                minZoom: 0.05,
                maxZoom: 5.0
            }}
        }});
        
        // Zoom button functions
        function zoomIn() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.min(scale * 1.4, 5.0),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function zoomOut() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.max(scale * 0.7, 0.05),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function resetZoom() {{
            network.moveTo({{
                scale: 1.0,
                animation: {{duration: 600, easingFunction: 'easeInOutCubic'}}
            }});
        }}
        
        
        // Study click functionality
        var companyStudyMap = {json.dumps(company_study_map)};
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                var studies = companyStudyMap[{js_lookup}] || "No studies found for this connection.";
                document.getElementById("study-info").innerHTML = "<h3>Studies for " + {js_lookup} + ":</h3>" + studies;
            }}
        }});
    </script>
    """
    html = html.replace("</body>", injection + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True

'''
# Having the affiliations per row

cut_down = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Authors'], axis = 1)

cut_down['Affiliation'] = match_items_against_master_aff(cut_down, 'Affiliations', new_no_dup_aff)
cut_down.drop('Affiliations',axis=1)
cut_down['Chemicals'] = cut_down['Chemicals with InChIKey'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
cut_down['Companies'] = cut_down['Funding Sources'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
cut_down_exploded = cut_down.explode('Affiliation').reset_index(drop=True)
comparing_affiliations = cut_down_exploded.drop(['Affiliations','Funding Sources','Chemicals with InChIKey'], axis =1)

def extract_university(affil, university_keys):
    if pd.isna(affil) or affil is None:
        return None
    affil = str(affil)
    if ',' in affil:
        found = False
        attributes = [a.strip() for a in affil.split(',')]
        for attr in attributes:
            for key in university_keys:
                if (key.lower() in attr.lower()) and (not any(char.isdigit() for char in attr)):
                    if ';' in attr:
                        att = [a.strip() for a in affil.split(';')]
                        for at in att:
                            if key.lower() in at.lower():
                                found = True
                                uni = at.strip()
                    else:
                        found = True 
                        uni = attr.strip()
        if (found == False):
                    return ', '.join(attributes)  # fallback
        else:
            return uni
    return affil
comparing_affiliations['University'] = comparing_affiliations['Affiliation'].apply(lambda x: extract_university(x, university_keys))
comparing_unis = comparing_affiliations.groupby('University').agg({
    'Chemicals': lambda x: sum(x, []),   # Flattens with duplicates
    'Companies': lambda x: sum(x, [])
})
comparing_unis.reset_index(inplace = True)

comparing_unis['Companies'] = comparing_unis['Companies'].apply(classify_companies_series)

comparing_unis.to_csv(os.path.join(settings.BASE_DIR, 'data', 'comparing_unis.csv'), index=False)
'''
CSV_PATH_unis = os.path.join(settings.BASE_DIR, 'data', 'comparing_unis.csv')
comparing_unis = pd.read_csv(CSV_PATH_unis)
comparing_unis['Companies'] = comparing_unis['Companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
comparing_unis['Chemicals'] = comparing_unis['Chemicals'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
def show_uni_network_pyvis(uni_name, category='Funding Sources', chemical_group='All', output_file=None):
    if output_file is None:
        # Generate unique filename based on ALL parameters
        safe_uni = uni_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_category = category.replace(' ', '_')
        
        if category == 'Chemicals':
            if chemical_group == 'All':
                output_file = f"networkviewer/static/network_{safe_uni}_{safe_category}_all.html"
            elif chemical_group == 'Organic':
                output_file = f"networkviewer/static/network_{safe_uni}_{safe_category}_organic.html"
        else:
            # For Companies, etc.
            output_file = f"networkviewer/static/network_{safe_uni}_{safe_category}.html"    # Filter for the selected company
    row = comparing_unis[comparing_unis['University'] == uni_name]
    if row.empty:
        print(f"University '{uni_name}' not found.")
        return False
    if category == 'Funding Sources':
        data = row.iloc[0]['Companies']
    else:
        data = row.iloc[0][category]
    if category == 'Chemicals':
        parsed_chems = (parse_chemical_entry(c) for c in data)
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add company node
    net.add_node(uni_name, label=uni_name, color="red", shape="ellipse", size=55)

    # Add affiliation nodes and edges
    if category == 'Chemicals':
        if chemical_group == 'All':
            total_inchikeys = []
            for name, inchikey in parsed_chems:
                if inchikey:
                    if inchikey != 'Not Found':
                        if inchikey not in total_inchikeys:
                            net.add_node(
                                name,
                                label=name,
                                title=f"InChIKey: {inchikey}",
                                color='lightgreen',
                                shape='ellipse'
                                )
                            total_inchikeys.append(inchikey)
                        else:
                            total_inchikeys.append(inchikey)
                    else:
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        net.add_edge(uni_name, name)
            study_counts = {}
            for node in net.nodes:
                if node['id'] != uni_name:  # Skip the university node itself
                    node_title = node.get('title', '')
                    if 'InChIKey:' in node_title:
                        inchikey = node_title.replace('InChIKey:', '').strip()
                        if inchikey and inchikey != 'Not Found':
                            # Count studies mentioning this InChIKey at this university
                            studies = main[
                                (main['Affiliations'].str.contains(uni_name, na=False, regex=False)) &
                                (main['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = main[
                                (main['Affiliations'].str.contains(uni_name, na=False, regex=False)) &
                                (main['Chemicals with InChIKey'].str.contains(chemical_name, na=False, regex=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        
                        study_counts[node['id']] = study_count
                        net.add_edge(
                            uni_name, 
                            node['id'], 
                            width=max(1, study_count),
                            title=f"Studies: {study_count}"
                        )
        elif chemical_group =='Organic':
            added_inchikeys = []
            for name, inchikey in parsed_chems:
                if inchikey:
                    if ((is_organic(name) or is_likely_organic(name)) and (inchikey not in added_inchikeys)) and (inchikey != 'Not Found'):
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        added_inchikeys.append(inchikey)
                    elif inchikey in added_inchikeys:
                        added_inchikeys.append(inchikey)
                    elif (is_organic(name)) and (inchikey == 'Not Found'):
                        net.add_node(
                            name,
                            label=name,
                            title=f"InChIKey: {inchikey}",
                            color='lightgreen',
                            shape='ellipse'
                        )
                        net.add_edge(uni_name, name)
            study_counts = {}
            for node in net.nodes:
                if node['id'] != uni_name:  # Skip the university node itself
                    node_title = node.get('title', '')
                    if 'InChIKey:' in node_title:
                        inchikey = node_title.replace('InChIKey:', '').strip()
                        if inchikey and inchikey != 'Not Found':
                            # Count studies mentioning this InChIKey
                            studies = main[
                                (main['Affiliations'].str.contains(uni_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(inchikey, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = main[
                                (main['Affiliations'].str.contains(uni_name, na=False)) &
                                (main['Chemicals with InChIKey'].str.contains(chemical_name, na=False))
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        
                        study_counts[node['id']] = study_count
                        net.add_edge(
                            uni_name, 
                            node['id'], 
                            width=max(1, study_count),
                            title=f"Studies: {study_count}"
                        )
    if category == 'Funding Sources':
        total_comp = []
        entity_stats = {}
        for comp in data:
            if comp not in total_comp:
                original_name, entity_category = extract_name_and_class(comp)
                entity_color = get_category_color(entity_category)
                entity_stats[entity_category] = entity_stats.get(entity_category, 0) + 1
                net.add_node(
                    original_name,
                    label=original_name,
                    title=f"{original_name}\n Category: {entity_category}",
                    color=entity_color,
                    shape="ellipse",
                    size=15
                )
                total_comp.append(comp)
            else:
                _, entity_category = extract_name_and_class(comp)
                entity_stats[entity_category] = entity_stats.get(entity_category, 0) + 1
                total_comp.append(comp)
        study_counts = {}
        for node in net.nodes:
            if node['id'] != uni_name:  # Skip the university node itself
                company = node['id']  # Company name is in title
                if company:
                    # Count studies mentioning this company at this university
                    studies = main[
                        (main['Affiliations'].str.contains(uni_name, na=False, regex=False)) &
                        (main['Funding Sources'].str.contains(company, na=False, regex=False))
                    ]
                    study_count = len(studies.drop_duplicates(subset=['DOI']))
                    
                    study_counts[node['id']] = study_count
                    net.add_edge(
                        uni_name,
                        node['id'], 
                        width=max(1, study_count), 
                        title=f"Studies: {study_count}"
                    )
    num_nodes = len(net.nodes)

    net.options.interaction = {
    "zoomView": True,          
    "dragView": True,        
    "zoomSpeed": 0.00000000000000000000000000000000000000000000000000000000001,            
    "minZoom": 0.1,           
    "maxZoom": 4.0,           
    "wheelSensitivity": 0,    
    "hideEdgesOnDrag": False,
    "hideEdgesOnZoom": False,
    "keyboard": {
        "enabled": False,
        "bindToWindow": False
        }
    }

    if num_nodes > 100:
        net.options.physics.barnesHut.gravitationalConstant = -2000
        net.options.physics.barnesHut.springLength = 200
        net.options.physics.barnesHut.springConstant = 0.0005
    elif num_nodes > 40:
        net.options.physics.barnesHut.gravitationalConstant = -1200
        net.options.physics.barnesHut.springLength = 120
        net.options.physics.barnesHut.springConstant = 0.001
    else:
        net.options.physics.barnesHut.gravitationalConstant = -500
        net.options.physics.barnesHut.springLength = 60
        net.options.physics.barnesHut.springConstant = 0.002

    net.options.physics.minVelocity = 0.75
    company_study_map = {}
    if category =='Funding Sources':
        for comp in data:
            original_name, _ = extract_name_and_class(comp)
            studies = main[
                (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
                (main['Affiliations'].str.contains(uni_name, na=False, regex=False))
            ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection."
            company_study_map[original_name] = study_info
    elif category == 'Chemicals':
        parsed_chems = [parse_chemical_entry(c) for c in data]
        for name, inchikey in parsed_chems:
            key = name  # node label is the chemical name
            if inchikey and inchikey != 'Not Found':
                # Search by InChIKey if available
                studies = main[
                    (main['Affiliations'].str.contains(uni_name, na=False, regex=False)) &
                    (main['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False))
                ]
            else:
                # Fallback to chemical name search
                studies = main[
                    (main['Affiliations'].str.contains(uni_name, na=False, regex=False)) &
                    (main['Chemicals with InChIKey'].str.contains(name, na=False, regex=False))
                ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.drop_duplicates(subset=['DOI']).iterrows()
            ) or "No studies found for this connection."
            company_study_map[key] = study_info
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()
    color_legend = ""
    if category == 'Funding Sources':
        color_legend = """
        <div class="color-legend" style="flex: 1; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-right: 10px;">
            <h4 style="margin-bottom: 10px; color: #333; font-size: 16px;">Funding Source Categories:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FF6B6B; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #96CEB4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #4ECDC4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFEAA7; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #DDD6FE; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Unknown</span>
                </div>
            </div>
        </div>
        """
    injection = f"""
    <style>
        .controls-container {{
            display: flex;
            margin: 10px 0;
            gap: 0;
            align-items:stretch;
        }}
        .zoom-controls {{
            flex: 0 0 auto;
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            min-width:300px;
        }}
        .zoom-btn {{
            padding: 10px 16px;
            margin: 4px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }}
        .zoom-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .zoom-in {{ background: #007bff; color: white; }}
        .zoom-out {{ background: #6c757d; color: white; }}
        .zoom-reset {{ background: #28a745; color: white; }}
    </style>
    <div class="controls-container">
        {color_legend}
        <div class="zoom-controls">
            <button class="zoom-btn zoom-in" onclick="zoomIn()">🔍+ Zoom In</button>
            <button class="zoom-btn zoom-out" onclick="zoomOut()">🔍- Zoom Out</button>
            <button class="zoom-btn zoom-reset" onclick="resetZoom()">🎯 Reset View</button>
        </div>
    </div>
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
        // Configure zoom options to disable scroll zoom
        network.setOptions({{
            interaction: {{
                zoomView: true,
                dragView: true,
                wheelSensitivity: 0,  // DISABLE scroll zoom
                minZoom: 0.05,
                maxZoom: 5.0
            }}
        }});
        
        // AGGRESSIVE SCROLL DISABLE
        setTimeout(function() {{
            var visContainers = document.querySelectorAll('.vis-network');
            visContainers.forEach(function(container) {{
                container.addEventListener('wheel', function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }}, {{ passive: false }});
            }});
        }}, 1000);
        
        // Zoom button functions
        function zoomIn() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.min(scale * 1.4, 5.0),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function zoomOut() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.max(scale * 0.7, 0.05),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function resetZoom() {{
            network.moveTo({{
                scale: 1.0,
                animation: {{duration: 600, easingFunction: 'easeInOutCubic'}}
            }});
        }}
        
        // Study click functionality
        var companyStudyMap = {json.dumps(company_study_map)};
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                var studies = companyStudyMap[node.label] || "No studies found for this connection.";
                document.getElementById("study-info").innerHTML = "<h3>Studies for " + node.label + ":</h3>" + studies;
            }}
        }});
    </script>
    """
    html = html.replace("</body>", injection + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True

# showing researchers and their company funding
'''
reduced = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Chemicals with InChIKey'], axis = 1)

reduced['Researchers'] = reduced['Authors'].apply(split_researchers)
reduced['Aff'] = reduced['Affiliations'].apply(
    lambda x: [item.strip() for item in x.split('|')] if isinstance(x, str) and x.strip() != '' else []
)
reduced['Companies'] = reduced['Funding Sources'].str.split(';').apply(lambda lst: [x.strip() for x in lst])
reduced = reduced.drop(['Authors','Affiliations','Funding Sources'],axis=1)

reduced['ResearcherAffPairs'] = reduced.apply(
    lambda row: list(zip(row['Researchers'], row['Aff'])),
    axis=1
)


reduced_expanded = reduced.explode('ResearcherAffPairs')


reduced_expanded[['Researcher', 'Affiliation']] = pd.DataFrame(
    reduced_expanded['ResearcherAffPairs'].tolist(),
    index=reduced_expanded.index
)

final_reduced = reduced_expanded[['Researcher', 'Affiliation', 'Companies']].reset_index(drop=True)

final_reduced['NormalizedName'] = final_reduced['Researcher'].apply(normalize_name)


final_reduced['GroupKey'] = final_reduced['NormalizedName'] + '|' + final_reduced['Affiliation'].str[:20]

comparing_researchers = final_reduced.groupby('GroupKey').agg({
    'Researcher': 'first',
    'Affiliation': lambda affs: max(affs, key=len),  # longest affiliation
    'Companies': lambda lists: sum(lists, [])        # flatten company lists
}).reset_index(drop=True)

comparing_researchers['Companies'] = comparing_researchers['Companies'].apply(classify_companies_series)

comparing_researchers.to_csv(os.path.join(settings.BASE_DIR, 'data', 'comparing_researchers.csv'), index=False)
'''
CSV_PATH_researchers = os.path.join(settings.BASE_DIR, 'data', 'comparing_researchers.csv')
comparing_researchers = pd.read_csv(CSV_PATH_researchers)
comparing_researchers['Companies'] = comparing_researchers['Companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

def show_researcher_network_pyvis(researcher, output_file = "networkviewer/static/company_network.html"):    # Filter for the selected company
    # Filter for the selected company
    matches = comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]
    
    if matches.empty:
        print(f"Researcher: '{researcher}' not found.")
        return False
    if len(matches) > 1:
        print(f"\nMultiple entries found for '{researcher}':\n")
        for i, row in matches.iterrows():
            print(f"[{i}] Affiliation: {row['Affiliation']}, Companies: {', '.join(row['Companies'])}")
        print("[c] Combine all entries")
    
        while True:
            choice = input("\nEnter the number of the entry you'd like to graph, or 'c' to combine all: ").strip().lower()
            
            if choice == 'c':
                # Combine all companies and pick the longest affiliation
                all_companies = sum(matches['Companies'], [])
                unique_affiliations = matches['Affiliation'].dropna().unique()
                combined_aff = '; '.join(unique_affiliations)
                row = {
                    'Researcher': researcher,
                    'Affiliation': combined_aff,
                    'Companies': all_companies
                }
                break
            elif choice.isdigit() and int(choice) in matches.index:
                row = matches.loc[int(choice)]
                break
            else:
                print("❌ Invalid selection. Please enter a valid number or 'c'.")
                continue  # keep prompting until valid input
    if len(matches) == 1:
        row = matches.iloc[0]
    data = row['Companies']
    aff = row['Affiliation']
    if aff == '':
        aff = 'Not Found'
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add company node
    net.add_node(researcher, label=researcher, title=f"Affiliation: {aff}",color="red", shape="ellipse", size=55)

    # Add affiliation nodes and edges
    total_comp = []
    for affil in data:
        if affil not in total_comp:
            net.add_node(affil, label=affil, title=affil, color="lightblue", shape="ellipse",size=15)
            total_comp.append(affil)
        else:
            total_comp.append(affil)
    study_counts = {}
    for node in net.nodes:
        if node['id'] != researcher:  # Skip the researcher node itself
            company = node.get('label', '')  # Company name is the label
            if company:
                # Count studies mentioning this researcher with this company
                studies = main[
                    (main['Funding Sources'].str.contains(company, na=False)) &
                    (main['Authors'].str.contains(researcher, na=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                
                study_counts[node['id']] = study_count
                net.add_edge(
                    researcher,
                    node['id'], 
                    width=max(1, study_count), 
                    title=f"Studies: {study_count}"
                )
    num_nodes = len(net.nodes)

    
    if num_nodes > 100:
        net.options.physics.barnesHut.gravitationalConstant = -2000
        net.options.physics.barnesHut.springLength = 200
        net.options.physics.barnesHut.springConstant = 0.0005
    elif num_nodes > 40:
        net.options.physics.barnesHut.gravitationalConstant = -1200
        net.options.physics.barnesHut.springLength = 120
        net.options.physics.barnesHut.springConstant = 0.001
    else:
        net.options.physics.barnesHut.gravitationalConstant = -500
        net.options.physics.barnesHut.springLength = 60
        net.options.physics.barnesHut.springConstant = 0.002

    net.options.physics.minVelocity = 0.75
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    net.show(output_file)
    return True


# Creating a dataframe that has chemicals per row
'''
red_chem = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract','Authors','Affiliations'], axis = 1)
def parse_chemicals(chem_string):
    chemicals = []
    for entry in chem_string.split(';'):
        entry = entry.strip()
        parts = re.findall(r'\(([^()]*)\)', entry)
        if parts:
            inchikey = parts[-1]  # Last set of parentheses is likely InChIKey
            name = entry[:entry.rfind('(')].strip()
            chemicals.append((name, inchikey))
        else:
            chemicals.append((entry, None))  # No InChIKey found
    return chemicals
def parse_companies(company_string):
    return [c.strip() for c in company_string.split(';') if c.strip()]

records = []
for idx, row in red_chem.iterrows():
    chemicals = parse_chemicals(row['Chemicals with InChIKey'])
    companies = parse_companies(row['Funding Sources'])
    
    for chem_name, inchikey in chemicals:
        for company in companies:
            records.append({
                'chemical': chem_name,
                'inchikey': inchikey,
                'company': company
            })
flat_red_chem = pd.DataFrame(records)
flat_red_chem['group_key'] = flat_red_chem.apply(
    lambda row: row['inchikey'] if row['inchikey'] != 'Error' else row['chemical'],
    axis=1
)

chem_per_row = (
    flat_red_chem
    .groupby(['group_key'])  # Smart key: inchikey or chemical name
    .agg({
        'inchikey': 'first',  # Retain original InChIKey (or Error)
        'chemical': lambda names: sorted(set(names)),  # All name variants
        'company': list  # All associated companies
    })
    .reset_index(drop=True)
)
chem_per_row['company'] = chem_per_row['company'].apply(classify_companies_series)
chem_per_row.to_csv(os.path.join(settings.BASE_DIR, 'data', 'chem_per_row.csv'), index=False)
'''
CSV_PATH_chem = os.path.join(settings.BASE_DIR, 'data', 'chem_per_row.csv')
chem_per_row = pd.read_csv(CSV_PATH_chem)
chem_per_row['company'] = chem_per_row['company'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
chem_per_row['chemical'] = chem_per_row['chemical'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

def show_chemical_network(chemical, inch='Error', output_file=None):
    if output_file is None:
        safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        if inch != 'Error':
            safe_inch = inch.replace('/', '_').replace('\\', '_').replace('-', '_')
            output_file = f"networkviewer/static/network_{safe_chemical}_{safe_inch}.html"
        else:
            output_file = f"networkviewer/static/network_{safe_chemical}_no_inchikey.html"
    # Filter for the selected company
    if inch == 'Error':
        row = chem_per_row[chem_per_row['chemical'].apply(lambda x: any(chemical.lower() == name.lower() for name in x))]
        if row.empty:
            print(f"Chemical '{chemical}' not found.")
            return False
        inchikey = row.iloc[0]['inchikey']
        if inchikey and inchikey != 'Error':
            inch = inchikey
            chemical = row.iloc[0]['chemical'][0]
    else:
        row = chem_per_row[chem_per_row['inchikey'] == inch]
        if row.empty:
            print(f"InChIKey '{inch}' not found.")
            return False
        chemical = row.iloc[0]['chemical'][0]
        inchikey = inch
    data = row.iloc[0]['company']
    if inchikey == 'Error':
        inchikey = 'Not Found'
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add company node
    net.add_node(chemical, label=chemical, title=f"Inchikey: {inchikey}",color="lightgreen", shape="ellipse", size=55)

    # Add affiliation nodes and edges
    total_comp = []
    for comp in data:
        if comp not in total_comp:
            original_name, entity_category = extract_name_and_class(comp)
            entity_color = get_category_color(entity_category)
            net.add_node(
                    original_name,
                    label=original_name,
                    title=f"{original_name}\n Category: {entity_category}",
                    color=entity_color,
                    shape="ellipse",
                    size=15
                )
            total_comp.append(comp)
        else:
            total_comp.append(comp)
    study_counts = {}
    for node in net.nodes:
        if node['id'] != chemical:  # Skip the chemical node itself
            company = node.get('id')  # Company name is the id
            if company:
                # Count studies mentioning this chemical with this company
                if inch and inch != 'Error' and inch != 'Not Found':
                    # Use InChIKey for search
                    studies = main[
                        (main['Funding Sources'].str.contains(company, na=False, regex=False)) &
                        (main['Chemicals with InChIKey'].str.contains(inch, na=False, regex=False))
                    ]
                else:
                    # Fallback to chemical name
                    studies = main[
                        (main['Funding Sources'].str.contains(company, na=False, regex=False)) &
                        (main['Chemicals with InChIKey'].str.contains(chemical, na=False, regex=False))
                    ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                
                study_counts[node['id']] = study_count
                net.add_edge(
                    chemical,
                    node['id'], 
                    width=max(1, study_count), 
                    title=f"Studies: {study_count}",
                    color='red'
                )
    num_nodes = len(net.nodes)

    net.options.interaction = {
    "zoomView": True,          
    "dragView": True,        
    "zoomSpeed": 0.00000000000000000000000000000000000000000000000000000000001,            
    "minZoom": 0.1,           
    "maxZoom": 4.0,           
    "wheelSensitivity": 0,    
    "hideEdgesOnDrag": False,
    "hideEdgesOnZoom": False,
    "keyboard": {
        "enabled": False,
        "bindToWindow": False
        }
    }

    if num_nodes > 100:
        net.options.physics.barnesHut.gravitationalConstant = -2000
        net.options.physics.barnesHut.springLength = 200
        net.options.physics.barnesHut.springConstant = 0.0005
    elif num_nodes > 40:
        net.options.physics.barnesHut.gravitationalConstant = -1200
        net.options.physics.barnesHut.springLength = 120
        net.options.physics.barnesHut.springConstant = 0.001
    else:
        net.options.physics.barnesHut.gravitationalConstant = -500
        net.options.physics.barnesHut.springLength = 60
        net.options.physics.barnesHut.springConstant = 0.002

    net.options.physics.minVelocity = 0.75
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    company_study_map = {}
    for comp in data:
        original_name, entity_category = extract_name_and_class(comp)
        if inch and inch != 'Error' and inch != 'Not Found':
            studies = main[
                (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
                (main['Chemicals with InChIKey'].str.contains(inch, na=False))
            ]
        else:
            # fallback, but this should rarely happen
            studies = main[
                (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
                (main['Chemicals with InChIKey'].str.contains(chemical, na=False, regex=False))
            ]
        study_info = "<br>".join(
            f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
        ) or "No studies found for this connection."
        company_study_map[original_name] = study_info
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()
    color_legend = """
        <div class="color-legend" style="flex: 1; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-right: 10px;">
            <h4 style="margin-bottom: 10px; color: #333; font-size: 16px;">Funding Source Categories:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FF6B6B; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #96CEB4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #4ECDC4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFEAA7; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #DDD6FE; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Unknown</span>
                </div>
            </div>
        </div>
    """
    injection = f"""
    <style>
        .controls-container {{
            display: flex;
            margin: 10px 0;
            gap: 0;
            align-items:stretch;
        }}
        .zoom-controls {{
            flex: 0 0 auto;
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            min-width:300px;
        }}
        .zoom-btn {{
            padding: 10px 16px;
            margin: 4px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }}
        .zoom-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .zoom-in {{ background: #007bff; color: white; }}
        .zoom-out {{ background: #6c757d; color: white; }}
        .zoom-reset {{ background: #28a745; color: white; }}
    </style>
    <div class="controls-container">
        {color_legend}
        <div class="zoom-controls">
            <button class="zoom-btn zoom-in" onclick="zoomIn()">🔍+ Zoom In</button>
            <button class="zoom-btn zoom-out" onclick="zoomOut()">🔍- Zoom Out</button>
            <button class="zoom-btn zoom-reset" onclick="resetZoom()">🎯 Reset View</button>
        </div>
    </div>
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
        // Configure zoom options to disable scroll zoom
        network.setOptions({{
            interaction: {{
                zoomView: true,
                dragView: true,
                wheelSensitivity: 0,  // DISABLE scroll zoom
                minZoom: 0.05,
                maxZoom: 5.0
            }}
        }});
        
        // AGGRESSIVE SCROLL DISABLE
        setTimeout(function() {{
            var visContainers = document.querySelectorAll('.vis-network');
            visContainers.forEach(function(container) {{
                container.addEventListener('wheel', function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }}, {{ passive: false }});
            }});
        }}, 1000);
        
        // Zoom button functions
        function zoomIn() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.min(scale * 1.4, 5.0),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function zoomOut() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.max(scale * 0.7, 0.05),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function resetZoom() {{
            network.moveTo({{
                scale: 1.0,
                animation: {{duration: 600, easingFunction: 'easeInOutCubic'}}
            }});
        }}
        
        // Study click functionality
        var companyStudyMap = {json.dumps(company_study_map)};
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                var studies = companyStudyMap[node.label] || "No studies found for this connection.";
                document.getElementById("study-info").innerHTML = "<h3>Studies for " + node.label + ":</h3>" + studies;
            }}
        }});
    </script>
    """
    html = html.replace("</body>", injection + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True

def show_researcher_network_pyvis_from_row(row, output_file=None):
    if output_file is None:
        researcher = row['Researcher']
        safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
        # Use first 20 chars of affiliation to make filename more unique
        safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        output_file = f"networkviewer/static/network_{safe_researcher}_{safe_aff}.html"
    data = row['Companies']
    aff = row['Affiliation']
    researcher = row['Researcher']
    if aff == '':
        aff = 'Not Found'
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add researcher node
    net.add_node(researcher, label=researcher, title=f"Affiliation: {aff}", color="red", shape="ellipse", size=55)

    # Add company nodes and edges
    total_comp = []
    for comp in data:
        if comp not in total_comp:
            original_name, entity_type = extract_name_and_class(comp)
            entity_color = get_category_color(entity_type)
            net.add_node(
            original_name,
            label=original_name,
            title=f"{original_name}\nCategory: {entity_type}",
            color=entity_color,
            shape="ellipse",
            size=15
            )
            total_comp.append(comp)
        else:
            total_comp.append(comp)
    study_counts = {}
    for node in net.nodes:
        if node['id'] != researcher:  # Skip the researcher node itself
            company = node.get('id')  # Company name is the label
            if company:
                # Count studies mentioning this researcher with this company
                studies = main[
                    (main['Funding Sources'].str.contains(company, na=False)) &
                    (main['Authors'].str.contains(researcher, na=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                
                study_counts[node['id']] = study_count
                net.add_edge(
                    researcher,
                    node['id'], 
                    width=max(1, study_count), 
                    title=f"Studies: {study_count}"
                )
    num_nodes = len(net.nodes)

    net.options.interaction = {
    "zoomView": True,          
    "dragView": True,        
    "zoomSpeed": 0.00000000000000000000000000000000000000000000000000000000001,            
    "minZoom": 0.1,           
    "maxZoom": 4.0,           
    "wheelSensitivity": 0,    
    "hideEdgesOnDrag": False,
    "hideEdgesOnZoom": False,
    "keyboard": {
        "enabled": False,
        "bindToWindow": False
    }
}
    
    if num_nodes > 100:
        net.options.physics.barnesHut.gravitationalConstant = -2000
        net.options.physics.barnesHut.springLength = 200
        net.options.physics.barnesHut.springConstant = 0.0005
    elif num_nodes > 40:
        net.options.physics.barnesHut.gravitationalConstant = -1200
        net.options.physics.barnesHut.springLength = 120
        net.options.physics.barnesHut.springConstant = 0.001
    else:
        net.options.physics.barnesHut.gravitationalConstant = -500
        net.options.physics.barnesHut.springLength = 60
        net.options.physics.barnesHut.springConstant = 0.002

    net.options.physics.minVelocity = 0.75
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    company_study_map = {}
    for comp in data:
        original_name, _ = extract_name_and_class(comp)
        studies = main[
            (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
            (main['Authors'].str.contains(researcher, na=False, regex=False))
        ]
        study_info = "<br>".join(
            f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.drop_duplicates(subset=['DOI']).iterrows()
        ) or "No studies found for this connection."
        company_study_map[original_name] = study_info
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()
    color_legend = """
        <div class="color-legend" style="flex: 1; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-right: 10px;">
            <h4 style="margin-bottom: 10px; color: #333; font-size: 16px;">Funding Source Categories:</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FF6B6B; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #96CEB4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #4ECDC4; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFEAA7; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #DDD6FE; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Unknown</span>
                </div>
            </div>
        </div>
        """
    injection = f"""
    <style>
        .controls-container {{
            display: flex;
            margin: 10px 0;
            gap: 0;
            align-items: stretch;
        }}
        .zoom-controls {{
            margin: 10px 0;
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .zoom-btn {{
            padding: 10px 16px;
            margin: 4px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }}
        .zoom-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .zoom-in {{ background: #007bff; color: white; }}
        .zoom-out {{ background: #6c757d; color: white; }}
        .zoom-reset {{ background: #28a745; color: white; }}
    </style>
    </style>
    <div class="controls-container">
        {color_legend}
        <div class="zoom-controls">
            <button class="zoom-btn zoom-in" onclick="zoomIn()">🔍+ Zoom In</button>
            <button class="zoom-btn zoom-out" onclick="zoomOut()">🔍- Zoom Out</button>
            <button class="zoom-btn zoom-reset" onclick="resetZoom()">🎯 Reset View</button>
        </div>
    </div>
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
        // Configure zoom options to disable scroll zoom
        network.setOptions({{
            interaction: {{
                zoomView: true,
                dragView: true,
                wheelSensitivity: 0,  // DISABLE scroll zoom
                minZoom: 0.05,
                maxZoom: 5.0
            }}
        }});
        
        // AGGRESSIVE SCROLL DISABLE
        setTimeout(function() {{
            var visContainers = document.querySelectorAll('.vis-network');
            visContainers.forEach(function(container) {{
                container.addEventListener('wheel', function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                }}, {{ passive: false }});
            }});
        }}, 1000);
        
        // Zoom button functions
        function zoomIn() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.min(scale * 1.4, 5.0),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function zoomOut() {{
            var scale = network.getScale();
            network.moveTo({{
                scale: Math.max(scale * 0.7, 0.05),
                animation: {{duration: 400, easingFunction: 'easeOutCubic'}}
            }});
        }}
        
        function resetZoom() {{
            network.moveTo({{
                scale: 1.0,
                animation: {{duration: 600, easingFunction: 'easeInOutCubic'}}
            }});
        }}
        
        // Study click functionality
        var companyStudyMap = {json.dumps(company_study_map)};
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                var studies = companyStudyMap[node.label] || "No studies found for this connection.";
                document.getElementById("study-info").innerHTML = "<h3>Studies for " + node.label + ":</h3>" + studies;
            }}
        }});
    </script>
    """
    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>")
    else:
        html += injection

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True

def show_company_connections(company_name):
    row = company_assoc[company_assoc['Company'] == company_name]
    if row.empty:
        print(f"Company '{company_name}' not found.")
        return False

    affiliations = row.iloc[0]['Affiliations']
    countries = row.iloc[0]['Countries']
    parsed_chems = list(parse_chemical_entry(c) for c in row.iloc[0]['Chemicals'])
    res_list = row.iloc[0]['Researchers']
    aff_list = row.iloc[0]['Affs']
    universities = row.iloc[0]['Universities']

    # Chemicals
    labeled_chemicals = []
    processed_inchikeys = set()
    
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            if inchikey not in processed_inchikeys:
                # Chemicals with InChIKey
                studies = main[
                    (main['Funding Sources'].str.contains(company_name, na=False)) &
                    (main['Chemicals with InChIKey'].str.contains(inchikey, na=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                processed_inchikeys.add(inchikey)
        else:
            # Chemicals without InChIKey
            studies = main[
                (main['Funding Sources'].str.contains(company_name, na=False)) &
                (main['Chemicals with InChIKey'].str.contains(name, na=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_chemicals.append(f"{name} ({study_count})")

    # Countries
    unique_countries = []
    labeled_countries = []
    country_affil_counts = {}
    
    for country in countries:
        country_affil_counts[country] = country_affil_counts.get(country, 0) + 1

    for country in country_affil_counts:
        if country not in unique_countries:    
            affiliation_count = country_affil_counts[country]
            labeled_countries.append(f"{country} ({affiliation_count} affiliations)")
            unique_countries.append(country)

    # Affiliations 
    unique_affiliations = []
    labeled_affiliations = []
    
    for affil in affiliations:
        if affil not in unique_affiliations:
            studies = main[
                (main['Funding Sources'].str.contains(company_name, na=False)) &
                (main['Affiliations'].str.contains(affil, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_affiliations.append(f"{affil} ({study_count})")
            unique_affiliations.append(affil)

    # Researchers
    unique_researchers = []
    labeled_researchers = []
    
    for res in res_list:
        if res not in unique_researchers:
            studies = main[
                (main['Funding Sources'].str.contains(company_name, na=False)) &
                (main['Authors'].str.contains(res, na=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_researchers.append(f"{res} ({study_count})")
            unique_researchers.append(res)

    # Universities
    unique_universities = []
    labeled_universities = []
    
    for uni in universities:
        if uni not in unique_universities:
            # Count studies mentioning this university
            studies = main[
                (main['Funding Sources'].str.contains(company_name, na=False)) &
                (main['Affiliations'].str.contains(uni, na=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_universities.append(f"{uni} ({study_count})")
            unique_universities.append(uni)

    return {
        "Affiliations": labeled_affiliations,
        "Countries": labeled_countries,
        "Researchers": labeled_researchers,
        "Universities": labeled_universities,
        "Chemicals": labeled_chemicals
    }

def show_uni_connections(university):
    row = comparing_unis[comparing_unis['University'] == university]
    if row.empty:
        print(f"University '{university}' not found.")
        return False

    parsed_chems = list(parse_chemical_entry(c) for c in row.iloc[0]['Chemicals'])
    companies = row.iloc[0]['Companies']

    # Chemicals
    labeled_chemicals = []
    processed_inchikeys = set()
    
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            if inchikey not in processed_inchikeys:
                # Chemicals with InChIKey
                studies = main[
                    (main['Affiliations'].str.contains(university, na=False, regex=False)) &
                    (main['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                processed_inchikeys.add(inchikey)
        else:
            # Chemicals without InChIKey
            studies = main[
                (main['Affiliations'].str.contains(university, na=False, regex=False)) &
                (main['Chemicals with InChIKey'].str.contains(name, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_chemicals.append(f"{name} ({study_count})")

    # Companies
    unique_companies = []
    labeled_companies = []
    
    for comp in companies:
        original_name, _ = extract_name_and_class(comp)
        if original_name not in unique_companies:
            studies = main[
                (main['Affiliations'].str.contains(university, na=False)) &
                (main['Funding Sources'].str.contains(original_name, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{comp} ({study_count})")
            unique_companies.append(original_name)

    return {
        "Funding Sources": labeled_companies,
        "Chemicals": labeled_chemicals
    }

def show_res_connections(researcher):
    matches = comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]
    if matches.empty:
        print(f"Researcher: '{researcher}' not found.")
        return False
    
    if len(matches) > 1:
        all_companies = sum(matches['Companies'], [])
        unique_affiliations = matches['Affiliation'].dropna().unique()
        combined_aff = '; '.join(unique_affiliations)
        row = {
            'Researcher': researcher,
            'Affiliation': combined_aff,
            'Companies': all_companies
        }
    else:
        row = matches.iloc[0]

    data = row['Companies']
    aff = row['Affiliation']
    if aff == '':
        aff = 'Not Found'

    # Companies
    unique_companies = []
    labeled_companies = []
    
    for comp in data:
        original_name, _ = extract_name_and_class(comp)
        if original_name not in unique_companies:
            studies = main[
                (main['Authors'].str.contains(researcher, na=False, regex=False)) &
                (main['Funding Sources'].str.contains(original_name, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{comp} ({study_count})")
            unique_companies.append(comp)

    return {
        "Affiliation(s)": aff,
        "Funding Sources": labeled_companies
    }

def show_chem_connections(chemical=None, inchikey=None):
    if inchikey:
        row = chem_per_row[chem_per_row['inchikey'] == inchikey]
    elif chemical:
        row = chem_per_row[chem_per_row['chemical'].apply(lambda x: any(chemical.lower() == name.lower() for name in x))]
    else:
        return False
    
    if row.empty:
        print(f"Chemical '{chemical}' not found.")
        return False
    
    data = row.iloc[0]['company']
    inchikey_val = row.iloc[0]['inchikey']
    if inchikey_val == 'Error':
        inchikey_val = 'Not Found'

    # Companies
    unique_companies = []
    labeled_companies = []
    
    for comp in data:
        original_name, _ = extract_name_and_class(comp)
        if original_name not in unique_companies:
            if inchikey_val and inchikey_val != 'Not Found':
                studies = main[
                    (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
                    (main['Chemicals with InChIKey'].str.contains(inchikey_val, na=False, regex=False))
                ]
            else:
                studies = main[
                    (main['Funding Sources'].str.contains(original_name, na=False, regex=False)) &
                    (main['Chemicals with InChIKey'].str.contains(chemical, na=False, regex=False))
                ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{comp} ({study_count})")
            unique_companies.append(comp)

    return {
        "Inchikey": inchikey_val,
        "Funding Sources": labeled_companies
    }
