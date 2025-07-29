import pandas as pd
from pyvis.network import Network
import re
from country_list import countries_for_language
import pubchempy as pcp
from collections import defaultdict
import os
import json
import numpy as np

main = pd.read_csv(r'C:\Users\pelom\Downloads\esandt_papers_2024_with_inchikeys.csv')

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
university_keys = ['Academy of Sciences','chinese academy of sciences','institute','university','instituto','Universidad','Universita','Universit']

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
    
def show_company_network_pyvis(company_name, category='Affiliations', chemical_group='none', sep_country=False, output_file = "networkviewer/static/company_network.html"):
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
        if chemical_group == 'none':
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
            inchikey_counts = {}
            for inch in total_inchikeys:
                inchikey_counts[inch] = inchikey_counts.get(inch, 0) + 1
            for inchikey in total_inchikeys:
                for node in net.nodes:
                    if inchikey in node.get('title', ''):
                        net.add_edge(company_name, node['id'], width = inchikey_counts[inchikey], title=f"Appearances: {inchikey_counts[inchikey]}")
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
            inchikey_counts = {}
            for inch in added_inchikeys:
                inchikey_counts[inch] = inchikey_counts.get(inch, 0) + 1
            for inchikey in added_inchikeys:
                for node in net.nodes:
                    node_title = node.get('title', '')
                    if node_title and isinstance(node_title, str) and inchikey in node_title:
                        net.add_edge(company_name, node['id'], width = inchikey_counts[inchikey], title=f"Appearances: {inchikey_counts[inchikey]}")
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
            aff_counts = {}
            for affil in total_affil:
                aff_counts[affil] = aff_counts.get(affil,0) + 1
            for affil in total_affil:
                for node in net.nodes:
                    node_title = node.get('title','')
                    if node_title and isinstance(node_title, str) and affil in node_title:
                        net.add_edge(company_name,node['id'], width = aff_counts[affil], title=f"Appearances: {aff_counts[affil]}")
        elif sep_country == True:
            total_affil=[]
            aff_counts={}
            country_affil_counts = {}

            for affil, country in zip(data['Affiliations'], data['Countries']):
                total_affil.append(affil)
                aff_counts[affil] = aff_counts.get(affil,0)+1
                country_affil_counts[country] = country_affil_counts.get(country, 0) + 1

            for country in country_affil_counts:
                net.add_node(
                    country, 
                    label=country,
                    color='lightgreen', 
                    shape='box',
                    size=20
                )
                net.add_edge(
                    company_name, 
                    country, 
                    width=country_affil_counts[country], 
                    title=f"Affiliations: {country_affil_counts[country]}"
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
                
                net.add_edge(country, affil,width=aff_counts[affil], title=f"Appearances: {aff_counts[affil]}")
        else:
            print('Invalid')
    elif category == 'Universities':
        total_uni = []
        for uni in data:
            if uni not in total_uni:
                net.add_node(uni, label=uni, title=uni, color="lightblue", shape="ellipse",size=15)
                total_uni.append(uni)
            else:
                total_uni.append(uni)
        uni_counts = {}
        for uni in total_uni:
            uni_counts[uni] = uni_counts.get(uni, 0) + 1
        for uni in total_uni:
            for node in net.nodes:
                node_title = node.get('title','')
                if node_title and isinstance(node_title, str) and uni in node_title:
                    net.add_edge(company_name,node['id'], width = uni_counts[uni], title=f"Appearances: {uni_counts[uni]}")
    elif category == 'Researchers':
        total_res = []
        for res, aff in zip(res_list, aff_list):
            if (res + '|' + aff[:20]) not in total_res:
                net.add_node(res,label=res,title=aff, color='lightblue',shape='ellipse',size = 15)
                total_res.append(res+'|'+aff[:20])
            else:
                total_res.append(res+'|'+aff[:20])
        res_counts = {}
        for res in total_res:
            res_counts[res] = res_counts.get(res,0) + 1
        for res in total_res:
            for node in net.nodes:
                node_title = node.get('label','')+'|'+node.get('title','')
                if node_title and isinstance(node_title,str) and res in node_title:
                    net.add_edge(company_name,node['id'], width = res_counts[res],title=f"Appearances: {res_counts[res]}")
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
    # Show graph in browser
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
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
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

def show_uni_network_pyvis(uni_name, category='Companies', chemical_group='none', output_file = "networkviewer/static/company_network.html"):    # Filter for the selected company
    # Filter for the selected company
    row = comparing_unis[comparing_unis['University'] == uni_name]
    if row.empty:
        print(f"University '{uni_name}' not found.")
        return False
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
        if chemical_group == 'none':
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
            inchikey_counts = {}
            for inch in total_inchikeys:
                inchikey_counts[inch] = inchikey_counts.get(inch, 0) + 1
            for inchikey in total_inchikeys:
                for node in net.nodes:
                    if inchikey in node.get('title', ''):
                        net.add_edge(uni_name, node['id'], width = inchikey_counts[inchikey], title=f"Appearances: {inchikey_counts[inchikey]}")
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
            inchikey_counts = {}
            for inch in added_inchikeys:
                inchikey_counts[inch] = inchikey_counts.get(inch, 0) + 1
            for inchikey in added_inchikeys:
                for node in net.nodes:
                    node_title = node.get('title', '')
                    if node_title and isinstance(node_title, str) and inchikey in node_title:
                        net.add_edge(uni_name, node['id'], width = inchikey_counts[inchikey], title=f"Appearances: {inchikey_counts[inchikey]}")
    if category == 'Companies':
        total_comp = []
        for affil in data:
            if affil not in total_comp:
                net.add_node(affil, label=affil, title=affil, color="lightblue", shape="ellipse",size=15)
                total_comp.append(affil)
            else:
                total_comp.append(affil)
        comp_counts = {}
        for comp in total_comp:
            comp_counts[comp] = comp_counts.get(comp, 0) + 1
        for comp in total_comp:
            for node in net.nodes:
                node_title = node.get('title','')
                if node_title and isinstance(node_title, str) and comp in node_title:
                    net.add_edge(uni_name,node['id'], width = comp_counts[comp], title=f"Appearances: {comp_counts[comp]}")
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
    company_study_map = {}
    if category =='Companies':
        for comp in data:
            studies = main[
                (main['Funding Sources'].str.contains(comp, na=False, regex=False)) &
                (main['Affiliations'].str.contains(uni_name, na=False))
            ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection."
            company_study_map[comp] = study_info
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
    # Show graph in browser
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    injection = f"""
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
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
    studies_by_company = row['StudiesByCompany']
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
    comp_counts = {}
    for comp in total_comp:
        comp_counts[comp] = comp_counts.get(comp, 0) + 1
    for comp in total_comp:
        for node in net.nodes:
            node_title = node.get('label','')
            if node_title and isinstance(node_title, str) and comp in node_title:
                net.add_edge(researcher,node['id'], width = comp_counts[comp], title=f"Appearances: {comp_counts[comp]}")
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

def show_chemical_network(chemical, inch = 'Error', output_file = "networkviewer/static/company_network.html"):    # Filter for the selected company
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
            net.add_node(comp, label=comp, title=comp, color="lightblue", shape="ellipse",size=15)
            total_comp.append(comp)
        else:
            total_comp.append(comp)
    comp_counts = {}
    for comp in total_comp:
        comp_counts[comp] = comp_counts.get(comp, 0) + 1
    for comp in total_comp:
        for node in net.nodes:
            node_title = node.get('label','')
            if node_title and isinstance(node_title, str) and comp in node_title:
                net.add_edge(chemical,node['id'], width = comp_counts[comp], title=f"Appearances: {comp_counts[comp]}",color='red')
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
    # Show graph in browser
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    company_study_map = {}
    for comp in data:
        if inch and inch != 'Error' and inch != 'Not Found':
            studies = main[
                (main['Funding Sources'].str.contains(comp, na=False, regex=False)) &
                (main['Chemicals with InChIKey'].str.contains(inch, na=False))
            ]
        else:
            # fallback, but this should rarely happen
            studies = main[
                (main['Funding Sources'].str.contains(comp, na=False)) &
                (main['Chemicals with InChIKey'].str.contains(chemical, na=False))
            ]
        study_info = "<br>".join(
            f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
        ) or "No studies found for this connection."
        company_study_map[comp] = study_info
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    injection = f"""
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
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

def show_researcher_network_pyvis_from_row(row, output_file="networkviewer/static/company_network.html"):
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
            net.add_node(comp, label=comp, title=comp, color="lightblue", shape="ellipse", size=15)
            total_comp.append(comp)
        else:
            total_comp.append(comp)
    comp_counts = {}
    for comp in total_comp:
        comp_counts[comp] = comp_counts.get(comp, 0) + 1
    for comp in total_comp:
        for node in net.nodes:
            node_title = node.get('label', '')
            if node_title and isinstance(node_title, str) and comp in node_title:
                net.add_edge(researcher, node['id'], width=comp_counts[comp], title=f"Appearances: {comp_counts[comp]}")
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
    company_study_map = {}
    for comp in data:
        studies = main[
            (main['Funding Sources'].str.contains(comp, na=False, regex=False)) &
            (main['Authors'].str.contains(researcher, na=False, regex=False))
        ]
        study_info = "<br>".join(
            f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.drop_duplicates(subset=['DOI']).iterrows()
        ) or "No studies found for this connection."
        company_study_map[comp] = study_info
    net.show(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()

    injection = f"""
    <div id="study-info" style="margin-top:20px; background:#fff; color:#222; padding:10px; border-radius:8px;"></div>
    <script type="text/javascript">
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

    # === Chemicals ===
    chem_counts = {}
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            key = f"{inchikey}||{name}"
        else:
            key = name
        chem_counts[key] = chem_counts.get(key, 0) + 1

    labeled_chemicals = []
    for key, count in chem_counts.items():
        if '||' in key:
            _, name = key.split('||', 1)
        else:
            name = key
        labeled_chemicals.append(f"{name} ({count})")

    # === Countries ===
    country_counts = {}
    for country in countries:
        country_counts[country] = country_counts.get(country, 0) + 1
    total_countries = [f"{country} ({count})" for country, count in country_counts.items()]

    # === Affiliations ===
    aff_counts = {}
    for aff in affiliations:
        aff_counts[aff] = aff_counts.get(aff, 0) + 1
    total_aff = [f"{aff} ({count})" for aff, count in aff_counts.items()]

    # === Researchers ===
    res_counts = {}
    for res in res_list:
        res_counts[res] = res_counts.get(res, 0) + 1
    total_res = [f"{res} ({count})" for res, count in res_counts.items()]

    # === Universities ===
    uni_counts = {}
    for uni in universities:
        uni_counts[uni] = uni_counts.get(uni, 0) + 1
    total_uni = [f"{uni} ({count})" for uni, count in uni_counts.items()]

    return {
        "Affiliations": total_aff,
        "Countries": total_countries,
        "Researchers": total_res,
        "Universities": total_uni,
        "Chemicals": labeled_chemicals
    }

def show_uni_connections(university):
    row = comparing_unis[comparing_unis['University'] == university]
    if row.empty:
        print(f"University '{university}' not found.")
        return False

    parsed_chems = list(parse_chemical_entry(c) for c in row.iloc[0]['Chemicals'])
    companies = row.iloc[0]['Companies']

    # === Chemicals ===
    chem_counts = {}
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            key = f"{inchikey}||{name}"
        else:
            key = name
        chem_counts[key] = chem_counts.get(key, 0) + 1

    labeled_chemicals = []
    for key, count in chem_counts.items():
        if '||' in key:
            _, name = key.split('||', 1)
        else:
            name = key
        labeled_chemicals.append(f"{name} ({count})")

    # === Companies ===
    company_counts = {}
    for comp in companies:
        company_counts[comp] = company_counts.get(comp, 0) + 1
    total_companies = [f"{comp} ({count})" for comp, count in company_counts.items()]

    return {
        "Companies": total_companies,
        "Chemicals": labeled_chemicals
    }

def show_res_connections(researcher):
    matches = comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]
    if matches.empty:
        print(f"Researcher: '{researcher}' not found.")
        return
    if len(matches) > 1:
        # Combine all companies and pick the longest affiliation
        all_companies = sum(matches['Companies'], [])
        unique_affiliations = matches['Affiliation'].dropna().unique()
        combined_aff = '; '.join(unique_affiliations)
        row = {
            'Researcher': researcher,
            'Affiliation': combined_aff,
            'Companies': all_companies
        }

    if len(matches) == 1:
        row = matches.iloc[0]

    data = row['Companies']
    aff = row['Affiliation']
    if aff == '':
        aff = 'Not Found'

    # === Companies ===
    company_counts = {}
    for comp in data:
        company_counts[comp] = company_counts.get(comp, 0) + 1
    total_companies = [f"{comp} ({count})" for comp, count in company_counts.items()]

    return {
        "Affiliation(s)": aff,
        "Companies": total_companies
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

    # === Companies ===
    company_counts = {}
    for comp in data:
        company_counts[comp] = company_counts.get(comp, 0) + 1
    total_companies = [f"{comp} ({count})" for comp, count in company_counts.items()]

    return {
        "Inchikey": inchikey_val,
        "Companies": total_companies
    }