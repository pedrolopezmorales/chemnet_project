import pandas as pd
from pyvis.network import Network
import re
from country_list import countries_for_language
import pubchempy as pcp
import os
import json
from django.conf import settings
import ast
from collections import Counter
import requests
import time
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Directory for locally cached copies of the remote datasets. These act as a
# fallback so the app can still start (with slightly stale data) if the remote
# host is temporarily unavailable, instead of failing to import entirely.
_DATA_CACHE_DIR = os.path.join(settings.BASE_DIR, 'data', 'cache')
_PUBCHEM_CACHE_DIR = os.path.join(_DATA_CACHE_DIR, 'pubchem')
_PUBCHEM_DESC_CACHE_FILE = os.path.join(_PUBCHEM_CACHE_DIR, 'descriptions.json')
_PUBCHEM_IMAGE_CACHE_FILE = os.path.join(_PUBCHEM_CACHE_DIR, 'image_urls.json')
_FUNDING_DESC_CACHE_FILE = os.path.join(_DATA_CACHE_DIR, 'funding_source_descriptions.json')

# Absolute directory where generated network graphs are written. Anchoring to
# STATIC_ROOT (not a relative "staticfiles/" path) ensures the files land in the
# exact location Django serves /static/ from, regardless of the current working
# directory the server was launched from.
_GRAPH_DIR = os.path.join(settings.BASE_DIR, 'staticfiles')

_pubchem_desc_cache = None
_pubchem_image_cache = None
_funding_desc_cache = None


def _load_json_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to load cache file %s: %s", cache_file, exc)
    return {}


def _save_json_cache(cache_file, data):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write cache file %s: %s", cache_file, exc)


def _pubchem_cache_key(chemical_name=None, inchikey=None):
    inch = str(inchikey or '').strip()
    if inch and inch != 'Error':
        return f"inchikey:{inch.upper()}"
    chem = str(chemical_name or '').strip().lower()
    return f"name:{chem}"


def _get_pubchem_image_cache():
    global _pubchem_image_cache
    if _pubchem_image_cache is None:
        _pubchem_image_cache = _load_json_cache(_PUBCHEM_IMAGE_CACHE_FILE)
    return _pubchem_image_cache


def _get_pubchem_desc_cache():
    global _pubchem_desc_cache
    if _pubchem_desc_cache is None:
        _pubchem_desc_cache = _load_json_cache(_PUBCHEM_DESC_CACHE_FILE)
    return _pubchem_desc_cache


def _funding_cache_key(funding_source):
    return re.sub(r'\s+', ' ', str(funding_source or '')).strip().lower()


def _get_funding_desc_cache():
    global _funding_desc_cache
    if _funding_desc_cache is None:
        _funding_desc_cache = _load_json_cache(_FUNDING_DESC_CACHE_FILE)
    return _funding_desc_cache


def load_remote_csv(url, cache_name, **read_csv_kwargs):
    """Load a CSV from a remote URL, with a local cache as a fallback.

    Tries the remote URL first. On success, refreshes the on-disk cache so a
    good copy is always available. If the download fails (network error, host
    down, timeout), falls back to the cached copy when one exists. Raises only
    if the remote fetch fails AND no cache is available.
    """
    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_DATA_CACHE_DIR, cache_name)
    try:
        df = pd.read_csv(url, **read_csv_kwargs)
        try:
            df.to_csv(cache_path, index=False)
        except Exception as exc:  # caching is best-effort, never fatal
            logger.warning("Could not write cache file %s: %s", cache_path, exc)
        return df
    except Exception as exc:
        logger.warning(
            "Failed to fetch %s (%s). Falling back to local cache.", url, exc
        )
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, **read_csv_kwargs)
        raise RuntimeError(
            f"Could not load '{cache_name}' from {url} and no cached copy "
            f"exists at {cache_path}."
        ) from exc



MAIN_CSV_URL = "https://ucsf.box.com/shared/static/9c95305i26voc83jikp8fv6kt5zardth.csv"
main = load_remote_csv(MAIN_CSV_URL, 'main.csv')
main = main.dropna(subset=['Authors'])

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

_GOVERNMENT_PATTERNS = [
    r'\bdepartment\s+of\b',
    r'\bgovernment\b',
    r'\bministry\s+of\b',
    r'\bnational\s+institute(s)?\b',
    r'\bnational\s+center(s)?\b',
    r'\bnational\s+lab(oratory|oratories)\b',
    r'\bnational\s+science\s+foundation\b',
    r'\bscience\s+foundation\b',
    r'\bscience\s+and\s+technology\b',
    r'\bpublic\s+health\s+service\b',
    r'\b(research\s+)?council\b',
    r'\bagency\b',
    r'\bbureau\b',
    r'\boffice\b',
]

_FOUNDATION_PATTERNS = [
    r'\bfoundation\b',
    r'\btrust\b',
    r'\bendowment\b',
    r'\bcharity\b',
]

_COMPANY_PATTERNS = [
    r'\binc\.?\b',
    r'\bcorp\.?\b',
    r'\bcorporation\b',
    r'\bcompany\b',
    r'\bco\.?\b',
    r'\bllc\b',
    r'\bltd\b',
    r'\bpharmaceutical\b',
    r'\bbiotech\b',
    r'\btechnolog(y|ies)\b',
]

_UNIVERSITY_PATTERNS = [
    r'\buniversity\b',
    r'\bcollege\b',
    r'\binstitute of\b',
    r'\binstitute\b',
    r'\binstituto\b',
    r'\buniversidad\b',
    r'\buniversita\b',
    r'\buniversit\b',
]


def normalize_entity_name(entity_name):
    if entity_name is None or pd.isna(entity_name):
        return ''
    return re.sub(r'\s+', ' ', str(entity_name)).strip().lower()


def matches_any_pattern(entity_lower, patterns):
    return any(re.search(pattern, entity_lower) for pattern in patterns)


def classify_funding_source_by_rules(entity_name):
    entity_lower = normalize_entity_name(entity_name)
    if not entity_lower:
        return 'Unknown'

    if matches_any_pattern(entity_lower, _GOVERNMENT_PATTERNS):
        return 'Government'
    if matches_any_pattern(entity_lower, _FOUNDATION_PATTERNS):
        return 'Foundation'
    if matches_any_pattern(entity_lower, _COMPANY_PATTERNS):
        return 'Company'
    if matches_any_pattern(entity_lower, _UNIVERSITY_PATTERNS):
        return 'University'

    return 'Unknown'

#function for categorizing funding sources

@lru_cache(maxsize=1000)
def categorize_funding_source(entity_name):
    if not entity_name or pd.isna(entity_name):
        return 'Unknown'
    
    result = check_government_databases(entity_name)
    if result != 'Unknown':
        return result

    result = classify_funding_source_by_rules(entity_name)
    if result != 'Unknown':
        return result

    result = check_opencorporates_api(entity_name)
    if result != 'Unknown':
        return result
    
    result = check_wikipedia_api(entity_name)
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
                entity_lower = normalize_entity_name(entity_name)

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
            combined_text = f"{title} {extract}"

            if any(term in combined_text for term in [
                'government agency', 'federal agency', 'department of', 
                'ministry of', 'government department', 'public agency',
                'federal government', 'united states government', 'government'
            ]):
                return 'Government'
            if any(term in combined_text for term in [
                'university', 'college', 'institute of technology',
                'academic institution', 'higher education', 'medical school'
            ]):
                return 'University'
            if any(term in combined_text for term in [
                'foundation', 'charitable foundation', 'non-profit',
                'nonprofit', 'charity', 'philanthropic', 'endowment'
            ]):
                return 'Foundation'
            if any(term in combined_text for term in [
                'corporation', 'company', 'inc.', 'pharmaceutical company',
                'biotechnology company', 'multinational corporation',
                'publicly traded', 'private company'
            ]):
                return 'Company'
        time.sleep(0.2)
    except Exception as e:
        print(f"Wikipedia API error for {entity_name}: {e}")
    return 'Unknown'
def obtain_inchikey_from_pubchem(chemical_name):
    if not chemical_name:
        return None
    try:
        compounds = pcp.get_compounds(chemical_name, 'name')
        if not compounds:
            return None
        inchikey = getattr(compounds[0], 'inchikey', None)
        if inchikey and str(inchikey).strip():
            return str(inchikey).strip()
    except Exception as e:
        return None
        
def check_government_databases(entity_name):
    entity_lower = normalize_entity_name(entity_name)

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
        r'\bnational\s+(institute|institutes|center|centers|laboratory|laboratories)\b',
        r'\bministry\s+of\b',
        r'\bscience\s+foundation\b',
        r'\bresearch\s+council\b',
        r'\bpublic\s+health\s+service\b',
    ]

    if any(re.search(pattern, entity_lower) for pattern in government_patterns):
            return 'Government'
    
    return 'Unknown'

def categorize_funding_source_keywords(entity_name):
    return classify_funding_source_by_rules(entity_name)
#graphing funding source function to get category color
def get_category_color(category):
    color_map = {
        'Government': '#DD403A',      # Cinnabar
        'University': '#7B4B94',      # Deep Lilac
        'Foundation': '#B7E3CC',      # Celadon
        'Company': '#7D82B8',         # Blue Gray
        'Unknown': '#FFC145'          # Gold
      }
    return color_map.get(category, "#DDD6FE")


def get_category_display_name(category):
    if category == 'Unknown':
        return 'Not Recognized'
    return category

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
    """
    Classify companies using the pre-built classification dictionary for faster performance.
    Falls back to the categorize_funding_source function for any companies not in the dictionary.
    """
    classified_companies = []
    total = len(companies_list)
    cache_hits = 0
    api_calls = 0
    
    for i, company in enumerate(companies_list):
        if company and not pd.isna(company):
            company_clean = company.strip()
            # Try to get classification from our pre-built dictionary first
            if company_clean in company_classification_dict:
                category = company_classification_dict[company_clean]
                cache_hits += 1
            else:
                # Fallback to API call if not in dictionary
                category = categorize_funding_source(company_clean)
                company_classification_dict[company_clean] = category  # Cache for future use
                api_calls += 1
            
            classified_company = f"{company_clean} [{category}]"
            classified_companies.append(classified_company)
        else:
            classified_companies.append(company)
        
        if (i + 1) % 100 == 0 or (i + 1) == total:
            percentage = ((i + 1) / total) * 100
            print(f"Classified {i + 1} out of {total} companies ({percentage:.2f}%) - Cache hits: {cache_hits}, API calls: {api_calls}")
    
    print(f"Classification complete! Cache hits: {cache_hits}, API calls: {api_calls}")
    return classified_companies

# Modifying Database by removing certain columns

comparing_companies = main.drop(['DOI', 'URL','Year','Title','Chemicals Mentioned','Abstract'], axis = 1)

# Creating new dataframe that had a list of companies, a list of chemicals, and a list of affiliations per row

companies = comparing_companies['Funding Sources'].str.split(r'[;]').explode().str.strip().tolist()
no_dup_comp = list(set(companies))
'''
# Create company classification dictionary
print("Creating company classification dictionary...")
company_classification_dict = {}

# Check if we have a saved classification dictionary
classification_file_path = os.path.join(settings.BASE_DIR, 'data', 'company_classifications.json')

if os.path.exists(classification_file_path):
    print("Loading existing company classifications...")
    import json
    try:
        with open(classification_file_path, 'r', encoding='utf-8') as f:
            company_classification_dict = json.load(f)
        print(f"Loaded {len(company_classification_dict)} existing classifications")
    except Exception as e:
        print(f"Error loading classifications: {e}")
        company_classification_dict = {}

# Classify any new companies not in the saved dictionary
new_companies = [comp for comp in no_dup_comp if comp not in company_classification_dict]
if new_companies:
    print(f"Classifying {len(new_companies)} new companies...")
    for i, company in enumerate(new_companies):
        if company and not pd.isna(company):
            category = categorize_funding_source(company.strip())
            company_classification_dict[company.strip()] = category
        else:
            company_classification_dict[company] = 'Unknown'
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(new_companies):
            percentage = ((i + 1) / len(new_companies)) * 100
            print(f"Classified {i + 1} out of {len(new_companies)} new companies ({percentage:.2f}%)")
    
    # Save the updated dictionary
    try:
        import json
        with open(classification_file_path, 'w', encoding='utf-8') as f:
            json.dump(company_classification_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved updated classifications to {classification_file_path}")
    except Exception as e:
        print(f"Error saving classifications: {e}")
else:
    print("All companies already classified!")

print(f"Company classification dictionary created with {len(company_classification_dict)} entries!")
print(f"Sample classifications: {dict(list(company_classification_dict.items())[:5])}")

# Helper functions for company classification
def get_company_classification(company_name):
    """Get the classification for a company from the pre-built dictionary."""
    return company_classification_dict.get(company_name.strip() if company_name else '', 'Unknown')

def get_classification_stats():
    """Get statistics about company classifications."""
    from collections import Counter
    stats = Counter(company_classification_dict.values())
    return dict(stats)

def get_companies_by_category(category):
    """Get all companies belonging to a specific category."""
    return [company for company, cat in company_classification_dict.items() if cat == category]

# Print classification statistics
classification_stats = get_classification_stats()
print("\nCompany Classification Statistics:")
for category, count in sorted(classification_stats.items()):
    print(f"  {category}: {count} companies")

'''
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

def create_researcher_affiliation_pairs_components(row):
    researchers = row['Names']
    affiliations = row['Aff']

    if len(researchers) > len(affiliations):
        affiliations = affiliations + [''] * (len(researchers) - len(affiliations))
    elif len(researchers) < len(affiliations):
        affiliations = affiliations[:len(researchers)]
    
    return list(zip(researchers, affiliations))



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
def normalize_author_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[\u2010-\u2015-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def author_match_mask(authors_series, target_name):
    normalized_authors = authors_series.apply(normalize_author_text)
    normalized_target = normalize_author_text(target_name)
    return normalized_authors.str.contains(normalized_target, na=False, regex=False)
def normalize_funding_source_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
def funding_source_match_mask(funding_sources_series, target_name):
    normalized_target = normalize_funding_source_text(target_name)
    def matches(cell_value):
        if pd.isna(cell_value):
            return False

        parts = [
            normalize_funding_source_text(part)
            for part in str(cell_value).split(";")
            if part.strip()
        ]
        return normalized_target in parts

    return funding_sources_series.apply(matches)


def parse_collaborator_entry(entry):
    """Return collaborator (name, affiliation) from dict/tuple/string input."""
    if isinstance(entry, dict):
        name = str(entry.get('Researcher', '')).strip()
        affiliation = str(entry.get('Affiliation', '')).strip()
        return name, affiliation

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        name = str(entry[0]).strip()
        affiliation = str(entry[1]).strip()
        return name, affiliation

    if isinstance(entry, str):
        return entry.strip(), ''

    return str(entry).strip(), ''


def parse_collaborators_cell(value):
    """Normalize collaborator cell to a list of {'Researcher', 'Affiliation'} dicts."""
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('['):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = []
        elif not text:
            parsed = []
        else:
            parsed = [text]

    if parsed is None or (isinstance(parsed, float) and pd.isna(parsed)):
        return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    normalized = []
    for collaborator in parsed:
        name, affiliation = parse_collaborator_entry(collaborator)
        if not name:
            continue
        normalized.append({
            'Researcher': name,
            'Affiliation': affiliation,
        })
    return normalized
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

FUNDING_SOURCE_CSV_URL =  "https://ucsf.box.com/shared/static/rw5cidunf09u2ftfoxrzc89is94wcb4f.csv"
company_assoc = load_remote_csv(FUNDING_SOURCE_CSV_URL, 'company_assoc.csv')

def _safe_parse_list_cell(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value is None:
        return []
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    def _split_numpy_style_list_string(raw_text):
        raw_text = raw_text.strip()
        if not (raw_text.startswith('[') and raw_text.endswith(']')):
            return None
        inner = raw_text[1:-1]
        # Matches values quoted as 'value' or "value" in strings like ['a' 'b' 'c']
        matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
        if not matches:
            return None
        items = [(m1 or m2).strip() for m1, m2 in matches if (m1 or m2).strip()]
        return items if len(items) > 1 else None

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            # Handle NumPy-style serialization without commas: ['a' 'b' 'c']
            # ast.literal_eval may fold this into a single concatenated string.
            numpy_style_items = _split_numpy_style_list_string(text)
            if numpy_style_items:
                return numpy_style_items
            return parsed
        return [str(parsed)]
    except Exception:
        numpy_style_items = _split_numpy_style_list_string(text)
        if numpy_style_items:
            return numpy_style_items

        cleaned = text.strip('[]').strip()
        cleaned = cleaned.strip('"').strip("'")
        if not cleaned:
            return []

        for delimiter in [';', '|']:
            if delimiter in cleaned:
                return [item.strip().strip('"').strip("'") for item in cleaned.split(delimiter) if item.strip()]

        paren_groups = re.findall(r'[^()]+\([^()]*\)', cleaned)
        if len(paren_groups) > 1:
            return [item.strip() for item in paren_groups if item.strip()]

        return [cleaned]

company_assoc['Affiliations'] = company_assoc['Affiliations'].apply(_safe_parse_list_cell)
company_assoc['Chemicals'] = company_assoc['Chemicals'].apply(_safe_parse_list_cell)
company_assoc['Researchers'] = company_assoc['Researchers'].apply(_safe_parse_list_cell)
company_assoc['Affs'] = company_assoc['Affs'].apply(_safe_parse_list_cell)
company_assoc['Universities'] = company_assoc['Universities'].apply(_safe_parse_list_cell)
company_assoc['Countries'] = company_assoc['Countries'].apply(_safe_parse_list_cell)

company_assoc['Company'] = company_assoc['Company'].fillna('').astype(str).str.strip()
company_assoc = company_assoc[company_assoc['Company'] != ''].reset_index(drop=True)


def inject_node_slider(html, center_node):
    """Inject a node count slider into PyVis-generated HTML.

    Adds a range slider between the control bar and the study-info panel.
    Nodes are sorted by their edge weight to the center node; sliding left
    hides the weakest-connected nodes first.
    """
    slider_html = '''    <div style="padding:8px 16px;background:#f8f9fa;border-radius:8px;margin:4px 0;">
        <label style="font-size:13px;color:#555;font-weight:500;display:block;margin-bottom:4px;">
            Showing <strong id="slider-label">0 / 0</strong> nodes
            <span style="font-weight:normal;"> (sorted by connections)</span>
            <div style="font-size:20px;color:#777;margin-top:4px;">Slide to show fewer nodes with more connections</div>
                    </label>
        <input type="range" id="node-slider" min="1" max="1" value="1" step="1"
               style="width:100%;cursor:pointer;accent-color:#007bff;display:block;">
    </div>
    '''
    slider_js = f'''<script type="text/javascript">
    (function() {{
        var _cid = {json.dumps(center_node)};
        var _an = nodes.get(), _ae = edges.get();
        var _w = {{}};
        _ae.forEach(function(e) {{
            var m = (e.title||"").match(/(\\d+)$/);
            var w = m ? parseInt(m[1]) : 0;
            if (e.from !== _cid) _w[e.from] = Math.max(_w[e.from]||0, w);
            if (e.to !== _cid) _w[e.to] = Math.max(_w[e.to]||0, w);
        }});
        var _ids = _an
            .filter(function(n) {{ return n.id !== _cid; }})
            .sort(function(a, b) {{ return (_w[b.id]||0) - (_w[a.id]||0); }})
            .map(function(n) {{ return n.id; }});
        var _tot = _ids.length;
        var _sl = document.getElementById("node-slider");
        _sl.max = _tot || 1;
        _sl.value = _tot;
        document.getElementById("slider-label").textContent = _tot + " / " + _tot;
        _sl.addEventListener("input", function() {{
            var cnt = parseInt(this.value);
            var vis = new Set(_ids.slice(0, cnt));
            nodes.update(
                _an.filter(function(n) {{ return n.id !== _cid; }})
                   .map(function(n) {{ return {{id: n.id, hidden: !vis.has(n.id)}}; }})
            );
            edges.update(
                _ae.map(function(e) {{
                    var o = (e.from === _cid) ? e.to : e.from;
                    return {{id: e.id, hidden: !vis.has(o)}};
                }})
            );
            document.getElementById("slider-label").textContent = cnt + " / " + _tot;
        }});
    }})();
    </script>
    '''
    if '<div id="study-info"' in html:
        html = html.replace('<div id="study-info"', slider_html + '    <div id="study-info"', 1)
    if "</body>" in html:
        html = html.replace("</body>", slider_js + "\n</body>", 1)
    else:
        html += slider_js
    return html


def get_company_funding_rows(company_name):
    return main[funding_source_match_mask(main['Funding Sources'], company_name)]


def get_university_rows(university):
    return main[main['Affiliations'].str.contains(university, na=False, regex=False)]


def get_researcher_matches(researcher):
    return comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]


def get_researcher_rows(researcher):
    return main[author_match_mask(main['Authors'], researcher)]


def get_chemical_row(chemical=None, inchikey=None):
    if inchikey:
        row = chem_per_row[chem_per_row['inchikey'] == inchikey]
    elif chemical:
        row = chem_per_row[chem_per_row['chemical'].apply(lambda x: any(chemical.lower() == name.lower() for name in x))]
    else:
        row = pd.DataFrame()
    return row


def _graph_output_exists(output_file):
    return bool(output_file) and os.path.isfile(output_file) and os.path.getsize(output_file) > 0


def _with_connection_filter_suffix(output_file, max_connection_count=0):
    if not max_connection_count:
        return output_file
    suffix = '_no_singletons' if max_connection_count == 1 else f'_le{max_connection_count}'
    if output_file.endswith('.html'):
        return output_file[:-5] + suffix + '.html'
    return output_file + suffix


def _apply_connection_filter(net, central_node_id, max_connection_count):
    filtered_edges = []
    connected_nodes = {central_node_id}

    for edge in net.edges:
        width = edge.get('width', 1)
        try:
            edge_weight = float(width)
        except (TypeError, ValueError):
            edge_weight = 1.0

        if edge_weight <= float(max_connection_count):
            continue

        filtered_edges.append(edge)
        connected_nodes.add(edge.get('from'))
        connected_nodes.add(edge.get('to'))

    net.edges = filtered_edges
    net.nodes = [
        node for node in net.nodes
        if node.get('id') == central_node_id or node.get('id') in connected_nodes
    ]


def show_company_network_pyvis(company_name, category='Affiliations', chemical_group='All', sep_country=False, output_file=None, company_funding_rows=None, max_connection_count=0):
    if output_file is None:
        # Generate unique filename based on ALL parameters
        safe_company = company_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_category = category.replace(' ', '_')
        
        if category == 'Chemicals':
            if chemical_group == 'All':
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_company}_{safe_category}_all.html")
            elif chemical_group == 'Organic':
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_company}_{safe_category}_organic.html")
        elif category == 'Affiliations':
            if sep_country:
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_company}_{safe_category}_by_country.html")
            else:
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_company}_{safe_category}_combined.html")
        else:
            # For Universities, Researchers, etc.
            output_file = os.path.join(_GRAPH_DIR, f"network_{safe_company}_{safe_category}.html")
    output_file = _with_connection_filter_suffix(output_file, max_connection_count=max_connection_count)
    if _graph_output_exists(output_file):
        return True
    # Filter for the selected company
    row = company_assoc[company_assoc['Company'] == company_name]
    if row.empty:
        print(f"Company '{company_name}' not found.")
        return False
    if company_funding_rows is None:
        company_funding_rows = get_company_funding_rows(company_name)
    if category != 'Affiliations':
        data = row.iloc[0][category]
    else:
        data = row.iloc[0]['Affs']
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
                            studies = company_funding_rows[
                                company_funding_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name for chemicals without InChIKey
                            chemical_name = node.get('label', '')
                            studies = company_funding_rows[
                                company_funding_rows['Chemicals with InChIKey'].str.contains(chemical_name, na=False, regex=False)
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
                            studies = company_funding_rows[
                                company_funding_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = company_funding_rows[
                                company_funding_rows['Chemicals with InChIKey'].str.contains(chemical_name, na=False, regex=False)
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
                        studies = company_funding_rows[
                            company_funding_rows['Affiliations'].str.contains(affiliation, na=False, regex=False)
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
                studies = company_funding_rows[
                    company_funding_rows['Affiliations'].str.contains(affil, na=False, regex=False)
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
                    studies = company_funding_rows[
                        company_funding_rows['Affiliations'].str.contains(university, na=False, regex=False)
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
                    studies = company_funding_rows[
                        company_funding_rows['Authors'].str.contains(researcher, na=False, regex=False)
                    ]
                    study_count = len(studies.drop_duplicates(subset=['DOI']))
                    
                    study_counts[node['id']] = study_count
                    net.add_edge(
                        company_name,
                        node['id'], 
                        width=max(1, study_count), 
                        title=f"Studies: {study_count}"
                    )
    if max_connection_count:
        _apply_connection_filter(net, company_name, max_connection_count)

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
                studies = company_funding_rows[
                    company_funding_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                ]
            else:
                # Fallback to chemical name search
                studies = company_funding_rows[
                    company_funding_rows['Chemicals with InChIKey'].str.contains(name, na=False, regex=False)
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
                studies = company_funding_rows[
                    company_funding_rows['Affiliations'].str.contains(affil_str, na=False, regex=False)
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[affil_str] = study_info


            # Optionally, map studies for country nodes (use country name as key)
            for country in countries:
                country_str = str(country)
                studies = company_funding_rows[
                    company_funding_rows['Affiliations'].str.contains(country_str, na=False, regex=False)
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[country_str] = study_info
        else:
            for affil in data:
                affil_str = str(affil)
                studies = company_funding_rows[
                    company_funding_rows['Affiliations'].str.contains(affil_str, na=False, regex=False)
                ]
                study_info = "<br>".join(
                    f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
                ) or "No studies found for this connection"
                company_study_map[affil_str] = study_info
    elif category =='Universities':
        for uni in data:
            studies = company_funding_rows[
                company_funding_rows['Affiliations'].str.contains(uni, na=False, regex=False)
            ]
            study_info = '<br>'.join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.iterrows()
            ) or "No studies found for this connection"
            company_study_map[uni] = study_info
    elif category == 'Researchers':
        res_list = row.iloc[0]['Researchers']
        for res in res_list:
            studies = company_funding_rows[
                author_match_mask(company_funding_rows['Authors'], res)
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
    html = inject_node_slider(html, company_name)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True


#CSV_PATH_unis = os.path.join(settings.BASE_DIR, 'data', 'comparing_unis.csv')
UNI_CSV_URL = 'https://ucsf.box.com/shared/static/wrb0245iuc7k8htn3bwxrhwtvyc6trtj.csv' 
comparing_unis = load_remote_csv(UNI_CSV_URL, 'comparing_unis.csv')
comparing_unis['Companies'] = comparing_unis['Companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
comparing_unis['Chemicals'] = comparing_unis['Chemicals'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
def show_uni_network_pyvis(uni_name, category='Funding Sources', chemical_group='All', output_file=None, uni_rows=None, max_connection_count=0):
    if output_file is None:
        # Generate unique filename based on ALL parameters
        safe_uni = uni_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_category = category.replace(' ', '_')
        
        if category == 'Chemicals':
            if chemical_group == 'All':
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_uni}_{safe_category}_all.html")
            elif chemical_group == 'Organic':
                output_file = os.path.join(_GRAPH_DIR, f"network_{safe_uni}_{safe_category}_organic.html")
        else:
            # For Companies, etc.
            output_file = os.path.join(_GRAPH_DIR, f"network_{safe_uni}_{safe_category}.html")    # Filter for the selected company
    output_file = _with_connection_filter_suffix(output_file, max_connection_count=max_connection_count)
    if _graph_output_exists(output_file):
        return True
    row = comparing_unis[comparing_unis['University'] == uni_name]
    if row.empty:
        print(f"University '{uni_name}' not found.")
        return False
    if uni_rows is None:
        uni_rows = get_university_rows(uni_name)
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
                            studies = uni_rows[
                                uni_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = uni_rows[
                                uni_rows['Chemicals with InChIKey'].str.contains(chemical_name, na=False, regex=False)
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
                            studies = uni_rows[
                                uni_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                            ]
                            study_count = len(studies.drop_duplicates(subset=['DOI']))
                        else:
                            # Fallback to chemical name
                            chemical_name = node.get('label', '')
                            studies = uni_rows[
                                uni_rows['Chemicals with InChIKey'].str.contains(chemical_name, na=False, regex=False)
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
                    title=f"{original_name}\n Category: {get_category_display_name(entity_category)}",
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
                    studies = uni_rows[
                        uni_rows['Funding Sources'].str.contains(company, na=False, regex=False)
                    ]
                    study_count = len(studies.drop_duplicates(subset=['DOI']))
                    
                    study_counts[node['id']] = study_count
                    net.add_edge(
                        uni_name,
                        node['id'], 
                        width=max(1, study_count), 
                        title=f"Studies: {study_count}"
                    )
    if max_connection_count:
        _apply_connection_filter(net, uni_name, max_connection_count)

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
            studies = uni_rows[
                uni_rows['Funding Sources'].str.contains(original_name, na=False, regex=False)
            ]
            study_info = "<br>".join(
                f"{row['Title']} (DOI: {row['DOI']})" for _, row in studies.drop_duplicates(subset=['DOI']).iterrows()
            ) or "No studies found for this connection."
            company_study_map[original_name] = study_info
    elif category == 'Chemicals':
        parsed_chems = [parse_chemical_entry(c) for c in data]
        for name, inchikey in parsed_chems:
            key = name  # node label is the chemical name
            if inchikey and inchikey != 'Not Found':
                # Search by InChIKey if available
                studies = uni_rows[
                    uni_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                ]
            else:
                # Fallback to chemical name search
                studies = uni_rows[
                    uni_rows['Chemicals with InChIKey'].str.contains(name, na=False, regex=False)
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
                    <div style="width: 16px; height: 16px; background: #DD403A; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7B4B94; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #B7E3CC; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7D82B8; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFC145; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Not Recognized</span>
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
    html = inject_node_slider(html, uni_name)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True



#CSV_PATH_researchers = os.path.join(settings.BASE_DIR, 'data', 'comparing_researchers.csv')
RESEARCHER_CSV_URL = 'https://ucsf.box.com/shared/static/jrks0l53xm4wk58xi5ykix0a9xjf8bfp.csv'
comparing_researchers = load_remote_csv(RESEARCHER_CSV_URL, 'comparing_researchers.csv')
comparing_researchers['Companies'] = comparing_researchers['Companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
if 'Collaborators' in comparing_researchers.columns:
    comparing_researchers['Collaborators'] = comparing_researchers['Collaborators'].apply(parse_collaborators_cell)
else:
    comparing_researchers['Collaborators'] = [[] for _ in range(len(comparing_researchers))]

def show_researcher_network_pyvis(researcher, output_file = os.path.join(_GRAPH_DIR, "company_network.html"), category='Funding Sources'):    # Filter for the selected company
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
                all_collaborators = sum(matches['Collaborators'], []) if 'Collaborators' in matches.columns else []
                unique_affiliations = matches['Affiliation'].dropna().unique()
                combined_aff = '; '.join(unique_affiliations)
                row = {
                    'Researcher': researcher,
                    'Affiliation': combined_aff,
                    'Companies': all_companies,
                    'Collaborators': all_collaborators,
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
    category_key = 'Collaborators' if str(category).strip().lower() == 'collaborators' else 'Funding Sources'
    data = row['Collaborators'] if category_key == 'Collaborators' else row['Companies']
    aff = row['Affiliation']
    if aff == '':
        aff = 'Not Found'
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add company node
    net.add_node(researcher, label=researcher, title=f"Affiliation: {aff}",color="red", shape="ellipse", size=55)

    # Add connection nodes and edges
    total_comp = []
    collaborator_affiliations = {}
    for affil in data:
        if category_key == 'Collaborators':
            collaborator_name, collaborator_aff = parse_collaborator_entry(affil)
            if not collaborator_name:
                continue
            if collaborator_name not in collaborator_affiliations:
                collaborator_affiliations[collaborator_name] = set()
            if collaborator_aff:
                collaborator_affiliations[collaborator_name].add(collaborator_aff)
            node_id = collaborator_name
            aff_text = '; '.join(sorted(collaborator_affiliations[collaborator_name])) if collaborator_affiliations[collaborator_name] else 'Not Found'
            node_title = f"Collaborator: {collaborator_name}\\nAffiliation: {aff_text}"
        else:
            node_id = affil
            node_title = affil

        if node_id not in total_comp:
            if category_key == 'Collaborators':
                net.add_node(node_id, label=node_id, title=node_title, color="#A78BFA", shape="ellipse", size=15)
            else:
                net.add_node(node_id, label=node_id, title=node_title, color="lightblue", shape="ellipse", size=15)
            total_comp.append(node_id)
    study_counts = {}
    # Pre-filter to this researcher's rows once instead of scanning all of
    # `main` for every company node.
    researcher_rows = main[
        main['Authors'].str.contains(researcher, na=False, regex=False)
    ]
    for node in net.nodes:
        if node['id'] != researcher:  # Skip the researcher node itself
            company = node.get('label', '')  # Company name is the label
            if company:
                if category_key == 'Collaborators':
                    studies = researcher_rows[
                        author_match_mask(researcher_rows['Authors'], company)
                    ]
                else:
                    studies = researcher_rows[
                        researcher_rows['Funding Sources'].str.contains(company, na=False, regex=False)
                    ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))

                study_counts[node['id']] = study_count
                net.add_edge(
                    researcher,
                    node['id'], 
                    width=max(1, study_count), 
                    title=f"Collaborations: {study_count}" if category_key == 'Collaborators' else f"Studies: {study_count}"
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



#CSV_PATH_chem = os.path.join(settings.BASE_DIR, 'data', 'chem_per_row.csv')
CHEM_CSV_URL = 'https://ucsf.box.com/shared/static/ixxni1j5q945008p76ru86jn19xx5ng8.csv'
chem_per_row = load_remote_csv(CHEM_CSV_URL, 'chem_per_row.csv')
chem_per_row['company'] = chem_per_row['company'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
chem_per_row['chemical'] = chem_per_row['chemical'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

def show_chemical_network(chemical, inch='Error', output_file=None, row=None, max_connection_count=0):
    if output_file is None:
        safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        if inch != 'Error':
            safe_inch = inch.replace('/', '_').replace('\\', '_').replace('-', '_')
            output_file = os.path.join(_GRAPH_DIR, f"network_{safe_chemical}_{safe_inch}.html")
        else:
            output_file = os.path.join(_GRAPH_DIR, f"network_{safe_chemical}_no_inchikey.html")
    output_file = _with_connection_filter_suffix(output_file, max_connection_count=max_connection_count)
    if _graph_output_exists(output_file):
        return True
    # Filter for the selected company
    if row is None:
        row = get_chemical_row(chemical=chemical, inchikey=inch if inch != 'Error' else None)
    if inch == 'Error':
        if row.empty:
            print(f"Chemical '{chemical}' not found.")
            return False
        inchikey = row.iloc[0]['inchikey']
        if inchikey and inchikey != 'Error':
            inch = inchikey
            chemical = row.iloc[0]['chemical'][0]
    else:
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
                    title=f"{original_name}\n Category: {get_category_display_name(entity_category)}",
                    color=entity_color,
                    shape="ellipse",
                    size=15
                )
            total_comp.append(comp)
        else:
            total_comp.append(comp)
    study_counts = {}
    # Pre-filter to this chemical's rows once, then reuse for both the edge
    # counts here and the study tooltips below, instead of scanning all of
    # `main` for every company node.
    chem_key = inch if (inch and inch != 'Error' and inch != 'Not Found') else chemical
    chem_rows = main[
        main['Chemicals with InChIKey'].str.contains(chem_key, na=False, regex=False)
    ]
    for node in net.nodes:
        if node['id'] != chemical:  # Skip the chemical node itself
            company = node.get('id')  # Company name is the id
            if company:
                studies = chem_rows[
                    chem_rows['Funding Sources'].str.contains(company, na=False, regex=False)
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
    if max_connection_count:
        _apply_connection_filter(net, chemical, max_connection_count)

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
    # Reuse the chem_rows subset computed above; skip duplicate funding sources
    # since their study list would be identical.
    for comp in data:
        original_name, entity_category = extract_name_and_class(comp)
        if original_name in company_study_map:
            continue
        studies = chem_rows[
            chem_rows['Funding Sources'].str.contains(original_name, na=False, regex=False)
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
                    <div style="width: 16px; height: 16px; background: #DD403A; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7B4B94; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #B7E3CC; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7D82B8; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFC145; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Not Recognized</span>
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
    html = inject_node_slider(html, chemical)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True

def show_researcher_network_pyvis_from_row(row, output_file=None, researcher_rows=None, category='Funding Sources'):
    if output_file is None:
        researcher = row['Researcher']
        safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
        # Use first 20 chars of affiliation to make filename more unique
        safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_category = str(category).strip().lower().replace(' ', '_')
        output_file = os.path.join(_GRAPH_DIR, f"network_{safe_researcher}_{safe_aff}_{safe_category}.html")
    if _graph_output_exists(output_file):
        return True
    category_key = str(category).strip().lower()
    if category_key == 'collaborators':
        data = row.get('Collaborators', [])
    else:
        data = row['Companies']
    aff = row['Affiliation']
    researcher = row['Researcher']
    if aff == '':
        aff = 'Not Found'
    if researcher_rows is None:
        researcher_rows = get_researcher_rows(researcher)
    # Initialize PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True)
    net.barnes_hut()  # for better layout dynamics

    # Add researcher node
    net.add_node(researcher, label=researcher, title=f"Affiliation: {aff}", color="red", shape="ellipse", size=55)

    # Add connection nodes and edges
    total_comp = []
    collaborator_affiliations = {}
    for comp in data:
        if category_key == 'collaborators':
            original_name, collaborator_aff = parse_collaborator_entry(comp)
            if not original_name:
                continue
            if original_name not in collaborator_affiliations:
                collaborator_affiliations[original_name] = set()
            if collaborator_aff:
                collaborator_affiliations[original_name].add(collaborator_aff)
            aff_text = '; '.join(sorted(collaborator_affiliations[original_name])) if collaborator_affiliations[original_name] else 'Not Found'
            node_key = original_name
            entity_color = '#A78BFA'
            node_title = f"Collaborator: {original_name}\nAffiliation: {aff_text}"
        else:
            original_name, entity_type = extract_name_and_class(comp)
            node_key = comp
            entity_color = get_category_color(entity_type)
            node_title = f"{original_name}\nCategory: {get_category_display_name(entity_type)}"

        if node_key not in total_comp:
            net.add_node(
            original_name,
            label=original_name,
            title=node_title,
            color=entity_color,
            shape="ellipse",
            size=15
            )
            total_comp.append(node_key)
    study_counts = {}
    for node in net.nodes:
        if node['id'] != researcher:  # Skip the researcher node itself
            company = node.get('id')  # Company name is the label
            if company:
                if category_key == 'collaborators':
                    studies = researcher_rows[
                        author_match_mask(researcher_rows['Authors'], company)
                    ]
                else:
                    # Count studies mentioning this researcher with this company
                    studies = researcher_rows[
                        researcher_rows['Funding Sources'].str.contains(company, na=False, regex=False)
                    ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                
                study_counts[node['id']] = study_count
                net.add_edge(
                    researcher,
                    node['id'], 
                    width=max(1, study_count), 
                    title=f"Collaborations: {study_count}" if category_key == 'collaborators' else f"Studies: {study_count}"
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
    # Pre-filter once with the same predicate the original loop used
    # (substring match on Authors), instead of scanning all of `main` per company.
    researcher_main = main[
        main['Authors'].str.contains(researcher, na=False, regex=False)
    ]
    for comp in data:
        if category_key == 'collaborators':
            original_name, _ = parse_collaborator_entry(comp)
            if not original_name:
                continue
        else:
            original_name, _ = extract_name_and_class(comp)
        if category_key == 'collaborators':
            studies = researcher_main[
                author_match_mask(researcher_main['Authors'], original_name)
            ]
        else:
            studies = researcher_main[
                researcher_main['Funding Sources'].str.contains(original_name, na=False, regex=False)
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
                    <div style="width: 16px; height: 16px; background: #DD403A; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Government</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7B4B94; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">University</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #B7E3CC; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Foundation</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #7D82B8; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Company</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 16px; height: 16px; background: #FFC145; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 13px; color: #333;">Not Recognized</span>
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
    html = inject_node_slider(html, researcher)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    return True
def count_key(number):
    num = re.match(r'^.+\((\d+)\)$', number.strip())
    return int(num.group(1)) if num else 0
def show_company_connections(company_name, company_funding_rows=None):
    row = company_assoc[company_assoc['Company'] == company_name]
    if row.empty:
        print(f"Company '{company_name}' not found.")
        return False
    if company_funding_rows is None:
        company_funding_rows = get_company_funding_rows(company_name)
    affiliations = row.iloc[0]['Affs']
    countries = row.iloc[0]['Countries']
    parsed_chems = list(parse_chemical_entry(c) for c in row.iloc[0]['Chemicals'])
    res_list = row.iloc[0]['Researchers']
    universities = row.iloc[0]['Universities']

    # Chemicals
    labeled_chemicals = []
    processed_inchikeys = set()
    unique_no_inch_chemicals = []
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            if inchikey not in processed_inchikeys:
                # Chemicals with InChIKey
                studies = company_funding_rows[
                    (company_funding_rows['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                processed_inchikeys.add(inchikey)
        else:
            # Chemicals without InChIKey
            if name not in unique_no_inch_chemicals:
                studies = company_funding_rows[
                    (company_funding_rows['Chemicals with InChIKey'].str.contains(name, na=False,regex=False))
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                unique_no_inch_chemicals.append(name)
    labeled_chemicals = sorted(labeled_chemicals, key=count_key, reverse=True)

    # Countries
    unique_countries = []
    labeled_countries = []
    country_affil_counts = {}
    
    for country in countries:
        country_affil_counts[country] = country_affil_counts.get(country, 0) + 1

    for country in country_affil_counts:
        if country not in unique_countries:    
            affiliation_count = country_affil_counts[country]
            labeled_countries.append(f"{country} ({affiliation_count})")
            unique_countries.append(country)
    labeled_countries = sorted(labeled_countries, key=count_key, reverse=True)

    # Affiliations 
    unique_affiliations = []
    labeled_affiliations = []
    
    for affil in affiliations:
        if affil not in unique_affiliations:
            studies = company_funding_rows[
                (company_funding_rows['Affiliations'].str.contains(affil, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_affiliations.append(f"{affil} ({study_count})")
            unique_affiliations.append(affil)
    labeled_affiliations = sorted(labeled_affiliations, key=count_key, reverse=True)
    # Researchers
    unique_researchers = []
    labeled_researchers = []
    
    for res in res_list:
        if res not in unique_researchers:
            studies = company_funding_rows[
                (author_match_mask(company_funding_rows['Authors'], res))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            
            labeled_researchers.append(f"{res} ({study_count})")
            unique_researchers.append(res)
    labeled_researchers = sorted(labeled_researchers, key=count_key, reverse=True)
    # Universities
    unique_universities = []
    labeled_universities = []
    
    for uni in universities:
        if uni not in unique_universities:
            # Count studies mentioning this university
            studies = company_funding_rows[
                (company_funding_rows['Affiliations'].str.contains(uni, na=False, regex=False))
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_universities.append(f"{uni} ({study_count})")
            unique_universities.append(uni)
    labeled_universities = sorted(labeled_universities, key=count_key, reverse=True)
    return {
        "Affiliations": labeled_affiliations,
        "Countries": labeled_countries,
        "Researchers": labeled_researchers,
        "Universities": labeled_universities,
        "Chemicals": labeled_chemicals
    }

def show_uni_connections(university, uni_rows=None):
    row = comparing_unis[comparing_unis['University'] == university]
    if row.empty:
        print(f"University '{university}' not found.")
        return False
    if uni_rows is None:
        uni_rows = get_university_rows(university)

    parsed_chems = list(parse_chemical_entry(c) for c in row.iloc[0]['Chemicals'])
    companies = row.iloc[0]['Companies']

    # Chemicals
    labeled_chemicals = []
    processed_inchikeys = set()
    unique_no_inch_chemicals = []
    # Pre-filter to this university's rows once instead of scanning all of
    # `main` for every chemical.
    uni_main = main[
        main['Affiliations'].str.contains(university, na=False, regex=False)
    ]
    for name, inchikey in parsed_chems:
        if inchikey and inchikey != 'Not Found':
            if inchikey not in processed_inchikeys:
                # Chemicals with InChIKey
                studies = uni_main[
                    uni_main['Chemicals with InChIKey'].str.contains(inchikey, na=False, regex=False)
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                processed_inchikeys.add(inchikey)
        else:
            # Chemicals without InChIKey
            if name not in unique_no_inch_chemicals:
                studies = uni_main[
                    uni_main['Chemicals with InChIKey'].str.contains(name, na=False, regex=False)
                ]
                study_count = len(studies.drop_duplicates(subset=['DOI']))
                labeled_chemicals.append(f"{name} ({study_count})")
                unique_no_inch_chemicals.append(name)
    labeled_chemicals = sorted(labeled_chemicals, key=count_key, reverse=True)
    # Companies
    seen_company_keys = set()
    labeled_companies = []
    
    for comp in companies:
        original_name, _ = extract_name_and_class(comp)
        company_key = original_name.strip().lower()
        if company_key not in seen_company_keys:
            studies = uni_main[
                funding_source_match_mask(uni_main["Funding Sources"], original_name)
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{comp} ({study_count})")
            seen_company_keys.add(company_key)
    labeled_companies = sorted(labeled_companies, key=count_key, reverse=True)

    return {
        "Funding Sources": labeled_companies,
        "Chemicals": labeled_chemicals
    }

def show_res_connections(researcher, matches=None, researcher_rows=None, category='Funding Sources'):
    if matches is None:
        matches = get_researcher_matches(researcher)
    if matches.empty:
        print(f"Researcher: '{researcher}' not found.")
        return False
    if researcher_rows is None:
        researcher_rows = get_researcher_rows(researcher)
    
    if len(matches) > 1:
        all_companies = sum(matches['Companies'], [])
        all_collaborators = sum(matches['Collaborators'], []) if 'Collaborators' in matches.columns else []
        unique_affiliations = matches['Affiliation'].dropna().unique()
        combined_aff = '; '.join(unique_affiliations)
        row = {
            'Researcher': researcher,
            'Affiliation': combined_aff,
            'Companies': all_companies,
            'Collaborators': all_collaborators,
        }
    else:
        row = matches.iloc[0]

    category_key = 'Collaborators' if str(category).strip().lower() == 'collaborators' else 'Funding Sources'
    data = row['Collaborators'] if category_key == 'Collaborators' and 'Collaborators' in row else row['Companies']
    aff = row['Affiliation']
    if aff == '':
        aff = 'Not Found'

    seen_company_keys = set()
    labeled_companies = []

    for comp in data:
        if category_key == 'Collaborators':
            original_name, _ = parse_collaborator_entry(comp)
        else:
            original_name, _ = extract_name_and_class(comp)
        if not original_name:
            continue
        company_key = original_name.strip().lower()
        if company_key not in seen_company_keys:
            if category_key == 'Collaborators':
                studies = researcher_rows[
                    author_match_mask(researcher_rows["Authors"], original_name)
                ]
            else:
                studies = researcher_rows[
                    funding_source_match_mask(researcher_rows["Funding Sources"], original_name)
                ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{original_name} ({study_count})")
            seen_company_keys.add(company_key)
    labeled_companies = sorted(labeled_companies, key=count_key, reverse=True)

    if category_key == 'Collaborators':
        return {
            "Affiliation(s)": aff,
            "Collaborators": labeled_companies,
        }

    return {
        "Affiliation(s)": aff,
        "Funding Sources": labeled_companies,
    }

def show_chem_connections(chemical=None, inchikey=None, row=None):
    if row is None:
        row = get_chemical_row(chemical=chemical, inchikey=inchikey)
    if row.empty:
        return False
    
    if row.empty:
        print(f"Chemical '{chemical}' not found.")
        return False
    
    data = row.iloc[0]['company']
    inchikey_val = row.iloc[0]['inchikey']
    if inchikey_val == 'Error':
        inchikey_val = 'Not Found'

    # Companies
    seen_company_keys = set()
    labeled_companies = []
    # Pre-filter to this chemical's rows once instead of scanning all of `main`
    # for every company.
    chem_key = inchikey_val if (inchikey_val and inchikey_val != 'Not Found') else chemical
    chem_main = main[
        main['Chemicals with InChIKey'].str.contains(chem_key, na=False, regex=False)
    ]
    for comp in data:
        original_name, _ = extract_name_and_class(comp)
        company_key = original_name.strip().lower()
        if company_key not in seen_company_keys:
            studies = chem_main[
                funding_source_match_mask(chem_main["Funding Sources"], original_name)
            ]
            study_count = len(studies.drop_duplicates(subset=['DOI']))
            labeled_companies.append(f"{comp} ({study_count})")
            seen_company_keys.add(company_key)
    labeled_companies = sorted(labeled_companies, key=count_key, reverse=True)
    return {
        "Inchikey": inchikey_val,
        "Funding Sources": labeled_companies
    }

# code for creating "periodic table of companies"
classification_file_path = os.path.join(settings.BASE_DIR, 'data', 'company_classifications.json')
try:
    with open(classification_file_path, 'r', encoding='utf-8') as f:
        company_classification_dict = json.load(f)
except FileNotFoundError:
    company_classification_dict = {}
    print(f"Company classifications file not found at {classification_file_path}; using empty cache")
except Exception as e:
    company_classification_dict = {}
    print(f"Error loading company classifications: {e}; using empty cache")

def get_pubchem_image_url(chemical_name, inchikey=None):
    cache_key = _pubchem_cache_key(chemical_name=chemical_name, inchikey=inchikey)
    image_cache = _get_pubchem_image_cache()
    cached_url = image_cache.get(cache_key)
    if isinstance(cached_url, str) and cached_url.strip():
        return cached_url

    try:
        if inchikey and inchikey != 'Error':
            try:
                compounds = pcp.get_compounds(inchikey, 'inchikey')
                if compounds:
                    cid = compounds[0].cid
                    image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
                    image_cache[cache_key] = image_url
                    _save_json_cache(_PUBCHEM_IMAGE_CACHE_FILE, image_cache)
                    return image_url
            except:
                pass
        if chemical_name:
            try:
                compounds = pcp.get_compounds(chemical_name, 'name')
                if compounds:
                    cid = compounds[0].cid
                    image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
                    image_cache[cache_key] = image_url
                    _save_json_cache(_PUBCHEM_IMAGE_CACHE_FILE, image_cache)
                    return image_url
            except:
                pass
    except Exception as e:
        print(f"Error fetching PubChem data for InChIKey {inchikey}: {e}")

    # Final fallback: return cached value if network lookups failed.
    cached_url = image_cache.get(cache_key)
    if isinstance(cached_url, str) and cached_url.strip():
        return cached_url
    return None

def get_top_chemicals_for_company(company_name, limit=5):
    try:
        company_studies = main[funding_source_match_mask(main["Funding Sources"], company_name)]

        if company_studies.empty:
            return []
        
        all_chemicals = []
        for chemicals_str in company_studies['Chemicals with InChIKey'].dropna():
            chemicals = chemicals_str.split(';')
            for chemical in chemicals:  
                chemical = chemical.strip()  
                if chemical and '(' in chemical:
                    name = chemical.split('(')[0].strip()
                    if name:
                        all_chemicals.append(name)
                elif chemical:
                    all_chemicals.append(chemical)
        chemical_counts = Counter(all_chemicals)
        top_chemicals = chemical_counts.most_common(limit)
        
        return top_chemicals
    except Exception as e:
        print(f"Error getting top chemicals for company {company_name}: {e}")
        return []

def get_pubchem_description(chemical_name, inchikey=None):
    """Get comprehensive PubChem description for a chemical compound."""
    cache_key = _pubchem_cache_key(chemical_name=chemical_name, inchikey=inchikey)
    desc_cache = _get_pubchem_desc_cache()
    cached_description = desc_cache.get(cache_key)
    if isinstance(cached_description, str) and cached_description.strip():
        return cached_description

    try:
        compound = None
        
        if inchikey and inchikey != 'Error':
            try:
                compounds = pcp.get_compounds(inchikey, 'inchikey')
                if compounds:
                    compound = compounds[0]
            except:
                pass
        
        # Fall back to chemical name if inchikey didn't work
        if not compound and chemical_name:
            try:
                compounds = pcp.get_compounds(chemical_name, 'name')
                if compounds:
                    compound = compounds[0]
            except:
                pass
        
        if compound:
            try:
                # Try to get description text from PubChem's REST API
                import requests
                import json
                
                # Get description from PubChem's compound summary
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{compound.cid}/JSON"
                print(f"Fetching PubChem data from: {url}")  # Debug line
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for high-priority descriptions in specific sections
                    if 'Record' in data and 'Section' in data['Record']:
                        priority_descriptions = []
                        all_descriptions = []
                        
                        def extract_from_information(info_list, section_heading=""):
                            """Extract text from Information array with priority based on section."""
                            texts = []
                            for info in info_list:
                                # Check Description field
                                description_text = info.get('Description', '')
                                
                                # Prioritize certain descriptions
                                priority_keywords = ['record description', 'drug summary', 'fda pharmacology', 
                                                   'physical description', 'livertox summary', 'mechanism of action']
                                is_priority = any(keyword in description_text.lower() for keyword in priority_keywords)
                                
                                # Extract from StringValue 
                                if 'StringValue' in info:
                                    text = info['StringValue'].strip()
                                    if len(text) > 30:
                                        if is_priority:
                                            priority_descriptions.append(text)
                                        texts.append(text)
                                
                                # Extract from Value -> StringWithMarkup
                                elif 'Value' in info and isinstance(info['Value'], dict):
                                    if 'StringWithMarkup' in info['Value']:
                                        for markup in info['Value']['StringWithMarkup']:
                                            if isinstance(markup, dict) and 'String' in markup:
                                                text = markup['String'].strip()
                                                if len(text) > 30:
                                                    if is_priority:
                                                        priority_descriptions.append(text)
                                                    texts.append(text)
                            return texts
                        
                        def process_section(section, depth=0):
                            """Recursively process sections, prioritizing key ones."""
                            if depth > 3:
                                return
                                
                            section_heading = section.get('TOCHeading', '').lower()
                            
                            # High priority sections for rich descriptions
                            high_priority = ['names and identifiers', 'drug and medication information', 
                                           'pharmacology and biochemistry']
                            is_high_priority = any(priority in section_heading for priority in high_priority)
                            
                            if 'Information' in section:
                                texts = extract_from_information(section['Information'], section_heading)
                                if is_high_priority:
                                    priority_descriptions.extend(texts[:3])  # Take first 3 from priority sections
                                all_descriptions.extend(texts)
                            
                            # Recurse into subsections
                            if 'Section' in section:
                                for subsection in section['Section']:
                                    process_section(subsection, depth + 1)
                        
                        # Process all sections
                        for section in data['Record']['Section']:
                            process_section(section)
                        
                        # Return best description based on priority and content quality
                        # First try priority descriptions
                        for desc in priority_descriptions:
                            # Look for rich pharmaceutical/medical descriptions
                            if len(desc) > 100 and any(word in desc.lower() for word in [
                                'is a', 'medication', 'drug', 'agent', 'compound', 'used for', 
                                'treatment', 'inhibitor', 'analgesic', 'anti-inflammatory', 'therapeutic'
                            ]):
                                desc_cache[cache_key] = desc
                                _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                                return desc
                            
                        # Then try substantial descriptions with good keywords
                        for desc in all_descriptions:
                            if len(desc) > 80 and any(word in desc.lower() for word in [
                                'is a member of', 'belongs to', 'medication', 'drug', 'therapeutic',
                                'appears as', 'crystalline', 'powder', 'used for', 'treatment of'
                            ]):
                                desc_cache[cache_key] = desc
                                _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                                return desc
                        
                        # Finally, try any decent description
                        for desc in all_descriptions:
                            if len(desc) > 50:
                                desc_cache[cache_key] = desc
                                _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                                return desc
                
                # Fallback to basic compound information
                full_record = pcp.Compound.from_cid(compound.cid)
                
                description_parts = []
                
                if hasattr(full_record, 'iupac_name') and full_record.iupac_name:
                    description_parts.append(f"IUPAC Name: {full_record.iupac_name}")
                    
                if hasattr(full_record, 'molecular_formula') and full_record.molecular_formula:
                    formula_info = f"Molecular Formula: {full_record.molecular_formula}"
                    if hasattr(full_record, 'molecular_weight') and full_record.molecular_weight:
                        formula_info += f" | Molecular Weight: {full_record.molecular_weight} g/mol"
                    description_parts.append(formula_info)
                
                if description_parts:
                    description = " | ".join(description_parts)
                    desc_cache[cache_key] = description
                    _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                    return description
                else:
                    description = f"PubChem CID: {compound.cid}"
                    desc_cache[cache_key] = description
                    _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                    return description
                    
            except Exception as e:
                print(f"Error in detailed lookup: {e}")
                # Simple fallback
                try:
                    full_record = pcp.Compound.from_cid(compound.cid)
                    if hasattr(full_record, 'molecular_formula') and full_record.molecular_formula:
                        description = f"Molecular Formula: {full_record.molecular_formula}"
                        desc_cache[cache_key] = description
                        _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                        return description
                    description = f"PubChem CID: {compound.cid}"
                    desc_cache[cache_key] = description
                    _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                    return description
                except:
                    description = f"PubChem CID: {compound.cid}"
                    desc_cache[cache_key] = description
                    _save_json_cache(_PUBCHEM_DESC_CACHE_FILE, desc_cache)
                    return description
                
    except Exception as e:
        print(f"Error fetching PubChem description for {chemical_name} (InChIKey: {inchikey}): {e}")

    # Final fallback: return cached value if lookups fail.
    cached_description = desc_cache.get(cache_key)
    if isinstance(cached_description, str) and cached_description.strip():
        return cached_description
    
    return None
def get_wikipedia_description_fundingsource(funding_source):
    if not funding_source or pd.isna(funding_source):
        return None

    cache_key = _funding_cache_key(funding_source)
    desc_cache = _get_funding_desc_cache()
    cached_value = desc_cache.get(cache_key)
    if isinstance(cached_value, dict) and cached_value.get('description'):
        return cached_value

    try:
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        funding_source_encoded = funding_source.replace(' ', '_')
        headers = {
            'User-Agent': 'ChemNet Research Tool (no-email@example.com)'
        }
        response = requests.get(f"{search_url}{funding_source_encoded}", headers=headers, timeout=200)

        if response.status_code == 200:
            data = response.json()
        
            extract = data.get('extract', '')

            if extract:
                result = {
                    'description': extract,
                    'title' : data.get('title', funding_source),
                    'url' : data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'thumbnail': data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else None
                }
                desc_cache[cache_key] = result
                _save_json_cache(_FUNDING_DESC_CACHE_FILE, desc_cache)
                return result
    except Exception as e:
        print(f"Error fetching Wikipedia description for {funding_source}: {e}")

    # Network fallback: return cached value when available.
    cached_value = desc_cache.get(cache_key)
    if isinstance(cached_value, dict) and cached_value.get('description'):
        return cached_value

    return None

company_counts = {}
# for company in company_classification_dict.keys():
#     if company and company.strip():
#         if company.strip().lower() != 'not found':
#             count = funding_source_match_mask(main["Funding Sources"], company).sum()
#             company_counts[company] = count


def _compute_company_counts_from_all_studies():
    """Count funding-source occurrences directly from all studies in main dataframe."""
    if 'Funding Sources' not in main.columns:
        return {}

    counts_series = (
        main['Funding Sources']
        .dropna()
        .astype(str)
        .str.split(r'[;]')
        .explode()
        .astype(str)
        .str.strip()
    )
    counts_series = counts_series[counts_series != '']
    return counts_series.value_counts().to_dict()


def _normalized_classification(value):
    text = str(value or '').strip().lower()
    return text if text else 'unknown'

def create_funding_source_dataframe(
    chem_limit=5,
    top_n=50,
    category='all',
    counts_override=None,
    auto_build_category_csvs=True,
):
    """
    Build funding table dataframe from the same company_assoc pipeline used by company search.
    This keeps funding table popup chemicals/counts consistent with company page results.
    """
    category_key = _normalize_funding_table_category(category)
    company_assoc_names = set(company_assoc['Company'].dropna().astype(str).str.strip())
    source_counts = counts_override if counts_override is not None else (company_counts if company_counts else _compute_company_counts_from_all_studies())

    sorted_companies = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    top_companies = []
    for name, count in sorted_companies:
        if name not in company_assoc_names:
            continue
        if category_key != 'all':
            classification = _normalized_classification(company_classification_dict.get(name, 'Unknown'))
            if classification != category_key:
                continue
        top_companies.append((name, count))
        if top_n is not None and len(top_companies) >= max(1, int(top_n)):
            break

    rows = []
    total = len(top_companies)

    for i, (company, study_count) in enumerate(top_companies, 1):
        print(f"Processing {i}/{total}: {company}")

        try:
            classification = company_classification_dict.get(company, 'Unknown')

            connections = show_company_connections(company)
            chem_entries = []
            if isinstance(connections, dict):
                chem_entries = connections.get('Chemicals', []) or []

            chem_entries_sorted = sorted(chem_entries, key=count_key, reverse=True)
            top_chemicals = ";".join(chem_entries_sorted[:chem_limit])

            wiki = get_wikipedia_description_fundingsource(company)
            description = wiki.get('description', '') if isinstance(wiki, dict) else ""
            wiki_url = wiki.get('url', '') if isinstance(wiki, dict) else ""

            rows.append({
                "company": company,
                "study_count": int(study_count),
                "classification": classification,
                "top_chemicals": top_chemicals,
                "description": description,
                "wiki_url": wiki_url,
            })
        except Exception as e:
            print(f"⚠️ Error processing {company}: {e}")
            rows.append({
                "company": company,
                "study_count": int(study_count),
                "classification": company_classification_dict.get(company, 'Unknown'),
                "top_chemicals": "",
                "description": "",
                "wiki_url": "",
            })

    print(f"✓ Completed! Processed {len(rows)} companies")
    result_df = pd.DataFrame(rows)

    # Keep category tables in sync whenever the main/all funding table is rebuilt.
    if auto_build_category_csvs and category_key == 'all' and counts_override is None:
        try:
            create_funding_source_category_dataframes(top_n=top_n, source_counts=source_counts)
        except Exception as exc:
            logger.warning("Failed to auto-build funding source category CSVs: %s", exc)

    return result_df

FUNDING_SOURCE_TABLE_URL = "https://ucsf.box.com/shared/static/ghk9hv5p7fuzoquqa54xjyaiwyp0za8g.csv"
funding_source_table_df = load_remote_csv(FUNDING_SOURCE_TABLE_URL, 'funding_source_table.csv')

_FUNDING_TABLE_CATEGORIES = ('all', 'government', 'university', 'foundation', 'company', 'unknown')


def _normalize_funding_table_category(category):
    normalized = str(category or 'all').strip().lower()
    return normalized if normalized in _FUNDING_TABLE_CATEGORIES else 'all'


def _funding_table_category_path(category):
    category_key = _normalize_funding_table_category(category)
    return os.path.join(settings.BASE_DIR, 'data', f'funding_source_table_{category_key}.csv')


def _prepare_funding_table_rows(df, category='all', top_n=50):
    rows = df.copy()
    category_key = _normalize_funding_table_category(category)
    if category_key != 'all':
        rows = rows[
            rows.get('classification', '').fillna('').astype(str).str.lower() == category_key
        ]

    count_series = pd.to_numeric(rows.get('study_count', rows.get('count', 0)), errors='coerce').fillna(0)
    rows = rows.assign(_count_value=count_series)
    rows = rows.sort_values(by='_count_value', ascending=False)
    if top_n is not None:
        rows = rows.head(max(1, int(top_n)))
    return rows.drop(columns=['_count_value'], errors='ignore').reset_index(drop=True)


def create_funding_source_category_dataframes(top_n=50, source_counts=None):
    """Create per-category funding source CSVs from raw study-level counts."""
    if source_counts is None:
        source_counts = _compute_company_counts_from_all_studies()
    created_files = {}
    for category in _FUNDING_TABLE_CATEGORIES:
        rows = create_funding_source_dataframe(
            top_n=top_n,
            category=category,
            counts_override=source_counts,
            auto_build_category_csvs=False,
        )
        output_path = _funding_table_category_path(category)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rows.to_csv(output_path, index=False)
        created_files[category] = output_path
    return created_files


def load_funding_source_table_for_category(category='all', top_n=50):
    """Load prebuilt per-category funding table CSV; fallback to in-memory table when missing."""
    category_key = _normalize_funding_table_category(category)
    category_path = _funding_table_category_path(category_key)
    if os.path.exists(category_path):
        try:
            df = pd.read_csv(category_path)
            return _prepare_funding_table_rows(df, category='all', top_n=top_n)
        except Exception as exc:
            logger.warning("Failed to load category table %s: %s", category_path, exc)

    # Build on demand using raw study-level counts when prebuilt files are unavailable.
    source_counts = _compute_company_counts_from_all_studies()
    return create_funding_source_dataframe(top_n=top_n, category=category_key, counts_override=source_counts)


def parse_list_cell(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value in ("", None):
        return []
    if isinstance(value, str):
        # Split on semicolons and clean up
        items = [item.strip() for item in value.split(";") if item.strip()]
        return items
    return []

def parse_chemical_with_count(chem_str):
    """Parse 'chemical (count)' format into [name, count] pair."""
    import re
    match = re.match(r'^(.+?)\s+\((\d+)\)$', chem_str.strip())
    if match:
        return [match.group(1), int(match.group(2))]
    return [chem_str.strip(), 0]

def parse_chemicals_list(chemicals_str):
    """Parse semicolon-separated chemicals into [[name, count], ...] format."""
    if not chemicals_str:
        return []
    items = parse_list_cell(chemicals_str)
    return [parse_chemical_with_count(item) for item in items]
def get_funding_source_row(company_name):
    table_df = load_funding_source_table_for_category('all', top_n=1000000)
    row = table_df[table_df["company"] == company_name]
    if row.empty:
        row = funding_source_table_df[funding_source_table_df["company"] == company_name]
    if row.empty:
        return None
    record = row.iloc[0]
    
    description = record["description"]
    if pd.isna(description):
        description = ""
    
    wiki_url = record["wiki_url"]
    if pd.isna(wiki_url):
        wiki_url = ""
    
    return {
        "company": record["company"],
        "study_count": int(record["study_count"]),
        "classification": record["classification"],
        "top_chemicals": record["top_chemicals"],
        "description": description,
        "wiki_url": wiki_url,
    }
