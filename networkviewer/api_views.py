from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from pathlib import Path
from .serializers import (
    ChemicalSearchSerializer,
    CompanySearchSerializer, 
    UniversitySearchSerializer,
    ResearcherSearchSerializer
)
from .network_functions import (
    show_chemical_network,
    show_company_network_pyvis,
    show_uni_network_pyvis,
    show_researcher_network_pyvis_from_row,
    comparing_researchers,
    show_chem_connections,
    show_company_connections,
    show_uni_connections,
    show_res_connections,
    chem_per_row,
    no_dup_comp,
    comparing_unis,
    get_top_chemicals_for_company,
    get_pubchem_description,
    get_wikipedia_description_fundingsource,
    funding_source_table_df,
    get_funding_source_row,
    parse_chemicals_list,
    obtain_inchikey_from_pubchem,
    get_company_funding_rows,
    get_university_rows,
    get_chemical_row,
    get_pubchem_image_url,
)
import difflib
import random


def resolve_case_insensitive_name(query, valid_names):
    if query is None:
        return query
    query_str = str(query).strip()
    if not query_str:
        return query_str

    lookup = {}
    for name in valid_names:
        name_str = str(name).strip()
        if name_str and name_str.lower() not in lookup:
            lookup[name_str.lower()] = name_str

    return lookup.get(query_str.lower(), query_str)


def get_close_matches_custom(query, valid_names, n=3, cutoff=0.6):
    if query is None:
        return []
    query_str = str(query).strip()
    if not query_str:
        return []

    normalized_map = {}
    for name in valid_names:
        name_str = str(name).strip()
        if name_str and name_str.lower() not in normalized_map:
            normalized_map[name_str.lower()] = name_str

    matched_keys = difflib.get_close_matches(
        query_str.lower(),
        list(normalized_map.keys()),
        n=n,
        cutoff=cutoff,
    )
    return [normalized_map[key] for key in matched_keys]


def get_request_mode(request):
    mode = request.query_params.get('mode', 'full')
    return mode if mode in {'full', 'connections', 'graph'} else 'full'


def get_connection_threshold(request):
    raw_value = request.query_params.get('connection_threshold')
    if raw_value is not None:
        try:
            threshold = int(raw_value)
        except (TypeError, ValueError):
            threshold = 0
    else:
        legacy_value = str(request.query_params.get('drop_singletons', '')).strip().lower()
        threshold = 1 if legacy_value in {'1', 'true', 'yes', 'y'} else 0

    return threshold if threshold in {0, 1, 2, 3} else 0


def load_graph_html(iframe_url):
    if not iframe_url:
        return None
    filename = iframe_url.rsplit('/', 1)[-1]
    graph_path = Path(__file__).resolve().parent.parent / 'staticfiles' / filename
    if not graph_path.is_file():
        return None
    try:
        return graph_path.read_text(encoding='utf-8')
    except Exception:
        return None

class ChemicalSearchAPI(APIView):
    def get(self, request):
        # Return example chemicals and all chemical names for autocomplete
        all_chemical_names = sorted((name for names in chem_per_row['chemical'] for name in names))
        example_chemicals = [
            {"name": "Aspirin", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
            {"name": "Caffeine", "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N"},
            {"name": "Glucose", "inchikey": "WQZGKKKJIJFFOK-GASJEMHNSA-N"},
            {"name": "Iron", "inchikey": ""},
            {"name": "Goethite", "inchikey": ""},
            {"name": "PAHs", "inchikey": ""},
            {"name": "Au", "inchikey": ""},
            {"name": "Copper", "inchikey": ""}
        ]
        random_examples = random.sample(example_chemicals, 3)
        
        return Response({
            'example_chemicals': random_examples,
            'all_chemical_names': all_chemical_names
        })
    
    def post(self, request):
        serializer = ChemicalSearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        mode = get_request_mode(request)
        connection_threshold = get_connection_threshold(request)
        
        data = serializer.validated_data
        chemical = data.get('chemical', '').strip()
        inchikey = data.get('inchikey', '').strip().upper()
        all_chemical_names = sorted((name for names in chem_per_row['chemical'] for name in names))
        if chemical:
            chemical = resolve_case_insensitive_name(chemical, all_chemical_names)
        chemical_inputted = bool(chemical)

        if inchikey and not chemical:
            chemical_inputted = False
            row = chem_per_row[chem_per_row['inchikey'] == inchikey]
            if not row.empty and isinstance(row.iloc[0]['chemical'], list) and row.iloc[0]['chemical']:
                chemical = row.iloc[0]['chemical'][0]
            elif not row.empty and row.iloc[0]['chemical']:
                # defensive fallback if parsing changed
                chem_value = row.iloc[0]['chemical']
                chemical = chem_value[0] if isinstance(chem_value, list) and chem_value else str(chem_value)
            else:
                chemical = "inchikey_search"
        if not chemical and not inchikey:
            return Response({'error': 'Either chemical name or inchikey is required'}, 
                          status=status.HTTP_400_BAD_REQUEST)

        if mode == 'connections':
            if inchikey:
                connections = show_chem_connections(inchikey=inchikey)
                image_url = get_pubchem_image_url(chemical, inchikey)
                description = get_pubchem_description(chemical, inchikey if inchikey != 'Error' else None)
            else:
                row = get_chemical_row(chemical=chemical)
                if row is None:
                    suggestions = get_close_matches_custom(chemical or inchikey, all_chemical_names)
                    return Response({
                        'success': False,
                        'chemical': chemical,
                        'inchikey': inchikey,
                        'suggestions': suggestions,
                        'message': f"Chemical '{chemical or inchikey}' not found"
                    })
                connections = show_chem_connections(chemical, row=row)
                image_url = get_pubchem_image_url(chemical, inchikey)
                description = get_pubchem_description(chemical)

            return Response({
                'success': True,
                'chemical': chemical,
                'inchikey': inchikey,
                'connections': connections,
                'image_url': image_url,
                'description': description,
            })

        # Process search
        if inchikey:
            found = show_chemical_network(chemical, inch=inchikey, max_connection_count=connection_threshold)
            connections = show_chem_connections(inchikey=inchikey)
        elif chemical:
            found = show_chemical_network(chemical, inch='Error', max_connection_count=connection_threshold)
            connections = show_chem_connections(chemical)
        
        if found:
            singleton_suffix = ''
            if connection_threshold == 1:
                singleton_suffix = '_no_singletons'
            elif connection_threshold > 1:
                singleton_suffix = f'_le{connection_threshold}'
            if chemical and inchikey and inchikey != 'Error':
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                iframe_url = f"/networks/network_{safe_chemical}_{safe_inch}{singleton_suffix}.html"
            else:
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe_url = f"/networks/network_{safe_chemical}_no_inchikey{singleton_suffix}.html"
            
            # Get PubChem description
            description = get_pubchem_description(chemical, inchikey if inchikey != 'Error' else None)
            
            payload = {
                'success': True,
                'chemical': chemical,
                'inchikey': inchikey,
                'iframe_url': iframe_url,
                'graph_html': load_graph_html(iframe_url) if mode == 'graph' else None,
                'connections': connections,
                'description': description
            }
            if mode == 'graph':
                payload.pop('connections', None)
                payload.pop('description', None)
            return Response(payload)
        else:
            if chemical and chemical_inputted:
                inchikey = obtain_inchikey_from_pubchem(chemical)
                if inchikey:
                    found = show_chemical_network(chemical, inch=inchikey, max_connection_count=connection_threshold)
                    connections = show_chem_connections(inchikey=inchikey)
                if found:
                    singleton_suffix = ''
                    if connection_threshold == 1:
                        singleton_suffix = '_no_singletons'
                    elif connection_threshold > 1:
                        singleton_suffix = f'_le{connection_threshold}'
                    if chemical and inchikey and inchikey != 'Error':
                        safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                        safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                        iframe_url = f"/networks/network_{safe_chemical}_{safe_inch}{singleton_suffix}.html"
                    description = get_pubchem_description(chemical, inchikey if inchikey != 'Error' else None)

                    payload = {
                        'success': True,
                        'chemical': chemical,
                        'inchikey': inchikey,
                        'iframe_url': iframe_url,
                        'graph_html': load_graph_html(iframe_url) if mode == 'graph' else None,
                        'connections': connections,
                        'description': description
                    }
                    if mode == 'graph':
                        payload.pop('connections', None)
                        payload.pop('description', None)
                    return Response(payload)
                else:
                    all_chemical_names = sorted((name for names in chem_per_row['chemical'] for name in names))
                    suggestions = get_close_matches_custom(chemical or inchikey, all_chemical_names)
            else:
                all_chemical_names = sorted((name for names in chem_per_row['chemical'] for name in names))
                suggestions = get_close_matches_custom(chemical or inchikey, all_chemical_names)
    
            return Response({
                'success': False,
                'chemical': chemical,
                'inchikey': inchikey,
                'suggestions': suggestions,
                'message': f"Chemical '{chemical or inchikey}' not found"
            })

class CompanySearchAPI(APIView):
    def get(self, request):
        all_company_names = sorted(set(no_dup_comp))
        example_companies = [
            "Dow Chemical Company",
            "U.S. Department of Energy", 
            "U.S. Department of Agriculture",
            "BASF Corporation",
            "Agilent Foundation",
            "Natural Science Foundation of China",
            "U.S. Department of Defense"
        ]
        random_examples = random.sample(example_companies, 3)
        
        return Response({
            'example_companies': random_examples,
            'all_company_names': all_company_names,
            'category_options': ['Affiliations', 'Chemicals', 'Researchers', 'Universities'],
            'chemical_group_options': ['All', 'Organic']
        })
    
    def post(self, request):
        serializer = CompanySearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        mode = get_request_mode(request)
        connection_threshold = get_connection_threshold(request)
        
        data = serializer.validated_data
        all_company_names = sorted(set(no_dup_comp))
        company = resolve_case_insensitive_name(data['company'], all_company_names)
        category = data['category']
        chemical_group = data['chemical_group']
        sep_country = data['sep_country']
        company_funding_rows = get_company_funding_rows(company)

        if mode == 'connections':
            if company_funding_rows is None or company_funding_rows.empty:
                suggestions = get_close_matches_custom(company, all_company_names)
                return Response({
                    'success': False,
                    'company': company,
                    'suggestions': suggestions,
                    'message': f"Company '{company}' not found"
                })

            connections = show_company_connections(company, company_funding_rows=company_funding_rows)
            description = get_wikipedia_description_fundingsource(company)
            return Response({
                'success': True,
                'company': company,
                'connections': connections,
                'description': description,
            })
        
        found = show_company_network_pyvis(company, category=category, 
                                         chemical_group=chemical_group, 
                                         sep_country=sep_country,
                                         company_funding_rows=company_funding_rows,
                                         max_connection_count=connection_threshold)
        
        if found:
            singleton_suffix = ''
            if connection_threshold == 1:
                singleton_suffix = '_no_singletons'
            elif connection_threshold > 1:
                singleton_suffix = f'_le{connection_threshold}'
            safe_company = company.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe_url = f"/networks/network_{safe_company}_{safe_category}_all{singleton_suffix}.html"
                elif chemical_group == 'Organic':
                    iframe_url = f"/networks/network_{safe_company}_{safe_category}_organic{singleton_suffix}.html"
            elif category == 'Affiliations':
                if sep_country:
                    iframe_url = f"/networks/network_{safe_company}_{safe_category}_by_country{singleton_suffix}.html"
                else:
                    iframe_url = f"/networks/network_{safe_company}_{safe_category}_combined{singleton_suffix}.html"
            else:
                iframe_url = f"/networks/network_{safe_company}_{safe_category}{singleton_suffix}.html"
            
            connections = show_company_connections(company, company_funding_rows=company_funding_rows)
            
            description = get_wikipedia_description_fundingsource(company)

            payload = {
                'success': True,
                'company': company,
                'iframe_url': iframe_url,
                'graph_html': load_graph_html(iframe_url) if mode == 'graph' else None,
                'connections': connections,
                'description': description
            }
            if mode == 'graph':
                payload.pop('connections', None)
                payload.pop('description', None)
            return Response(payload)
        else:
            all_company_names = sorted(set(no_dup_comp))
            suggestions = get_close_matches_custom(company, all_company_names)
            
            return Response({
                'success': False,
                'company': company,
                'suggestions': suggestions,
                'message': f"Company '{company}' not found"
            })

class UniversitySearchAPI(APIView):
    def get(self, request):
        all_university_names = sorted(comparing_unis['University'].dropna().unique())
        example_universities = [
            "Harvard University",
            "Stanford University", 
            "Massachusetts Institute of Technology",
            "University of Cambridge",
            "University of Oxford",
            "California Institute of Technology",
            "Princeton University",
            "Yale University",
            "University of Chicago",
            "Columbia University",
            "New York University"
        ]
        random_examples = random.sample(example_universities, 3)
        
        return Response({
            'example_universities': random_examples,
            'all_university_names': all_university_names,
            'category_options': ['Chemicals', 'Funding Sources'],
            'chemical_group_options': ['All', 'Organic']
        })
    
    def post(self, request):
        serializer = UniversitySearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        mode = get_request_mode(request)
        connection_threshold = get_connection_threshold(request)
        
        data = serializer.validated_data
        all_university_names = sorted(comparing_unis['University'].dropna().unique())
        university = resolve_case_insensitive_name(data['university'], all_university_names)
        category = data['category']
        chemical_group = data['chemical_group']
        uni_rows = get_university_rows(university)

        if mode == 'connections':
            if uni_rows is None or uni_rows.empty:
                suggestions = get_close_matches_custom(university, all_university_names)
                return Response({
                    'success': False,
                    'university': university,
                    'suggestions': suggestions,
                    'message': f"University '{university}' not found"
                })

            connections = show_uni_connections(university, uni_rows=uni_rows)
            return Response({
                'success': True,
                'university': university,
                'connections': connections,
            })
        
        found = show_uni_network_pyvis(university, category=category, 
                                     chemical_group=chemical_group,
                                     uni_rows=uni_rows,
                                     max_connection_count=connection_threshold)
        
        if found:
            singleton_suffix = ''
            if connection_threshold == 1:
                singleton_suffix = '_no_singletons'
            elif connection_threshold > 1:
                singleton_suffix = f'_le{connection_threshold}'
            safe_uni = university.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe_url = f"/networks/network_{safe_uni}_{safe_category}_all{singleton_suffix}.html"
                elif chemical_group == 'Organic':
                    iframe_url = f"/networks/network_{safe_uni}_{safe_category}_organic{singleton_suffix}.html"
            else:
                iframe_url = f"/networks/network_{safe_uni}_{safe_category}{singleton_suffix}.html"
            
            connections = show_uni_connections(university, uni_rows=uni_rows)

            payload = {
                'success': True,
                'university': university,
                'iframe_url': iframe_url,
                'graph_html': load_graph_html(iframe_url) if mode == 'graph' else None,
                'connections': connections
            }
            if mode == 'graph':
                payload.pop('connections', None)
            return Response(payload)
        else:
            all_university_names = sorted(comparing_unis['University'].dropna().unique())
            suggestions = get_close_matches_custom(university, all_university_names)
            
            return Response({
                'success': False,
                'university': university,
                'suggestions': suggestions,
                'message': f"University '{university}' not found"
            })

class ResearcherSearchAPI(APIView):
    def get(self, request):
        all_researcher_names = sorted(comparing_researchers['Researcher'].dropna().unique())
        example_researchers = [
            'Abrahamsson, Dimitri',
            'Jiang, Guibin',
            'Yang, Xin',
            'Xie, Hongyu',
            'Nikiforov, Vladimir A.',
            'Lynch, Iseult',
            'Pan, Wenxiao',
            'Restituito, Sophie',
            'Kyrtopoulos, Soterios A.',
            'Wei, Jing'
        ]
        random_examples = random.sample(example_researchers, 3)
        
        return Response({
            'example_researchers': random_examples,
            'all_researcher_names': all_researcher_names
        })
    
    def post(self, request):
        serializer = ResearcherSearchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        researcher = data['researcher'].strip()
        selected_index = data.get('selected_index')
        combine = data.get('combine', False)
        
        # Find all matches
        all_matches = comparing_researchers[comparing_researchers['Researcher'].str.lower() == researcher.lower()]
        matches = all_matches.to_dict('records')
        
        if not matches:
            all_researcher_names = sorted(comparing_researchers['Researcher'].dropna().unique())
            suggestions = get_close_matches_custom(researcher, all_researcher_names)
            
            return Response({
                'success': False,
                'researcher': researcher,
                'suggestions': suggestions,
                'message': f"Researcher '{researcher}' not found"
            })

        researcher = all_matches.iloc[0]['Researcher']
        
        if len(matches) == 1:
            # Only one match, generate graph immediately
            row = matches[0]
            found = show_researcher_network_pyvis_from_row(row)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe_url = f"/networks/network_{safe_researcher}_{safe_aff}.html"
                connections = show_res_connections(researcher=researcher)
                
                return Response({
                    'success': True,
                    'researcher': researcher,
                    'iframe_url': iframe_url,
                    'connections': connections,
                    'matches': matches
                })
        
        elif selected_index is not None or combine:
            if combine:
                # Combine all companies and affiliations
                all_companies = sum(all_matches['Companies'], [])
                unique_affiliations = all_matches['Affiliation'].dropna().unique()
                combined_aff = '; '.join(unique_affiliations)
                row = {
                    'Researcher': researcher,
                    'Affiliation': combined_aff,
                    'Companies': all_companies
                }
            else:
                row = matches[int(selected_index)]
            
            found = show_researcher_network_pyvis_from_row(row)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe_url = f"/networks/network_{safe_researcher}_{safe_aff}.html"
                connections = show_res_connections(researcher)
                
                return Response({
                    'success': True,
                    'researcher': researcher,
                    'iframe_url': iframe_url,
                    'connections': connections,
                    'matches': matches
                })
        
        # Multiple matches and no selection yet, return options
        return Response({
            'success': False,
            'researcher': researcher,
            'matches': matches,
            'needs_selection': True,
            'message': f"Multiple matches found for '{researcher}'. Please select one or combine all."
        })

class FundingTableAPI(APIView):
    def get(self, request):
        company_name = request.GET.get('company_name')
        if company_name:
            known_names = funding_source_table_df['company'].dropna().astype(str).tolist()
            company_name = resolve_case_insensitive_name(company_name, known_names)
        if company_name:
            try:
                row_data = get_funding_source_row(company_name)
                if row_data:
                    # Parse chemicals with counts from semicolon-separated string
                    top_chemicals = parse_chemicals_list(row_data.get('top_chemicals', ''))[:5]

                    description_text = row_data.get('description')
                    description = {
                        'title': row_data.get('company', company_name),
                        'description': description_text or '',
                        'url': row_data.get('wiki_url') or '',
                        'thumbnail': None,
                    } if description_text else None

                    return Response({
                        'success': True,
                        'company_name': company_name,
                        'top_chemicals': top_chemicals,
                        'description': description
                    })

                top_chemicals = get_top_chemicals_for_company(company_name, limit=5)
                description = get_wikipedia_description_fundingsource(company_name)
                
                return Response({
                    'success': True,
                    'company_name': company_name,
                    'top_chemicals': top_chemicals,
                    'description': description
                })
            except Exception as e:
                return Response({
                    'success': False,
                    'error': str(e)
                }, status=500)
        else:
            try:
                funding_data = []
                for _, row in funding_source_table_df.iterrows():
                    count_value = row.get('study_count', row.get('count', 0))
                    funding_data.append({
                        'company': row.get('company', ''),
                        'count': int(count_value) if count_value is not None else 0,
                        'classification': row.get('classification', 'Unknown')
                    })
                
                return Response({
                    'success': True,
                    'funding_data': funding_data,
                    'total_companies': len(funding_data)
                })
            except Exception as e:
                return Response({
                    'success': False,
                    'error': str(e)
                }, status=500)
