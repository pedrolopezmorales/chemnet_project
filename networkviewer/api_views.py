from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
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
    parse_chemicals_list
)
import difflib
import random

def get_close_matches_custom(query, valid_names, n=3, cutoff=0.6):
    return difflib.get_close_matches(query, valid_names, n=n, cutoff=cutoff)

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
        
        data = serializer.validated_data
        chemical = data.get('chemical', '').strip()
        inchikey = data.get('inchikey', '').strip()
        
        if not chemical and not inchikey:
            return Response({'error': 'Either chemical name or inchikey is required'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Process search
        if inchikey:
            found = show_chemical_network(chemical, inch=inchikey)
            connections = show_chem_connections(inchikey=inchikey)
        elif chemical:
            found = show_chemical_network(chemical, inch='Error')
            connections = show_chem_connections(chemical)
        
        if found:
            if chemical and inchikey and inchikey != 'Error':
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_inch = inchikey.replace('/', '_').replace('\\', '_').replace('-', '_')
                iframe_url = f"/static/network_{safe_chemical}_{safe_inch}.html"
            else:
                safe_chemical = chemical.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe_url = f"/static/network_{safe_chemical}_no_inchikey.html"
            
            # Get PubChem description
            description = get_pubchem_description(chemical, inchikey if inchikey != 'Error' else None)
            
            return Response({
                'success': True,
                'chemical': chemical,
                'inchikey': inchikey,
                'iframe_url': iframe_url,
                'connections': connections,
                'description': description
            })
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
        
        data = serializer.validated_data
        company = data['company']
        category = data['category']
        chemical_group = data['chemical_group']
        sep_country = data['sep_country']
        
        found = show_company_network_pyvis(company, category=category, 
                                         chemical_group=chemical_group, 
                                         sep_country=sep_country)
        
        if found:
            safe_company = company.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe_url = f"/static/network_{safe_company}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe_url = f"/static/network_{safe_company}_{safe_category}_organic.html"
            elif category == 'Affiliations':
                if sep_country:
                    iframe_url = f"/static/network_{safe_company}_{safe_category}_by_country.html"
                else:
                    iframe_url = f"/static/network_{safe_company}_{safe_category}_combined.html"
            else:
                iframe_url = f"/static/network_{safe_company}_{safe_category}.html"
            
            connections = show_company_connections(company)
            
            description = get_wikipedia_description_fundingsource(company)

            return Response({
                'success': True,
                'company': company,
                'iframe_url': iframe_url,
                'connections': connections,
                'description': description
            })
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
        
        data = serializer.validated_data
        university = data['university']
        category = data['category']
        chemical_group = data['chemical_group']
        
        found = show_uni_network_pyvis(university, category=category, 
                                     chemical_group=chemical_group)
        
        if found:
            safe_uni = university.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
            safe_category = category.replace(' ', '_')
            
            if category == 'Chemicals':
                if chemical_group == 'All':
                    iframe_url = f"/static/network_{safe_uni}_{safe_category}_all.html"
                elif chemical_group == 'Organic':
                    iframe_url = f"/static/network_{safe_uni}_{safe_category}_organic.html"
            else:
                iframe_url = f"/static/network_{safe_uni}_{safe_category}.html"
            
            connections = show_uni_connections(university)
            
            return Response({
                'success': True,
                'university': university,
                'iframe_url': iframe_url,
                'connections': connections
            })
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
        researcher = data['researcher']
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
        
        elif len(matches) == 1:
            # Only one match, generate graph immediately
            row = matches[0]
            found = show_researcher_network_pyvis_from_row(row)
            if found:
                safe_researcher = researcher.replace(' ', '_').replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                safe_aff = str(row['Affiliation'])[:20].replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')
                iframe_url = f"/static/network_{safe_researcher}_{safe_aff}.html"
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
                iframe_url = f"/static/network_{safe_researcher}_{safe_aff}.html"
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