import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import click


class ObservationRepository:
    def __init__(self, endpoint, page_size=50):
        self.endpoint = endpoint
        self.page_size = page_size

    def get_observations(self, patient_id):
        """Fetch all observations for a patient with pagination"""
        page_num = 1
        observations = []
        
        while True:
            try:
                observation_url = f"{self.endpoint}?patient={patient_id}&_count={self.page_size}&_page={page_num}"
                response = requests.get(observation_url)
                
                if response.status_code != 200:
                    break
                
                page_data = response.json()
                entries = page_data if isinstance(page_data, list) else page_data.get('entry', [])
                
                if len(entries) == 0:
                    break
                
                observations.extend(entries)
                
                # Check for next page
                next_link = None
                if not isinstance(page_data, list):
                    links = page_data.get('link', [])
                    next_link = next((link['url'] for link in links if link.get('relation') == 'next'), None)
                
                if not next_link:
                    break
                    
                page_num += 1
            except requests.exceptions.RequestException as error:
                print(f'Error fetching observations for patient {patient_id}: {error}')
                break
            except Exception as error:
                print(f'Unexpected error fetching observations for patient {patient_id}: {error}')
                break
        
        return observations


class ConditionRepository:
    def __init__(self, endpoint, page_size=50):
        self.endpoint = endpoint
        self.page_size = page_size

    def get_conditions(self, patient_id):
        """Fetch all conditions for a patient with pagination"""
        page_num = 1
        conditions = []
        
        while True:
            try:
                condition_url = f"{self.endpoint}?patient={patient_id}&_count={self.page_size}&_page={page_num}"
                response = requests.get(condition_url)
                
                if response.status_code != 200:
                    break
                
                page_data = response.json()
                entries = page_data if isinstance(page_data, list) else page_data.get('entry', [])
                
                if len(entries) == 0:
                    break
                
                conditions.extend(entries)
                
                # Check for next page
                next_link = None
                if not isinstance(page_data, list):
                    links = page_data.get('link', [])
                    next_link = next((link['url'] for link in links if link.get('relation') == 'next'), None)
                
                if not next_link:
                    break
                    
                page_num += 1
            except requests.exceptions.RequestException as error:
                print(f'Error fetching conditions for patient {patient_id}: {error}')
                break
            except Exception as error:
                print(f'Unexpected error fetching conditions for patient {patient_id}: {error}')
                break
        
        return conditions


class ObservationValueExtractor:
    """Handles all FHIR observation value types"""
    
    @staticmethod
    def extract_value(resource):
        """Extract value from observation resource supporting multiple FHIR value types"""
        # Try valueQuantity first (most common for measurements)
        if 'valueQuantity' in resource and resource['valueQuantity']:
            return resource['valueQuantity'].get('value')
        
        # Try valueInteger
        if 'valueInteger' in resource and resource['valueInteger'] is not None:
            return resource['valueInteger']
        
        # Try valueString
        if 'valueString' in resource and resource['valueString']:
            return resource['valueString']
        
        # Try valueBoolean
        if 'valueBoolean' in resource and resource['valueBoolean'] is not None:
            return resource['valueBoolean']
        
        # Try valueCodeableConcept
        if 'valueCodeableConcept' in resource and resource['valueCodeableConcept']:
            coding = resource['valueCodeableConcept'].get('coding', [])
            if coding and len(coding) > 0 and coding[0].get('display'):
                return coding[0]['display']
            
            # Try valueCodeableConcept text
            if resource['valueCodeableConcept'].get('text'):
                return resource['valueCodeableConcept']['text']
        
        return None


# Registering a custom accessor for pandas Series
@pd.api.extensions.register_series_accessor("snomed_cts")
class SnomedAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.Series):
            raise AttributeError("Must be a pandas Series")

    @property
    def metadata(self):
        df = self._obj._dataframe
        column_name = self._obj.name
        if hasattr(df, '_snomed_cts_metadata'):
            return df._snomed_cts_metadata.get(column_name, None)
        return None
    
    @property
    def display_name(self):
        metadata = self.metadata
        if metadata:
            return metadata.get('display_name', self._obj.name)
        return self._obj.name
    
    @property
    def code(self):
        metadata = self.metadata
        if metadata:
            return metadata.get('code', None)
        return None


# Custom DataFrame class to store additional metadata
class CustomDataFrame(pd.DataFrame):
    _metadata = ['_snomed_cts_metadata']

    def __init__(self, *args, **kwargs):
        self._snomed_cts_metadata = {}
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return CustomDataFrame

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            result._dataframe = self
        return result

    def set_snomed_cts(self, column_name, metadata):
        self._snomed_cts_metadata[column_name] = metadata

    def get_snomed_cts(self, column_name):
        return self._snomed_cts_metadata.get(column_name, None)


class PatientDataProcessor:
    def __init__(self, observation_repo, condition_repo, patients_df):
        self.observation_repo = observation_repo
        self.condition_repo = condition_repo
        self.patients_df = patients_df
        self.observation_names = []
        self.condition_names = []

    def get_patient_data(self, patient_id):
        """Process observations and conditions for a single patient"""
        observations = self.observation_repo.get_observations(patient_id)
        conditions = self.condition_repo.get_conditions(patient_id)
        
        patient_data = {'patient_id': patient_id}
        
        # Track observation counts to handle multiple values
        observation_counts = {}
        
        # Process observations
        for observation in observations:
            resource = observation.get('resource', {})
            code_display_pairs = []
            
            if 'coding' in resource.get('code', {}):
                for coding in resource['code']['coding']:
                    display = coding.get('display')
                    if display:
                        code_display_pairs.append(display)
            
            if len(code_display_pairs) > 0:
                # Use the new ObservationValueExtractor
                value = ObservationValueExtractor.extract_value(resource)
                
                if value is not None:
                    base_observation_name = " | ".join(code_display_pairs)
                    
                    # Handle multiple observations of the same type
                    if base_observation_name not in observation_counts:
                        observation_counts[base_observation_name] = 0
                    observation_counts[base_observation_name] += 1
                    
                    # Create unique column name for multiple observations
                    if observation_counts[base_observation_name] == 1:
                        column_name = base_observation_name
                    else:
                        column_name = f"{base_observation_name} ({observation_counts[base_observation_name]})"
                    
                    patient_data[column_name] = value
                    
                    # Track observation names
                    for display in code_display_pairs:
                        if display not in self.observation_names:
                            self.observation_names.append(display)
                    
                    if column_name not in self.observation_names:
                        self.observation_names.append(column_name)
        
        # Process conditions
        for condition in conditions:
            resource = condition.get('resource', {})
            verification_status = resource.get('verificationStatus', {})
            coding = verification_status.get('coding', [{}])
            code = coding[0].get('code', None) if coding else None
            
            if code == 'confirmed':
                condition_coding = resource.get('code', {}).get('coding', [{}])
                condition_name = condition_coding[0].get('display', 'Unknown') if condition_coding else 'Unknown'
                patient_data[condition_name] = 1
                
                if condition_name not in self.condition_names:
                    self.condition_names.append(condition_name)
        
        return patient_data

    def process_patient_data(self):
        """Process all patient data and create normalized DataFrame"""
        patient_data_list = []
        
        for _, patient_row in self.patients_df.iterrows():
            patient_id = patient_row['patient_id']
            data = self.get_patient_data(patient_id)
            # Merge with existing patient info
            merged_data = {**patient_row.to_dict(), **data}
            patient_data_list.append(merged_data)
        
        # Normalize data - ensure all patients have all columns
        normalized_data = self._normalize(patient_data_list)
        
        # Convert to CustomDataFrame and set SNOMED CT metadata
        df_patient_data_with_all_features = CustomDataFrame(normalized_data)
        
        for display_name in self.observation_names:
            snomed_cts = self.get_snomed_cts(display_name)
            metadata = snomed_cts if snomed_cts else {}
            metadata['display_name'] = display_name
            df_patient_data_with_all_features.set_snomed_cts(display_name, metadata)
        
        for display_name in self.condition_names:
            snomed_cts = self.get_snomed_cts(display_name)
            metadata = snomed_cts if snomed_cts else {}
            metadata['display_name'] = display_name
            df_patient_data_with_all_features.set_snomed_cts(display_name, metadata)
        
        return df_patient_data_with_all_features

    def _normalize(self, processed_data):
        """Ensure all patients have all columns, filling missing with 0"""
        if not processed_data:
            return []
        
        # Collect all unique columns
        columns_set = set()
        for patient in processed_data:
            columns_set.update(patient.keys())
        columns = list(columns_set)
        
        # Fill missing columns with 0
        normalized_data = []
        for patient in processed_data:
            normalized_patient = {}
            for col in columns:
                normalized_patient[col] = patient.get(col, 0)
            normalized_data.append(normalized_patient)
        
        return normalized_data

    def get_observation_names(self):
        return self.observation_names
    
    def get_condition_names(self):
        return self.condition_names

    def get_snomed_cts(self, column_name):
        """Fetch SNOMED CT metadata for a given column name"""
        if self.patients_df.empty:
            return None
        
        # For demonstration purposes, fetch metadata for the first patient
        patient_id = self.patients_df['patient_id'].iloc[0]
        
        # Check observations
        if column_name in self.observation_names:
            observations = self.observation_repo.get_observations(patient_id)
            for observation in observations:
                resource = observation.get('resource', {})
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        display = coding.get('display', '')
                        # Handle both exact matches and numbered variations
                        if display == column_name or column_name.startswith(display):
                            return {
                                "display_name": coding.get("display", None),
                                "code": coding.get("code", None),
                                "system": coding.get("system", None)
                            }
        
        # Check conditions
        if column_name in self.condition_names:
            conditions = self.condition_repo.get_conditions(patient_id)
            for condition in conditions:
                resource = condition.get('resource', {})
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        if coding.get('display') == column_name:
                            return {
                                "display_name": coding.get("display", None),
                                "code": coding.get("code", None),
                                "system": coding.get("system", None)
                            }
        
        return None


class DataNormalizer:
    """Utility class for data normalization and statistics"""
    
    @staticmethod
    def normalize(processed_data):
        """Ensure all patients have all columns"""
        if not processed_data:
            return {'normalizedData': [], 'columns': []}
        
        # Collect all unique columns
        columns_set = set()
        for patient in processed_data:
            columns_set.update(patient.keys())
        columns = list(columns_set)
        
        # Fill missing columns with 0
        normalized_data = []
        for patient in processed_data:
            normalized_patient = {}
            for col in columns:
                normalized_patient[col] = patient.get(col, 0)
            normalized_data.append(normalized_patient)
        
        return {'normalizedData': normalized_data, 'columns': columns}
    
    @staticmethod
    def calculate_stats(patients_data, observation_count, condition_count):
        """Calculate statistics about the patient dataset"""
        if not patients_data:
            return {
                'totalPatients': 0,
                'activePatients': 0,
                'inactivePatients': 0,
                'genderDistribution': {},
                'observationCount': observation_count,
                'conditionCount': condition_count
            }
        
        active_count = sum(1 for p in patients_data if p.get('active', False))
        
        gender_counts = {}
        for p in patients_data:
            gender = p.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        return {
            'totalPatients': len(patients_data),
            'activePatients': active_count,
            'inactivePatients': len(patients_data) - active_count,
            'genderDistribution': gender_counts,
            'observationCount': observation_count,
            'conditionCount': condition_count
        }