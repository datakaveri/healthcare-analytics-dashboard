import pandas as pd
import requests


class FHIRData:
    def __init__(self, base_url, group_id):
        self.base_url = base_url
        self.group_id = group_id
        self.group_url = f"{base_url}/Group/{self.group_id}"
        self.patient_url = f"{base_url}/Patient/"
        self.patient_ids = []
        # Automatically fetch patient IDs on initialization
        self.fetch_patient_ids()

    def fetch_patient_ids(self):
        """Fetch patient IDs from the group endpoint"""
        try:
            response = requests.get(self.group_url)
            response.raise_for_status()
            group_data = response.json()
            patient_ids = []
            
            for member in group_data.get('member', []):
                if 'entity' in member and member['entity']['reference'].startswith('Patient/'):
                    patient_ids.append(member['entity']['reference'].split('/')[-1])
            
            self.patient_ids = patient_ids
            return patient_ids
        except requests.exceptions.RequestException as error:
            print(f'Error fetching patient IDs from {self.group_url}: {error}')
            raise error
        except Exception as error:
            print(f'Unexpected error fetching patient IDs: {error}')
            raise error

    def get_patient_data(self):
        """Fetch detailed data for all patients"""
        if not self.patient_ids:
            print("Warning: No patient IDs available. Call fetch_patient_ids() first.")
            return []
        
        patients_data = []
        
        for patient_id in self.patient_ids:
            try:
                patient_url = f"{self.patient_url}{patient_id}"
                response = requests.get(patient_url)
                
                if response.status_code != 200:
                    print(f'Warning: Failed to fetch patient {patient_id}, status code: {response.status_code}')
                    continue
                
                p = response.json()
                patients_data.append({
                    'patient_id': p['id'],
                    'gender': p.get('gender', 'unknown'),
                    'active': p.get('active', False),
                    'last_updated': p.get('meta', {}).get('lastUpdated', '')
                })
            except requests.exceptions.RequestException as error:
                print(f'Error fetching patient {patient_id}: {error}')
                continue
            except Exception as error:
                print(f'Unexpected error processing patient {patient_id}: {error}')
                continue
        
        return patients_data


class Patient:
    def __init__(self, patient_id, gender, active, last_updated):
        self.patient_id = patient_id
        self.gender = gender
        self.active = active
        self.last_updated = last_updated


class PatientRepository:
    def __init__(self, fhir_data):
        """
        Initialize PatientRepository with FHIRData instance
        
        Args:
            fhir_data: FHIRData instance (already initialized with base_url and group_id)
        """
        self.FHIRData = fhir_data
        self.patients = self._load_patients()

    def _load_patients(self):
        """Load patients from FHIR data"""
        patients_data = self.FHIRData.get_patient_data()
        patients = []
        for pat_data in patients_data:
            patient = Patient(**pat_data)
            patients.append(patient)
        return patients

    def get_patients_dataframe(self):
        """Convert patients to DataFrame format"""
        patients_data = []
        for patient in self.patients:
            patients_data.append({
                'patient_id': patient.patient_id,
                'gender': patient.gender,
                'active': patient.active,
                'last_updated': patient.last_updated
            })
        return pd.DataFrame(patients_data)


# Usage example:
# fhir = FHIRData(base_url="https://api.example.com", group_id="12345")
# repository = PatientRepository(fhir)
# patients_df = repository.get_patients_dataframe()
# print(patients_df)