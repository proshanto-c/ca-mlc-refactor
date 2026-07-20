import os
from pathlib import Path
import pandas as pd

class BRSETDataSplitter:
    def __init__(self,
                 root_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 target_labels: list = ["diabetic_retinopathy",
                                        "macular_edema",
                                        "amd",
                                        "myopic_fundus",
                                        "increased_cup_disc"],
                 adequate_only: bool = True,
                 split_basis: str = "patient"):
        
        # Path Setup
        self.root = Path(root_dir)
        self.csv_path = self.root / "labels_brset.csv"
        self.image_directory = self.root / "fundus_photos"
        self.output_directory = self.root / "prepared"

        # User configuration settings
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.adequate_only = adequate_only

        # Split basis can either be "patient" or "image" - "patient" will filter
        # for patients with >= 2 images, and split based on patient ID, whilst 
        # "image" will split based on individual images, regardless of patient ID
        self.split_basis = split_basis.strip().lower()
        if self.split_basis not in ["patient", "image"]:
            raise ValueError("split_basis must be either 'patient' or 'image'")
        
        # Targets and required columns
        self.targets = target_labels

        self.require_columns = [
            "image_id",
            "patient_id",
            "quality",
            "DR_SDRG",
            "DR_ICDR",
            *self.targets
        ]

        # Declare variables that will hold data during pipeline execution
        self.df = None
        self.prevalence_table = None

    def split_data(self):
        """
        Splits the BRSET dataset into training, validation and test sets based on
        the specified ratios and split basis provided.
        """

        self.output_directory.mkdir(parents=True, exist_ok=True)

        self._load_and_validate_metadata()
        self._filter_eligible_cohort()
        self._verify_cohort_images()
        self._generate_splits()
        self._validate_split_integrity()
        self._audit_split_prevalence()
        self._save_manifests()

        print("Data splitting complete!")

    # Internal helper methods

    def _load_and_validate_metadata(self):
        pass

    def _filter_eligible_cohort(self):
        pass

    def _verify_cohort_images(self):
        pass
    
    def _generate_splits(self):
        pass

    def _validate_split_integrity(self):
        pass

    def _audit_split_prevalence(self):
        pass
    
    def _save_manifests(self):
        pass