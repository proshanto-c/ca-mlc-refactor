import os
from pathlib import Path
import pandas as pd
from PIL import Image

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

        self.required_columns = [
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
        self.df = pd.read_csv(self.csv_path)

        # Check for missing required columns
        missing_columns = sorted(
            set(self.required_columns) - set(self.df.columns)
        )
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        #  Check for missing values in required columns
        if self.df["image_id"].isna().any():
            raise ValueError("Missing values found in 'image_id' column.")
        if self.df["patient_id"].isna().any():
            raise ValueError("Missing values found in 'patient_id' column.")
        if self.df["image_id"].duplicated().any():
            raise ValueError("Duplicate values found in 'image_id' column.")
        
        # Strip ID and quality columns
        self.df["image_id"] = self.df["image_id"].astype(str).str.strip()
        self.df["patient_id"] = self.df["patient_id"].astype(str).str.strip()
        self.df["quality"] = self.df["quality"].astype(str).str.strip().str.lower()

        # Check that all target columns contain binary values (0 or 1) and convert to int8
        for target in self.targets:
            values = pd.to_numeric(self.df[target], errors='coerce')

            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError(f"Invalid values found in target column '{target}'. Expected binary values (0 or 1).")

            self.df[target] = values.astype("int8")
        
        # Check the DR Scale columns for valid integer values (0-4) and convert to int8
        for col in ["DR_SDRG", "DR_ICDR"]:
            values = pd.to_numeric(self.df[col], errors='coerce')

            if values.isna().any() or not values.isin([0, 1, 2, 3, 4]).all():
                raise ValueError(f"Invalid values found in column '{col}'. Expected integer values (0-4).")

            self.df[col] = values.astype("int8")
        
        self.df["image_name"] = self.df["image_id"].apply(lambda x: f"{x}.jpg")
        self.df["image_path"] = self.df["image_name"].apply(lambda x: self.image_directory / x)

    # Filter the dataset to only include eligible images based on user-specified criteria
    def _filter_eligible_cohort(self):

        # Filter out any images with conflicting DR labels between scaling and diabetic_retinopathy target
        self.df["dr_from_sdrg"] = self.df["DR_SDRG"].apply(lambda x: 1 if x >= 1 else 0)
        self.df["dr_from_icdr"] = self.df["DR_ICDR"].apply(lambda x: 1 if x >= 1 else 0)

        self.df["dr_disagreement"] = (self.df["dr_from_sdrg"].ne(self.df["diabetic_retinopathy"]) | self.df["dr_from_icdr"].ne(self.df["diabetic_retinopathy"]))
        # Remove images with disagreement in DR labels
        self.df = self.df[~self.df["dr_disagreement"]].copy()

        #  Filter for only adequate quality images if specified
        if self.adequate_only:
            self.df = self.df[self.df["quality"] == "adequate"]
        
        # Filter for patients with at least 2 images if splitting by patient
        if self.split_basis == "patient":
            patient_counts = self.df.groupby("patient_id").size()
            eligible_patients = patient_counts[patient_counts >= 2].index
            self.df = self.df[self.df["patient_id"].isin(eligible_patients)]
            # Keep only the first two images per patient for splitting purposes
            self.df = self.df.groupby("patient_id").head(2).reset_index(drop=True)
        
        print(f"Eligible cohort size after user filters: {len(self.df)} images from {self.df['patient_id'].nunique()} patients.")

    # Verify that every remaining image in the cohort exists and is decodable
    def _verify_cohort_images(self):
        for image_path in self.df["image_path"]:
            # Check for existence
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check for decodability
            with Image.open(image_path) as image:
                image.verify()
    
    def _generate_splits(self):
        pass

    def _validate_split_integrity(self):
        pass

    def _audit_split_prevalence(self):
        pass
    
    def _save_manifests(self):
        pass