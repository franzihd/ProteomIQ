import pandas as pd

SKIP_VALUES = [
    "Unknown",
    "not specified",
    "not identified",
    "Not Applicable",
    "Not Evaluated: Not provided or available",
    "Not Available",
    "GX Grading is not applicable, cannot be assessed or not specified"
    "Not Reported",
    "Not Specified",
    "Not documented",
    "Not known",
    "Not otherwise specified",
    "Not applicable",
    "Unspecified",
    "None specified",
    "Unknown: Could not be determined or unsure",
    "Unknown Tumor Status",
    "Staging Incomplete",
    "Smoking history not available",
    "Alcohol consumption history not available",
    "Exposure to secondhand smoke history not available",
    "IHC staining not done",
    "Cannot be assessed",
    "Presence of metastatic disease cannot be assessed",
    "Other: Not Collected",
    "Other: Unknown",
    "Other: unknown",
    "Other: Candidate does not know",
    "Other: TSS did not collect info",
    "Other: TSS did not collect this info",
    "Other: TSS did not collect this information",
    "Other: information not collected at TSS",
    "nan",
    "Smoking history not available"
]


def is_usable(value):
    if pd.isna(value):
        return False
    s = str(value).strip()
    if s in SKIP_VALUES:
        return False
    if s.startswith("Other:"):
        return False
    return True