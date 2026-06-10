import pandas as pd
import argparse


CANCER_TYPES = {
    "luad": "lung adenocarcinoma",
    "lscc": "lung squamous cell carcinoma",
    "brca": "breast cancer",
    "ccrcc": "clear cell renal cell carcinoma",
    "hnscc": "head and neck squamous cell carcinoma",
    "ucec": "uterine endometrial carcinoma",
    "gbm": "glioblastoma",
    "pdac": "pancreatic ductal adenocarcinoma",
    "ov": "ovarian cancer",
}

CLINICAL_COLUMNS = [
    "age",
    "sex",
    "ethnicity_race_ancestry_identified",
    "tumor_stage_pathological",
    "histologic_type",
    "histologic_grade",
    "tumor_size_cm",
    "tumor_laterality",
    "tumor_site",
    "tumor_focality",
    "tobacco_smoking_history",
    "alcohol_consumption",
    "bmi",
]


SKIP_VALUES = [
    "Unknown",
    "not specified",
    "not identified",
    "Not Applicable",
    "Not Evaluated: Not provided or available",
    "Not Available",
    "GX Grading is not applicable, cannot be assessed or not specified",
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
    "Smoking history not available",
    "Alcohol consumption history not available",
    "Patient declined to state."
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


def clinical_to_text(row, cancer_type):
    parts = []
    # age and sex
    age = row.get("age")
    sex = row.get("sex")
    if is_usable(age) and is_usable(sex):
        try:
            parts.append(f"{int(float(age))}-year-old {str(sex).lower()}")
        except ValueError:
            parts.append(f"{str(age)}-year-old {str(sex).lower()}")
    elif is_usable(age):
        try:
            parts.append(f"{int(float(age))}-year-old")
        except ValueError:
            parts.append(f"{str(age)}-year-old")


    # cancer type
    parts.append(f"{cancer_type} patient")

    # ethnicity
    value = row.get("ethnicity_race_ancestry_identified")
    if is_usable(value):
        parts.append(f"{value} ethnicity")

    # tumort stage
    value = row.get("tumor_stage_pathological")
    if is_usable(value):
        parts.append(value)

    # histologic type
    value = row.get("histologic_type")
    if is_usable(value):
        parts.append(f"with {value} histology")
    
    # histologic grade
    value = row.get("histologic_grade")
    if is_usable(value):
        parts.append(f"and histologic grade {value}")

    # tumor size
    value = row.get("tumor_size_cm")
    if is_usable(value):
        parts.append(f"tumor size {value} cm")

    # tumor laterality
    value = row.get("tumor_laterality")
    if is_usable(value):
        parts.append(f"{value.lower()} lung")

    # tumor site
    value = row.get("tumor_site")
    if is_usable(value):
        parts.append(value.lower())

    # tumor_focality
    value = row.get("tumor_focality")
    if is_usable(value):
        parts.append(value.lower())

    # smoking history
    SMOKING_DICT = {
        "Lifelong non-smoker: Less than 100 cigarettes smoked in lifetime": "lifelong non-smoker",
        "Current smoker: Includes daily and non-daily smokers": "current smoker",
        "Current reformed smoker, more than 15 years": "former smoker (>15 years)",
        "Current reformed smoker within past 15 years": "former smoker (<15 years)",
    }
    value = row.get("tobacco_smoking_history")
    if is_usable(value):
        parts.append(SMOKING_DICT.get(value, value))

    # alcohol consumption
    ALCOHOL_MAP = {
        "Lifelong non-drinker": "lifelong non-drinker",
        "Alcohol consumption equal to or less than 2 drinks per day for men and 1 drink or less per day for women": "moderate drinker",
        "Alcohol consumption more than 2 drinks per day for men and more than 1 drink per day for women": "heavy drinker",
        "Consumed alcohol in the past, but currently a non-drinker": "former drinker",
    }
    value = row.get("alcohol_consumption")
    if is_usable(value):
        parts.append(ALCOHOL_MAP.get(value, value))

    # bmi
    value = row.get("bmi")
    if is_usable(value):
        bmi = float(value)
        if bmi < 18.5:
            bmi_cat = "underweight"
        elif bmi < 25:
            bmi_cat = "normal weight"
        elif bmi < 30:
            bmi_cat = "overweight"
        else:
            bmi_cat = "obese"

        parts.append(f"{bmi_cat} (BMI {value})")
    
    # histologic grade
    value = row.get("histologic_grade")
    if is_usable(value):
        parts.append(f"and histologic grade {value}")
    
    # lymph node positivity
    value = row.get("number_of_lymph_nodes_positive_for_tumor_by_he_staining")
    if is_usable(value):
        try:
            n = int(float(value))
        except ValueError:
            pass
        else:
            if n == 0:
                parts.append("lymph node negative")
            else:
                parts.append(f"{n} lymph node{'s' if n > 1 else ''} positive")

    # tumor size in cm
    value = row.get("tumor_size_cm")
    if is_usable(value):
        parts.append(f"tumor size {value} cm")
    
    # tumor necrosis
    value = row.get("tumor_necrosis")
    if is_usable(value) and value == "Present":
        parts.append("tumor necrosis present")

    # margin status
    value = row.get("margin_status")
    if is_usable(value):
        if "uninvolved" in str(value):
            parts.append("clear surgical margins")
        elif "involved" in str(value):
            parts.append("positive surgical margins")



    return ", ".join(parts) + "."


# Apply clinical_to_text to every patient row and return {patient_id: description}
def generate_descriptions(clinical_df, cancer_type):
    descriptions = {}

    for idx, row in clinical_df.iterrows():
        text = clinical_to_text(row, cancer_type)
        descriptions[idx] = text

    return descriptions



# Save {patient_id: description} dict as a CSV with patient ID as index
def save_descriptions(descriptions, output_path):
    pd.Series(descriptions).to_csv(output_path, header=["description"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()

    cancer_type = CANCER_TYPES[args.cohort]
    clinical_df = pd.read_csv(f"data/{args.cohort}/clinical.csv", index_col=0)
    descriptions = generate_descriptions(clinical_df, cancer_type)
    save_descriptions(descriptions, f"data/{args.cohort}/clinical_descriptions.csv")
    print(f"Saved {len(descriptions)} descriptions to data/{args.cohort}/descriptions.csv")
