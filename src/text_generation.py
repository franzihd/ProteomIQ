import pandas as pd

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
        parts.append(f"{int(age)}-year-old {str(sex).lower()}")
    elif is_usable(age):
        parts.append(f"{int(age)}-year-old")

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



    return ", ".join(parts) + "."

