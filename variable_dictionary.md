# Variable Dictionary

Quick reference for what all the variables mean.

## Outcome

| Variable | What it is |
|----------|------------|
| `LOS` | Length of stay in days |

## Patient Variables

| Variable | What it is | Values |
|----------|------------|--------|
| `AGE` | Patient age | Years |
| `FEMALE` | Sex | 0=Male, 1=Female |
| `RACE` | Race/ethnicity | 1=White, 2=Black, 3=Hispanic, 4=Asian/Pacific Islander, 5=Native American, 6=Other |
| `ZIPINC_QRTL` | Income quartile based on zip code | 1-4 (1=poorest, 4=richest) |
| `I10_NDX` | Number of diagnoses | Count |
| `I10_NPR` | Number of procedures | Count |
| `APRDRG_Severity` | How sick the patient is | 1=Minor, 2=Moderate, 3=Major, 4=Extreme |
| `APRDRG_Risk_Mortality` | Risk of dying | 1=Minor, 2=Moderate, 3=Major, 4=Extreme |
| `has_comorbidities` | Has other chronic conditions | 0=No, 1=Yes |
| `I10_INJURY` | Injury diagnosis | 0=None, 1=Primary, 2=Secondary |
| `ELECTIVE` | Planned admission | 0=Emergency/urgent, 1=Elective |
| `PAY1` | Insurance | 1=Medicare, 2=Medicaid, 3=Private, 4=Self-pay, 5=No charge, 6=Other |
| `DIED` | Died in hospital | 0=No, 1=Yes |

## Hospital Variables

| Variable | What it is | Values |
|----------|------------|--------|
| `HOSP_NIS` | Hospital ID | Unique identifier |
| `HOSP_BEDSIZE` | Hospital size | 1=Small, 2=Medium, 3=Large |
| `HOSP_LOCTEACH` | Location and teaching status | 1=Rural, 2=Urban non-teaching, 3=Urban teaching |
| `H_CONTRL` | Who owns the hospital | 1=Government, 2=Private non-profit, 3=Private for-profit |
| `HOSP_REGION` | US region | 1=Northeast, 2=Midwest, 3=South, 4=West |

## Data Source

National Inpatient Sample (NIS), 2019  
Available from: https://www.hcup-us.ahrq.gov/nisoverview.jsp

Note: You need a data use agreement to access NIS data.
