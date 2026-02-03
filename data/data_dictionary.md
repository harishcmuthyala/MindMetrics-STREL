# 📖 STREL Data Dictionary

Every column in `data/raw/STREL_raw.csv` is documented below.  
Groups mirror the feature categories in `config/config.yaml` so you can
cross-reference quickly.

---

## 1. Identifiers & Time

| Column        | Type      | Example                | Description                                                                 |
|---------------|-----------|------------------------|-----------------------------------------------------------------------------|
| Participant   | string    | `T001`                 | Unique anonymised participant ID. Range: T001–T025 (T013 missing).         |
| Date          | date      | `2023-10-24`           | Calendar date of the observation.                                           |
| Time          | datetime  | `2023-10-24 06:30:00`  | Timestamp of the 5-minute interval start.                                   |
| subDC         | string    | `DC1.1`                | Sub-division of the day coding used internally in the study.                |

---

## 2. Categorical Context

| Column       | Type   | Levels                          | Description                                                                          |
|--------------|--------|---------------------------------|--------------------------------------------------------------------------------------|
| Day          | string | `WD1`, `CD`, `WD2`, `ND`       | Type of day. WD1/WD2 = standard workdays, CD = critical day, ND = non-workday.       |
| Period       | string | `Morning`, `Afternoon`          | Half of the day (split at 12:00). Reflects circadian variation.                      |
| Profession   | string | `SP`, `EC`                      | SP = Special Police, EC = Emergency Care.                                            |
| Activity4    | string | `Work`, `NonWork`, `Driving`, `Walking`, `Running`, `Biking`, `Sleeping` | Activity label assigned via GPS + cadence + debriefings. |

---

## 3. Demographics

| Column | Type    | Unit | Example | Description                                                          |
|--------|---------|------|---------|----------------------------------------------------------------------|
| Age    | integer | yrs  | `41`    | Participant age at time of study.                                    |
| Gender | string  | —    | `M`     | `M` = Male, `F` = Female.                                            |
| BMI    | float   | kg/m²| `24.4`  | Body Mass Index. High BMI can confound cardiac recordings (see paper §2.1). |

---

## 4. Physiological Signals  *(5-min interval means)*

| Column          | Type  | Unit  | Example    | Description                                                                                       |
|-----------------|-------|-------|------------|---------------------------------------------------------------------------------------------------|
| HR              | float | BPM   | `77.43`    | Heart Rate — mean over the 5-min window.                                                          |
| RR              | float | ms    | `0.784`    | RR interval (beat-to-beat). Inverse of HR; used by Kubios to derive HRV indices.                  |
| HR_Baseline     | float | BPM   | `52`       | Daily resting baseline HR for this participant (personalised reference).                          |
| HR_Normalized   | float | BPM   | `25.43`    | `HR − HR_Baseline`. The raw NHR signal before thresholding.                                       |
| PNSindex        | float | —     | `-0.88`    | Parasympathetic Nervous System index (Kubios). Negative values → sympathetic dominance.           |
| SNSindex        | float | —     | `0.476`    | Sympathetic Nervous System index (Kubios). Used as benchmark stress labelling method (threshold = 1). |
| Stressindex     | float | —     | `6.76`     | Baevskii Stress Index (Kubios), computed from HRV. Highly correlated with SNSindex (r = 0.91).    |

---

## 5. Motion Signals  *(5-min interval means)*

| Column  | Type  | Unit  | Example  | Description                                                                           |
|---------|-------|-------|----------|---------------------------------------------------------------------------------------|
| Cadence | float | CPM   | `7.52`   | Steps per minute. Key separator: sedentary ≈ 0–10 CPM, physical activity ≈ 30–85 CPM.|
| Speed   | float | km/h  | `1.34`   | Movement speed. Helps distinguish driving (high speed, low cadence) from running.     |

---

## 6. Physical Activity Summary  *(daily aggregates)*

| Column       | Type  | Unit | Example | Description                                                                      |
|--------------|-------|------|---------|----------------------------------------------------------------------------------|
| PA_Percent   | float | %    | `13.22` | Percentage of the day spent in physical activity.                                |
| PA_Intensity | float | —    | `40.3`  | Intensity score of physical activity that day (higher = more intense).           |
| PA_Activity  | string| —    | `Sedentary` | Dominant activity classification for the interval (`Sedentary` or `Physical`). |

---

## 7. Sleep

| Column     | Type  | Unit | Example | Description                                                                             |
|------------|-------|------|---------|-----------------------------------------------------------------------------------------|
| Sleep_Time | float | hrs  | `7`     | Total hours slept the previous night. Reconciled from watch analytics + self-report.    |

---

## 8. Stress Labels  ← **these are the targets & their components**

| Column              | Type   | Levels / Unit | Example | Description                                                                                                       |
|---------------------|--------|---------------|---------|-------------------------------------------------------------------------------------------------------------------|
| **NHR_Stress**      | string | `S`, `NS`     | `S`     | **Primary target.** Stressed if HR_Normalized > 2 SD above 0 during sedentary intervals. Paper's recommended method. |
| NHR_S               | float  | %             | `59.39` | Percentage of sedentary time labelled Stressed by the NHR method that day.                                        |
| NHR_NS              | float  | %             | `40.61` | Percentage of sedentary time labelled Not-Stressed by the NHR method that day.                                    |
| NHR_0_2SD           | float  | BPM           | `19.1`  | The 2·SD threshold value used for that participant on that day.                                                   |
| **SNS_Stress**      | string | `S`, `NS`     | `NS`    | **Benchmark target.** Stressed if SNSindex > 1. Used for model comparison only.                                   |
| SNS_S               | float  | %             | `20.81` | Percentage of sedentary time labelled Stressed by the SNS method that day.                                        |
| SNS_NS              | float  | %             | `72.59` | Percentage of sedentary time labelled Not-Stressed by the SNS method that day.                                    |
| SNSindexThreshold   | float  | —             | `1`     | Fixed threshold applied to SNSindex (always 1 per the paper).                                                     |

---

## 9. NASA-TLX  *(daily scores, scale 1–20)*

| Column | Type    | Scale  | Example | Description                                                                 |
|--------|---------|--------|---------|-----------------------------------------------------------------------------|
| N_MD   | integer | 1–20   | `3`     | **Mental Demand** — perceived cognitive load that day. Key stress predictor. |
| N_PD   | integer | 1–20   | `14`    | **Physical Demand** — perceived physical exertion.                           |
| N_TD   | integer | 1–20   | `4`     | **Temporal Demand** — perceived time pressure. *(dropped: correlated with N_MD)* |
| N_P    | integer | 1–20   | `10`    | **Performance** — perceived success in daily tasks.                          |
| N_E    | integer | 1–20   | `4`     | **Effort** — perceived work expended. *(dropped: correlated with N_MD)*      |
| N_F    | integer | 1–20   | `18`    | **Frustration** — perceived irritation. *(dropped: correlated with Profession)* |

> 💡 The paper's logistic regression dropped N_TD, N_E, and N_F due to collinearity. Consider doing the same.

---

## 10. Big Five Personality  *(trait scores, collected once per participant)*

| Column          | Type    | Scale | Example | Description                                                         |
|-----------------|---------|-------|---------|---------------------------------------------------------------------|
| Extraversion    | integer | 2–10  | `6`     | Outgoing / sociable tendency.                                       |
| Agreeableness   | integer | 2–10  | `7`     | Friendliness / cooperativeness.                                     |
| Conscientiousness | integer | 2–10 | `9`     | Organisation / discipline. High in this cohort (mean ≈ 8.0).       |
| Neuroticism     | integer | 2–10  | `4`     | Nervousness / emotional instability. Low in this cohort (mean ≈ 4.3). |
| Openness        | integer | 2–10  | `8`     | Curiosity / creativity.                                             |

---

## Quick-reference: Column Count

| Group                      | Columns |
|----------------------------|---------|
| Identifiers & Time         | 4       |
| Categorical Context        | 4       |
| Demographics               | 3       |
| Physiological Signals      | 7       |
| Motion Signals             | 2       |
| Physical Activity Summary  | 3       |
| Sleep                      | 1       |
| Stress Labels              | 8       |
| NASA-TLX                   | 6       |
| Big Five Personality       | 5       |
| **Total**                  | **43**  |

---

## Notes
- All physiological and motion values are **5-minute interval means**, consistent with cardiology society guidelines cited in the paper.
- Stress labels were assigned **only during sedentary intervals** to isolate cognitive/emotional stress from physical exertion.
- The dataset is available on GitHub: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- R code from the original study: https://github.com/UH-ACDC/STREL/
- Original paper OSF repository: https://osf.io/qshv7/