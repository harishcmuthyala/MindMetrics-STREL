# 📊 STREL Dataset - Feature Processing Guide

**Quick reference for ML model training: Which features to use and which to skip**

---

## 🎯 Quick Summary

| Total Features | ✅ Use for Training | ❌ Skip | 🎯 Target |
|----------------|---------------------|---------|-----------|
| 43 columns | 24 + 2 temporal = **26** | 16 | 1 |

---

## 📋 Feature-by-Feature Guide

### **Column 1-5: Identifiers & Time**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 1 | **Participant** | User ID (T001-T025) | ❌ **SKIP** | Only for splitting train/test. |
| 2 | **Date** | Calendar date | ❌ **SKIP** | Only for sorting/sequencing. Not predictive of stress. |
| 3 | **Time** | Timestamp | ❌ **SKIP** (extract Hour, Minute first) | Extract temporal features like Hour and Minute |
| 4 | **Day** | Type of day (WD1, CD, WD2, ND) | ✅ **USE** | Main predictor, Critical days have different stress patterns. |
| 5 | **subDC** | Internal study code | ❌ **SKIP** | Just organizational metadata. No predictive value. |

---

### **Column 6-7: Context**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 6 | **Period** | Morning or Afternoon | ✅ **USE** | Circadian rhythm affects stress. Morning vs afternoon patterns differ. |
| 7 | **Profession** | Emergency Care or Special Police | ✅ **USE** | Different professions have different stress responses. |

---

### **Column 8-10: Demographics**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 8 | **Age** | Participant age (years) | ✅ **USE** | Age affects heart rate and stress response. |
| 9 | **Gender** | Male or Female | ✅ **USE** | Gender differences in physiological stress response. |
| 10 | **BMI** | Body Mass Index | ✅ **USE** | BMI affects heart rate baseline and cardiovascular function. |

---

### **Column 11-19: Physiological Signals**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 11 | **HR** | Heart Rate (beats/min) | ✅ **USE** | Core stress indicator. Elevated HR = stress or physical activity. |
| 12 | **RR** | RR interval - heart rate variability (milli-seconds) | ✅ **USE** | Shows heart rhythm variability. |
| 13 | **Cadence** | Steps per minute | ✅ **USE** | High cadence during sedentary = micro-movements = stress indicator. |
| 14 | **Speed** | Movement speed (km/h) | ✅ **USE** | Helps validate sedentary vs active periods. |
| 15 | **PNSindex** | Parasympathetic index (relaxation) | ✅ **USE** | Measures "rest & digest" system. Positive = relaxed. |
| 16 | **SNSindex** | Sympathetic index (stress) | ✅ **USE** | Measures "fight or flight" system. >1 = stressed. |
| 17 | **Stressindex** | Baevskii Stress Index | ❌ **SKIP** | **Reason:** Highly correlated with SNSindex (r=0.91). Redundant. Using both confuses the model. |
| 18 | **HR_Baseline** | Daily personalized resting HR | ✅ **USE** | Can help normalize HR, but HR_Normalized is derived from it. Choose one. |
| 19 | **HR_Normalized** | HR minus HR_Baseline | ❌ **SKIP** | **Reason:** Derived feature (HR - HR_Baseline). Let model learn this relationship itself from HR and HR_Baseline. |

---

### **Column 20: Activity**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 20 | **Activity4** | Type of activity (Work, NonWork, Driving, etc.) | ✅ **USE**  | **Important:** Filter to keep only Work, NonWork, Driving (sedentary). Then encode. Different activities = different stress contexts. |

---

### **Column 21-24: Sleep & Physical Activity**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 21 | **Sleep_Time** | Hours slept previous night | ✅ **USE** | Poor sleep = lower stress resilience. Good predictor. |
| 22 | **PA_Percent** | % of day in physical activity | ✅ **USE** | Physical activity reduces stress on workdays. |
| 23 | **PA_Intensity** | How intense the physical activity was | ✅ **USE** | Intensity matters. Light walking vs intense running have different effects. |
| 24 | **PA_Activity** | Categorical: Sedentary or Physical | ❌ **SKIP** | **Reason:** Redundant. PA_Percent and PA_Intensity already capture this information. |

---

### **Column 25-32: Stress Labels (TARGET & METADATA)**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 25 | **NHR_Stress** | **PRIMARY TARGET** (S = Stress, NS = No Stress) | 🎯 **TARGET** | **This is what you're predicting!** Do not use as a feature. |
| 26 | **NHR_S** | % of day stressed (NHR method) | ❌ **SKIP** | Metadata about the target. |
| 27 | **NHR_NS** | % of day not stressed (NHR method) | ❌ **SKIP** | Metadata about the target. |
| 28 | **NHR_0_2SD** | Threshold value used for labeling | ❌ **SKIP** | Metadata about labeling process. Not a feature. |
| 29 | **SNS_Stress** | Alternative target (SNS method) | ⚠️ **ALTERNATIVE TARGET** | Only use if comparing methods. Don't use as feature with NHR_Stress. |
| 30 | **SNS_S** | % stressed (SNS method) | ❌ **SKIP** | Metadata. |
| 31 | **SNS_NS** | % not stressed (SNS method) | ❌ **SKIP** | Metadata. |
| 32 | **SNSindexThreshold** | Threshold (always 1) | ❌ **SKIP** | Constant value. No predictive information. |

---

### **Column 33-38: NASA-TLX Daily Workload (1-20 scale)**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 33 | **N_MD** | Mental Demand - "How mentally demanding was today?" | ✅ **USE** | Strong stress predictor. Critical days = high mental demand. |
| 34 | **N_PD** | Physical Demand - "How physically demanding?" | ✅ **USE** | Physical workload contributes to overall stress. |
| 35 | **N_TD** | Temporal Demand - "How time-pressured?" | ❌ **SKIP** | **Reason:** Highly correlated with N_MD (r=0.64). Redundant. Paper dropped it. |
| 36 | **N_P** | Performance - "How successful were you?" | ✅ **USE** | Low performance perception = higher stress. |
| 37 | **N_E** | Effort - "How hard did you work?" | ❌ **SKIP** | **Reason:** Highly correlated with N_MD (r=0.61). Redundant. Paper dropped it. |
| 38 | **N_F** | Frustration - "How frustrated/stressed?" | ❌ **SKIP** | **Reason:** Highly correlated with Profession (r=0.57). Redundant. Paper dropped it. |

---

### **Column 39-43: Big Five Personality (2-10 scale)**

| # | Feature | What It Is | ✅ Use or ❌ Skip | Why |
|---|---------|-----------|------------------|-----|
| 39 | **Extraversion** | How outgoing/sociable | ✅ **USE** | Personality affects stress response. Extraverts may handle stress differently. |
| 40 | **Agreeableness** | How friendly/cooperative | ✅ **USE** | Affects interpersonal stress in team environments. |
| 41 | **Conscientiousness** | How organized/disciplined | ❌ **SKIP** | **Reason:** Paper's AIC optimization removed it. Didn't improve model. |
| 42 | **Neuroticism** | How nervous/anxious | ✅ **USE** | **Strong predictor!** High neuroticism = higher stress susceptibility. |
| 43 | **Openness** | How curious/creative | ✅ **USE** | Affects how people perceive and handle new/stressful situations. |

---

## 🔢 Categorical Features That Need Encoding

These **5 features** are categorical and must be converted to numbers:

| Feature | Categories | Encoding |
|---------|-----------|----------|
| **Day** | WD1, CD, WD2, ND | CD=0, ND=1, WD1=2, WD2=3 |
| **Period** | Morning, Afternoon | Afternoon=0, Morning=1 |
| **Profession** | EC, SP | EC=0, SP=1 |
| **Gender** | M, F | F=0, M=1 |
| **Activity4** | Work, NonWork, Driving (after filtering) | Driving=0, NonWork=1, Work=2 |

**Target also needs encoding:**
| **NHR_Stress** | NS, S | NS=0, S=1 |

---

## 🎯 Final Feature List for Training

### ✅ **26 Features to USE:**

**Categorical (5) - encode these:**
1. Day
2. Period
3. Profession
4. Gender
5. Activity4

**Numeric (19):**
6. Age
7. BMI
8. HR
9. RR
10. Cadence
11. Speed
12. PNSindex
13. SNSindex
14. Sleep_Time
15. PA_Percent
16. PA_Intensity
17. N_MD (Mental Demand)
18. N_PD (Physical Demand)
19. N_P (Performance)
20. Extraversion
21. Agreeableness
22. Neuroticism
23. Openness
24. HR_Baseline 

**Temporal (2) - extract from Time:**
25. Hour
26. Minute

### 🎯 **Target Variable:**
- **NHR_Stress** (encode: NS=0, S=1)

---

## ❌ Why Skip These 16 Features?

| Feature | Why Skip |
|---------|----------|
| **Participant, Date, subDC** | Identifiers/metadata, not predictive |
| **Time** | Extract Hour/Minute, then drop raw timestamp |
| **HR_Normalized** | Derived from HR & HR_Baseline, let model learn |
| **Stressindex** | Correlated with SNSindex (r=0.91), redundant |
| **PA_Activity** | Redundant with PA_Percent + PA_Intensity |
| **NHR_S, NHR_NS, NHR_0_2SD** | Target metadata, might not be needed |
| **SNS_Stress, SNS_S, SNS_NS, SNSindexThreshold** | Alternative target/metadata |
| **N_TD** | Correlated with N_MD (r=0.64) |
| **N_E** | Correlated with N_MD (r=0.61) |
| **N_F** | Correlated with Profession (r=0.57) |
| **Conscientiousness** | Paper's optimization removed it (didn't help) |

---

## 📊 Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                   FEATURE SUMMARY                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ USE (26 features):                                  │
│     • 5 Categorical (need encoding)                    │
│     • 19 Numeric (ready to use)                        │
│     • 2 Temporal (extract from Time)                   │
│                                                         │
│  🎯 TARGET:                                             │
│     • NHR_Stress (encode: NS=0, S=1)                   │
│                                                         │
│  ❌ SKIP (16 features):                                 │
│     • 3 Identifiers (Participant, Date, subDC)         │
│     • 1 Raw timestamp (Time - extract then drop)       │
│     • 4 Derived/redundant (HR_Normalized, etc.)        │
│     • 8 Target metadata (NHR_S, SNS_S, etc.)           │
│                                                         │
│  ⚠️ REMEMBER:                                           │
│     • Filter to sedentary activities FIRST             │
│     • Split by Participant (to prevent leakage)           │
│     • Encode all categoricals                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Takeaways

1. **Start with 43 columns** → End with **26 features** for training

2. **Main reasons to skip features:**
   - **Identifiers** (Participant, Date, subDC)
   - **Derived features** (HR_Normalized = HR - HR_Baseline)
   - **High correlation** (Stressindex ↔ SNSindex, N_TD ↔ N_MD)
   - **Target metadata** (NHR_S, NHR_NS, etc.)
   - **Didn't help model** (Conscientiousness removed by paper)

3. **Critical pre-processing:**
   - Remove Missing Values (Especially; NHR_Stress is NA)
   - Encode 5 categorical features
   - Extract Hour, Minute from Time
   - Split by Participant (not random split!)

4. **Target:** NHR_Stress (Binary: 0 = No Stress, 1 = Stress)

---

**Ready to train!** 🚀