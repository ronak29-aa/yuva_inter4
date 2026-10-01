# Week 4 Public Source & Technical Research Notes

## 1. Crop production source
Government of India Open Government Data (OGD) Platform — District-wise, season-wise crop production statistics.
Resource: https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997

The OGD resource describes district-wise, crop-wise, season-wise and year-wise crop area and production data. It is published under the Ministry of Agriculture and Farmers Welfare / Department of Agriculture and Farmers Welfare and has annual granularity. Source reviewed for dataset-selection rationale.

## 2. Rainfall source
Government of India OGD Platform — Rainfall in India.
Resource: https://www.data.gov.in/catalog/rainfall-india

The catalog provides month-wise all-India rainfall and sub-division-wise rainfall/departure information, including historical rainfall resources. It is published under the Ministry of Earth Sciences / India Meteorological Department.

## 3. Model validation source
scikit-learn documentation recommends time-aware validation for time-ordered observations because ordinary random cross-validation can allow temporal leakage. TimeSeriesSplit keeps test observations later than training observations.

## 4. Important artifact note
The numeric dataset included in this submission is a SYNTHETIC TRAINING/DEMONSTRATION DATASET. It is designed to mimic the structure and plausible ranges of an agricultural yield problem using public-source concepts. It is not presented as official government observations.

## 5. Why synthetic data is used
A reproducible local sample makes the complete Week 4 workflow executable without redistributing a large external dataset and prevents accidental misrepresentation of simulated values as official observations.
