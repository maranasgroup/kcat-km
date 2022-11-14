# kcat-km
Predictive models for estimating kcat and km of enzymes from sequence/structure information

# eci-conf-2022
Models to reproduce results for eci conf 2022 poster

# AICHE-2022
Models for kcat model presented at AICHE 2022 annual meeting

# Requirements (all found in pip)
- tensorflow 2 above
- rdkit-pypi
- pandas
- datetime
- tqdm
- numpy

To perform prediction on sample inputs (input.csv) run,

`python predict.py input.csv output_km.csv km_model/ KM`
`python predict.py input.csv output_kcat.csv kcat_model/ KCAT`

