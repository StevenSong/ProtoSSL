* for age binning, use 20 year bins [0, 20, 40, 60, 80, 300] <-- right edge 300 given variable age censoring > 90 for adults, dataset-specific quartiles otherwise (e.g. ZZU pediatric dataset)
* for stratification (by binned age, sex, labels), when there are multiple ECGs per patient, need to define patient-level variables. take the max for each class when grouping by the patient ID. the implication of this for age is that a patient may have more than one age assignment, semantically meaning that the stratifier tries to preserve the prevalence of each age bin.
* if no author provided initial train/val/test → use ptbxl stratifier to make splits stratified by binned age, sex, labels
    * ensure train/val/test have at least 1 positive instance of all labels
    * applied to CinC Georgia and ZZU datasets
* for train subsets, use ptbxl stratifier to make nested subsets stratified by binned age, sex, labels
    * ensure each subset has at least 1 positive instance of all labels

|dataset |defines splits|has patient IDs|
|--------|--------------|---------------|
|echonext|train/val/test|yes            |
|ptbxl   |folds         |yes            |
|mimic   |train/val/test|yes            |
|cinc    |no            |no             |
|zzu     |no            |yes            |
