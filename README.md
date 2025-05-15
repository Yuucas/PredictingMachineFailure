# PredictingMachineFailure
The core task is to predict when a machine is likely to fail, given its operational data.

This synthetic dataset is modeled after an existing milling machine and consists of 10 000 data points from a stored as rows with 9 features in columns

* **UID:** unique identifier ranging from 1 to 10000
* **product ID:** consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
* **type:** just the product type L, M or H from column 2
* **air temperature [K]:** generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
* **process temperature [K]:** generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
* **rotational speed [rpm]:** calculated from a power of 2860 W, overlaid with a normally distributed noise
* **torque [Nm]:** torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values.
* **tool wear [min]:** The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
a 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.