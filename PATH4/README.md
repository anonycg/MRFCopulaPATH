# PROGRESSION AND TRANSMISSION OF HIV (PATH) 4.0 MODEL
## GENERAL
PATH 4.0 (progression and transmission of HIV) simulation model 
Developed in Netlogo 6.2.2 https://ccl.northwestern.edu/netlogo/ 
* The model is calibrated to simulate HIV in the United States for the period 2006 to 2017. The model can be used for academic and research purposes with indepth knowledge of PATH mathematical methods and modeling in Netlogo software.

## COMPUTATIONAL
* Open model in Netlogo
* Modify inputs on interface as needed. Keep defaults on interface to generate HIV representative of years 2006 to 2017 in US. See further instructions on Netlogo model interface
* To simulate model beyond 2017, changes should be made specific to analyses of interest. 
  * For analyses corresponding to changes in care continuum distributions from changes in testing and care behaviors, input files in 'data' subfolder can be modified (see ReadMe file in 'data' subfolder for instructions). This feature can also be used for updating model to calibrate to surveillance for years past 2017, by using care continuum distributions from HIV surveillance reports for each year. However, the model will assume no changes in other features such as sexual behavior (this will affect predictions over the COVID-19 era).  
  * Several other analyses can be done with indepth knowledge of the model and methods.

* To run model click 'RunExperiment' on interface. Epidemic features are output in csv files, and contains data for everymonth of simulation. Analyses can be conducted using these csv files. Alternatively, the following app can be downloaded to visualize key epidemic metrics by transmisison-group and age-group  
[Download PATHUI_App](https://people.umass.edu/chaitrag/Research/PATH-App/PATHUISetup.exe)
  * App installation instructions: after downloading app, during installation, it will ask for location of PATH model. Go to folder where above PATH4 folder was downloaded and select the .Netlogo file. The app reads in the files output from the simulation. Therefore, prior to use of the app for visualzing results, make sure the model has finished all runs. The app also has surveillance data for years 2006 to 2017 on several epidemic metrics and thus can be used for visualizing model fit for period 2006 to 2017, and visualizing results of any analyses post 2017. 

## METHODS
#### The general structure of PATH 4.0 is composed of four main modules (see related publications for mathematical methods), below are corresponding .nls files
1.  Disease progression module for simulating progression along disease and care continuum stages:     
"disease_progression.nls" "Data.nls" "testing-frequency.nls" "set-dropout-care.nls" "manage-care-continuum_juri.nls"  
(called in PATH4main.nls--> call-DiseaseProg-Demographics) 

2. Transmission module for simulating new infections:   
transmissions-ECNA.nls   
(called in PATH4main.nls --> infect-population)  

3.  HIV-ECNA network generation module for generating partnerships of newly infected nodes:  
"ECNA-generateNetwork-v2.nls" "contact-activation.nls" "dryrun.nls"   
(called in PATH4main.nls -->  determine-node-neighbors)  

4. Compartmental module for simulating susceptible persons:   
"age-functions.nls" "jurisdiction.nls" "riskGroupPreference.nls"   
(called in PATH4main.nls --> age-population-yearly)  

## CREDITS AND REFERENCES

### FUNDING 

National Institute of Allergy and Infectious Diseases of the National Institutes of Health under Award Number R01AI127236. The funding agreement ensured the authors' independence in designing the study, interpreting the data, writing, and publishing the report.

### RELATED PUBLICATIONS

Eden, M., Castonguay, R., Munkbat*, B., Balasubramanian, H., Gopalappa, C., Agent-based evolving network modeling: a new simulation method for modeling low prevalence infectious diseases, Health Care Management Science, 24, 623–639 (2021). https://doi.org/10.1007/s10729-021-09558-0 


Singh*, S., France, A. M., Chen, Y-H, Farnham, P. G., Oster, A. M., Gopalappa, C., Progression and Transmission of HIV (PATH4.0): A new agent-based evolving network simulation for modeling HIV transmission clusters; Mathematical Biosciences and Engineering, 18 (3), 2021 doi: 10.3934/mbe.2021109

Y. H. Chen, A. M. France, P. G. Farnham, S. L. Sansom, C. Gopalappa, A. Oster., Replicating HIV Transmission Clusters in a U.S. HIV Agent-Based Model[abstract]. In: Abstracts: SMDM 40th Annual Meeting; 2018 Oct; Montréal, Québec, Canada. 

### CONTACT
DISEASE MODELING LAB
Chaitra Gopalappa, PhD (chaitrag@umass.edu)
Associate Professor, Mechanical and Industrial Engineering, UMass Amherst 
Associate Professor, Commonwealth Honors College, UMass Amherst    
Lab website:   https://diseasemodeling.github.io/   
[LinkedIn](https://www.linkedin.com/in/chaitra-gopalappa-18bb181a/) | [ResearchGate](https://www.researchgate.net/profile/Chaitra_Gopalappa) |
Mastodan: @chaitra_gopalappa@home.social | Instagram: @chaitra_gopalappa 

### LICENSE
The model can be used for academic and research purposes with indepth knowledge of PATH mathematical methods and modeling in Netlogo software. Contact author for interest in collaborative work.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
