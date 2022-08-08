# us-county-disaster-clustering

Clustering Algorithm to group US counties based on how at-risk they are to natural disasters

Part of the Data Mining class at the University of Chicago's Master of Science in Analytics program 

## Project Intro/Objective
For this project we have clustered US counties based on how at-risk they are to natural disasters. The risk calculation is partially based on FEMA's risk metric, which includes:
1. <b> Expected Annual Loss:</b> Hazard’s risk component measuring the expected loss of building value, population, and/or agriculture value each year due to natural hazards 

2. <b>Social Vulnerability Index:</b> Consequence enhancing component and analyzes demographic characteristics to measure the susceptibility of social groups to the adverse impacts of natural hazards 

3. <b>Community Resilience: </b> Consequence reduction component and uses demographic characteristics to measure a community’s ability to prepare for, adapt to, withstand, and recover from the effects of natural hazards 

<b> Risk = (Expected Annual Loss * Social Vulnerability Index) / Community Resilience </b>

These metrics currently only focus on the tangible costs of natural disasters. However, we know there are intangible costs as well. <b>Indirect effects</b> are the subsequent or secondary results of the initial destruction, like the impact or the lasting effects on mental health. Therefore, our calculation of risk factors has been properly modified to include the risk and effects on mental health that prior environmental disasters have caused on the United States population. With this information, our calculation of risk has been modified accordingly to the below: 

<b>Risk = (Expected Annual Loss * Social Vulnerability Index * Mental Health Index) / Community Resilience </b>

## Methods Used
- Clustering 

## Technologies 
- Python 

## Dataset 
The data used in this analysis have originated from multiple national agencies. The primary sources are listed below: 

- National Risk Index - Federal Emergency Management Agency (FEMA) 

	- The primary dataset is built and maintained by FEMA in close collaboration with various stakeholders and partners in academia; local, state and federal government; and private industry. 

	- Link: [Data Resources | National Risk Index (fema.gov)](https://hazards.fema.gov/nri/data-resources)

	- The dataset was the source of information for social vulnerability index, community resiliency, and expected annual loss data used to calculate “Risk” 

- FEMA Disaster Declaration Summaries - FEMA 

	- Link: [Disaster Declarations Summaries - v2 | FEMA.gov] (https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2)

- Social Vulnerability Index - Agency for Toxic Substances and Disease Registry (ASTDR)  

	- Link: [CDC/ATSDR SVI Data and Documentation Download | Place and Health | ATSDR](https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html) 

	- Used to analyze composition of what comprises the social vulnerability index used to assess demographic information that could predict the vulnerability of particular areas to environmental hazards based on demographic / social information 

- County Health Rankings and Roadmaps (CHR&R) – University of Wisconsin 

	- The CHR&R program of the University of Wisconsin publishes various measures related to the quality of life for each community including Mental health data. 

	- The Mental health data is sourced from the Behavioral Risk Factor Surveillance System (BRFSS). BRFSS is a state-based random digit dial (RDD) telephone survey that is conducted annually in all states.   

	- Link: [County Health Rankings & Roadmaps](https://www.countyhealthrankings.org/)

	- Web-scraping: The only source for the Data was the information dynamically loaded onto the webpages using JavaScript. Simple requests of the webpage  

 
## Results 

- County clusters where red indicates the highest risk due to a high enhancing factor 
![alt text](https://github.com/shroffp05/us-county-disaster-clustering/blob/main/Images/newplot.png?raw=true)

- County clusters where red indicates the highest risk due to low reduction factor 
![alt text](https://github.com/shroffp05/us-county-disaster-clustering/blob/main/Images/newplot%20(1).png?raw=true)
