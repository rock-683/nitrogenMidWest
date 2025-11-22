## Project Background
BioSand Health Institute is a mock public health non-profit located in and serving the U.S. Environmental Protection Agency (EPA) Region 7 ( Mid West- Iowa, Kansas, Missouri, Nebraska).

The organization compiles regional health and environmental data, including behaviors, socioeconomic factors, and physical environment metrics (air quality, nitrogen levels in fertilizer). This project analyzes and synthesizes this data to 1) assess the health risks of nitrogen pollution on low birthweight and 2) develop a health risk assessment, guiding more effective program resource distribution and maximizing impact.

**Problem**: There is limited understanding of how nitrogen levels relate to key health outcomes such as low birthweight in the Midwest, creating challenges for planning and targeting intervention efforts.


**Goal**: To provide a nitrogen health risk assessment of county-level risk by analyzing spatial patterns between nitrogen concentrations and low birthweight outcomes

An interactive PowerBI dashboard can be viewed [here](https://app.powerbi.com/view?r=eyJrIjoiZDNkZmRiYmQtNDViOC00NDU2LTk1ZDAtMzc1N2FkZGFiNDBiIiwidCI6IjA2MTM4ZTY4LWJmOGItNDUwYS1iMmJmLWYyNTljMjczYWYxNiJ9).  

Python code can be found at the following links:  
+ Data [sources, cleaning and preparation](source_and_carpentry.py) for analysis.   
+ Nitrogen: [Exploratory data analysis](ExploratoryDataAnalysis_Nitrogen.py) and [Geospatial autocorrelation](SpatialAutocorrelation_Nitrogen.py)(Hot spot analysis)  
+ Low Birthweight: [Exploratory data analysis](ExploratoryDataAnalysis_LB.py) and [Geospatial autocorrelation](SpatialAutocorrelation_LB.py) (Hot spot analysis)   
+ [Multiscale Geographical Weighted Regression](GWR_MGWR_LowBirthweight.py) (MGWR)

### Data Structure and Initial Checks

BioSand's finailized project database structure as seen below consists of three tables: geodata_county, county_nitrogen, and county_health. All can be joined using each county’s  Federal Information Processing Standard (FIPS) code. 

