# *dh2loop* <br> 
[![DOI](https://zenodo.org/badge/243455612.svg)](https://zenodo.org/badge/latestdoi/243455612)

***dh2loop*** is a python library that provides the functionality to extract and standardize geologic drill hole data and export it into readily importable interval tables (collar, survey, lithology) to feed into 3D modelling packages. It addresses the subjective nature and variability of nomenclature of lithological descriptions within and across different drilling campaigns by integrating published dictionaries, glossaries and/or [thesauri](https://github.com/Loop3D/dh2loop/blob/master/thesauri.md) that were built to improve resolution of poorly defined or highly subjective use of terminology and idiosyncratic logging methods. It also classifies lithological data into multi-level groupings that can be used to systematically upscale and downscale drill hole data inputs in multiscale 3D geological model. It also provides drill hole desurveying (computes the geometry of a drillhole in three-dimensional space) and log correlation functions so that the results can be plotted in 3D and analysed against each other. The workflow behind the string matching is illustrated [here](images/fig07.png). 

![Upscaling Drillhole Data](images/drillholes2.png)
*Upscaling Drillhole Data*

**Mark Jessell** contributed the original idea, which was further developed by **Ranee Joshi** (ranee.joshi@research.uwa.edu.au). The code development is led by **Kavitha Madaiah** (kavitha.madaiah@uwa.edu.au). **Mark Lindsay** and **Guillaume Pirot** have made significant contributions to the direction of the research. 

## Where to start:
  
1. Install the dependencies:
- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy) <br>
`pip install fuzzywuzzy` <br>
- pandas (https://pandas.pydata.org/)  <br>
`pip install pandas` <br>
- psycopg2 (https://pypi.org/project/psycopg2/)  <br>
`pip install psycopg2` <br>
- numpy (https://github.com/numpy/numpy)  <br>
`pip install numpy` <br>
- nltk (https://github.com/nltk/nltk )  <br>
`pip install nltk` <br>
- pyproj (https://github.com/pyproj4/pyproj)  <br>
`pip install pyproj`  <br>
- vtk (https://vtk.org/download/)  <br>
`pip install vtk`  <br>
- ipyleaflet (https://ipyleaflet.readthedocs.io/en/latest/installation.html)  <br>
`pip install ipyleaflet`  <br>

2. Clone the repository: <br>
`$ git clone https://github.com/Loop3D/dh2loop.git`

3. Install from your local drive <br>
`cd \local_drive\` <br>
`python setup.py install --user --force`

4.Run the file Lithology_With_Comments_IDLE.py in IDLE environment , if lithology with comments need to be generated efficiently in less time.
start->search->IDLE->open the file Lithology_With_Comments_IDLE.py -> run ->check csv file having lithology with comments along with score.

5. Try out the demo jupyter notebook:
https://github.com/Loop3D/dh2loop/blob/master/notebooks/2_Exporting_and_Text_Parsing_of_Drillhole_Data_Demo.ipynb

## More information on the package
Please refer to the [preprint](https://gmd.copernicus.org/preprints/gmd-2020-391/) currently under review.

## Problems
For any bugs/feature requests/comments, please create a new [issue](https://github.com/Loop3D/dh2loop/issues). 

## Acknowledgements
*This research was carried out while in receipt of the Scholarship for International Research Fees (Australian Government Research Training Program Scholarship) and Automated 3D Geology Modelling PhD Scholarship (University Postgraduate Award) at the University of Western Australia. The work has been supported by the Mineral Exploration Cooperative Research Centre (MinEx CRC; https://minexcrc.com.au/) whose activities are funded by the Australian Government's Cooperative Research Centre Program. This work was also done with the Loop Consortium (http://loop3d.org) as part of an international effort to found a new open-source platform to build the next generation of 3D geological modelling tools.*
