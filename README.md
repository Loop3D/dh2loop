# dh2loop

dh2loop is a python library that provides the functionality to extract and standardize geologic drill hole data and export it into readily importable interval tables (collar, survey, lithology) to feed into 3D modelling packages. **Mark Jessell** contributed the original idea, which was further developed by **Ranee Joshi** (ranee.joshi@research.uwa.edu.au). The code development is lead by **Kavitha Madaiah** (kavitha.madaiah@uwa.edu.au). **Mark Lindsay** and **Guillaume Pirot** have made significant contributions to the direction of the research. 

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

2. Clone the repository
`$ git clone https://github.com/Loop3D/dh2loop.git`

3. Install from your local drive
`cd \local_drive\`
`python setup.py install --user --force`

4. Try out the demo jupyter notebook:
https://github.com/Loop3D/dh2loop/blob/master/notebooks/2_Exporting_and_Text_Parsing_of_Drillhole_Data_Demo.ipynb


## Problems
Any bugs/feature requests/comments please create a new [issue](https://github.com/Loop3D/dh2loop/issues). 

## Acknowledgements
*The research was supported in receipt of Scholarship for International Research Fees (Australian Government Research Training Program Scholarship) and Automated 3D Geology Modelling PhD Scholarship (University Postgraduate Award) at the University of Western Australia. The work has been supported by the Mineral Exploration Cooperative Research Centre (MinEx CRC; https://minexcrc.com.au/) whose activities are funded by the Australian Government's Cooperative Research Centre Program. This work was also done with the Loop Consortium (http://loop3d.org) as part of an international effort to found a new open-source platform to build the next generation of 3D geological modelling tools.*
