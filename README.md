## mlgpp - Combining flux and hyperspectral data for GPP estimation with extreme gradient boosting

### Description
------------
This project contains files for reproducing results from the paper "Comparison of data-driven models for ecosystem GPP estimation
based on mid-resolution hyperspectral data"

For preprocessing follow instructions in prisma\_gpp\_preproc\_main.py
For statistical analysis use prisma\_gpp\_stats\_main.py

Please note that the code has been designed with Python 3.9.3 and R 4.2.2 on Ubuntu 20.04.2 LTS. Error handling exists but is not exhaustive.

### Requirements
------------

Python scripts require the following modules:

 * fmch (not available on conda/pypi, use included version, i.e. add folder structure to your Python PATH or add the functions and class in hsicos.py manually to your Python environment.)
 * numpy
 * pandas
 * geopandas
 * shapely
 * pyproj
 * rasterio
 * fiona
 * h5py
 * spectral
 * xarray
 * matplotlib
 * mpl_toolkits
 * pathlib
 * datetime
 * tqdm
 * pyeto
 * cdsapi
 * hda
recommended:
 * HSI2RGB (https://github.com/JakobSig/HSI2RGB)

R scripts require the following packages:

 * tidyverse
 * tidync
 * ggthemes
 * data.table
 * mlr3verse
 * mlr3spatiotempcv
 * mlr3pipelines
 * paradox
 * visNetwork
 * DALEX
 * DALEXtra
 * iml
 * SHAPforxgboost
 * ks
 * zoo
 * sf
 * rhdf5
 * gridExtra 
 * caret
 * lubridate
 * future
 * RColorBrewer
 * docstring
 * rstudioapi (for current file location)

### License for Flux footprint modeling software (ffp.py)
------------
Copyright (c) 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, Natascha Kljun

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 

### License for all other included Python & R scripts and modules: BSD 3-Clause
------------
Copyright (c) 2023, Floris Hermanns

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

