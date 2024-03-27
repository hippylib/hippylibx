FROM  dolfinx/dolfinx:stable
MAINTAINER Venugopal Ranganathan <venu22@utexas.edu>
RUN pip install memory_profiler
RUN pip install psutil
RUN pip install pylint