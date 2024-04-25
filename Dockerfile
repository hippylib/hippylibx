FROM dolfinx/dolfinx:v0.8.0
MAINTAINER Venugopal Ranganathan <venu22@utexas.edu>
RUN pip install memory_profiler
RUN pip install psutil
RUN pip install pylint

RUN wget -O paraview.tar.gz "https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.12&type=binary&os=Linux&downloadFile=ParaView-5.12.0-MPI-Linux-Python3.10-x86_64.tar.gz"
RUN tar xzf paraview.tar.gz && mv ParaView-5.12.0-MPI-Linux-Python3.10-x86_64 ParaView_5.1.2

ENV PATH=$PATH:/root/ParaView_5.1.2/bin

RUN echo "#! /bin/bash\npvserver --server-port=11112" > /root/ParaView_5.1.2/bin/pvserver-d && chmod +x /root/ParaView_5.1.2/bin/pvserver-d

# EXPOSE Port 11112
EXPOSE 11112

