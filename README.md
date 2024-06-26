# FKS_AtrialModel_Ferrer
This repository contains an example of the fibrotic kernel signature (FKS) for the atrial model used in Ferrer, et al. (2015). 

- ```xdmf/``` contains the data from the 3D atrial model adapted to XDMF format. VTK files were retrieved from https://www.uv.es/commlab/blog-details-3DVENTRICULARMODEL.html. Here ```cellTags.xdmf``` has the individual tags for each cell of the mesh and ```AtrialVoxHexa.xdmf``` contains the organID and Fibers fields.

- ```biblio/``` contains publications related to the Ferrer model and setting for the conductivity parameters.

- ```FKS_FerrerAtria.py``` computes the FKS for the Ferrer model and generates an XDMF file of the signature evaluated at 50 time steps with a logarthmic progression in time.
