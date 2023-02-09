This folder contains examples running OpenOrb and JPL horizons. Also, work to complement the research done on minimoons, given the integration results from Dr. Fedorets.


# Minimoon_Trajectories.py

This is a stand-alone file used to create the orbits of the two none minimoons at the time, **2006 RH<sub>120</sub>** and **2020 CD<sub>3</sub>**. The file makes use of astropy (astroquery) to access JPL Horizons integrator. Then the OpenOrb simulator uses the initial osculating orbital elements from JPL Horizons (input directly) and integrates the two similar orbits.
