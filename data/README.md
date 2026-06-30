
## Download

The Enhanced NGSIM dataset is publicly available from the U.S. Department of Transportation:

**URL:** https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

**For the researcher:** Preprocess your NGSIM files locally into this format and save the result as data/ngsim_30_vehicles.csv before running main.py. This file is gitignored and will not be pushed to GitHub.

**For reproducibility:** Download the raw NGSIM trajectories from the link above, select 30 vehicles, extract the six columns below at 0.1 s resolution, and save as a CSV with these exact column names.

**Dataset name:** Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data  
**Collection locations:**
- Southbound US-101 and Lankershim Boulevard, Los Angeles, CA
- Eastbound I-80, Emeryville, CA

**Time resolution:** 0.1 seconds (10 Hz)  
**Total vehicles in database:** 1,000  
**Vehicles used in this study:** 30 (randomly selected)

## Required CSV format

After downloading and preprocessing, save your CSV here with these exact column names:

| Column | Type | Description |
|---|---|---|
| `vehicle_id` | int | Integer vehicle identifier |
| `vehicle_index` | int | Timestep index within this vehicle (0-based, ascending) |
| `velocity` | float | Follower vehicle velocity (m/s) |
| `delta_x` | float | Positional gap between follower and leader (m) |
| `delta_v` | float | Speed difference: follower − leader (m/s) |
| `acceleration` | float | Current acceleration of follower vehicle (m/s²) |

