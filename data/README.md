# data/

Place your NGSIM CSV file(s) here.

## Download

The Enhanced NGSIM dataset is publicly available from the U.S. Department of Transportation:

**URL:** https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

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

## If you don't have the data

Run with synthetic NGSIM-like data (IDM-inspired, same column structure):

```bash
python main.py
```

No `--data` flag needed — synthetic data is generated automatically.
