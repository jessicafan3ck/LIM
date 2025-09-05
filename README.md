# LIM MVP (Living Influence Model) - README
Please view 'LIM_Theory_Paper.pdf' and 'LIM_Simulation_Analytics.pdf' for insights on the theory and framework behind this proposed model.
A minimal, events-only implementation of the Living Influence Model (LIM) suitable for a coach-facing demo using FIFA Unified Event Data and a per‑player Physical Summary for the U17 Women's World Cup. Any quantities not present in the dataset are synthetically instantiated via stable, interpretable defaults so the pipeline is fully runnable end‑to‑end without tracking data. The logic and generation process for all synthetic data will be provided within this repository. The model in this repository is ran on 31 matches from the U17 Women's World Cup. 

## LIM Data
### Base Data For FIFA U17 Women World Cup

Starting data from the U17 Women's World Cup are sorted by 2 data types: events from matches and physical summaries of players in each match. These are separated in folders 'Events' and 'Phys_Summary' in this repository. This data is the private property of FIFA and may only be redistributed by their sole delegation. Do not redistribute this data nor use it for any purpose other than that which has been permitted. 
Summaries for the data contents and interpretations for each data type will be contained in a txt file in each data folder. Look for 'Events_DataCard.txt' and 'Phys_Summary_DataCard.txt'.

### Synthetic Data

Synthetic data was generated for each of these variables:

The generated data as well as the generating pipeline for each of these variables is contained in the 'Synthetic' folder. Read 'Synthetic_DataCard.txt' for a more granular description of each generating file and variable type, or read the comments in each code file to understand the generation process.

    
