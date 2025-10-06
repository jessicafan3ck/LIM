## Events Data Card
  #### Source: 
  FIFA Unified Event Data (U17 Women’s World Cup)
  #### Granularity: 
  Every on-ball event, frame-aligned to match clock
  #### Primary Keys: 
  match_id, event_id

#### Core Identifiers
  match_id: Unique FIFA identifier for the match
  
  event_id: Unique sequential identifier for the event
  
  team_id: Internal FIFA identifier for team
  
  team_name: Team name (string)
  
  from_player_id / from_player_name / from_player_shirt_number: Originating player metadata
  
  to_player_name / to_player_shirt_number: Receiving player metadata, if applicable
  
  player_seq_id: Order of players within roster
  
#### Timing
  match_run_time_in_ms: Elapsed ms since kickoff
  
  match_run_time: Clock-style string (hh:mm:ss)
  
  match_time_in_ms: Timestamp aligned to match clock
  
#### Event Structure
  event_order: Event sequence number within possession/sequence
  
  relevant_event_id: Links to associated events (e.g., passes, duels)
  
  event_type: Encoded type (pass, shot, duel, foul, etc.)
  
  outcome: Encoded success/failure flag (when applicable)
  
#### Spatial Information
  x_location_start, y_location_start: Normalized (0–1) pitch coordinates of event origin
  
  x_location_end, y_location_end: Normalized (0–1) pitch coordinates of event completion
  
  x/y_location_start_mirrored, x/y_location_end_mirrored: Coordinate flips for attacking direction normalization
  
#### Other
  version: Data feed version
  
  referees rows included for officiating events
  
Additional FIFA-enriched metadata fields (pressure, duel type, etc.) may be present depending on feed version
