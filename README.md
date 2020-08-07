# Parsing Atari game frames from ALE
Dependencies: cv2; numpy; math

Parses input RGB image from ALE to state dictionary, using predefined object groups.

Currently includes parsing functions for Freeway and Asterix in parse.py. <br>
Use function in visualize.py to see parsed state.
    
-

- Freeway: 
    - Groups: agent; car; dest(ination)
    - Templates needed: freeway_center_line.png
    
- Asterix: 
    - Groups: agent; target (rewarding objects); demon (punishing objects)
    - templates needed: everything in asterix_templ/
    
See docstring for details.
