
import pandas as pd
import json
import re
import os

#AV的system message
AV_msg = """"

You are ChatGPT, a large language model trained by OpenAI. You are now acting as a mature driving assistant. And your task is to make the decision that assures the safety.

You have access to the same information as a real human driver, which includes the following:
1. Description of the environment
2. Your own vehicle's position, dynamic information.
3. Vehicles in your surroundings, including their position, dynamic information.

This scenario takes place at an expressway with exits. The positive x-coordinate points east, while the positive y-coordinate points south. The top-left corner is the origin (0,0). The velocity components follow the same convention: a negative vx indicates movement toward the west, while a negative vy indicates movement toward the north.

Your available actions include:
1. IDLE: Remain in the current lane with the current speed (Action ID: 1)
2. Turn Left: Change to the lane on the left of the current lane (Action ID: 2)
3. Turn Right: Change to the lane on the right of the current lane (Action ID: 3)
4. Acceleration: Increase vehicle speed (Action ID: 4)
5. Deceleration: Reduce vehicle speed (Action ID: 5)

A clear action ID must be chosen at the end of your reasoning process.
"""


def generate_AV(json_path, frame_step=1, start_frame=None, end_frame=None):
    """
    Generate paragraph-style descriptions using 'Time_Index' from a flat JSON file
    containing lead and following vehicle information.
    """
    if not os.path.exists(json_path):
        return f"Error: File not found: {json_path}"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    if start_frame is not None:
        df = df[df["Time_Index"] >= start_frame]
    if end_frame is not None:
        df = df[df["Time_Index"] <= end_frame]

    frames = sorted(df["Time_Index"].unique())[::frame_step]
    descriptions = []

    for time_index in frames:
        row = df[df["Time_Index"] == time_index].iloc[0]

        paragraph = f"""
At time index {time_index:.1f}, the lead vehicle is driving at {row['Speed_LV']:.2f} m/s with an acceleration of {row['Acc_LV']:.2f} m/s². Its position is {row['Pos_LV']:.2f} meters.
The following vehicle is moving at {row['Speed_FAV']:.2f} m/s with an acceleration of {row['Acc_FAV']:.2f} m/s². Its position is {row['Pos_FAV']:.2f} meters.
The bump-to-bump spatial gap between them is {row['Spatial_Gap']:.2f} meters, and the spatial headway is {row['Spatial_Headway']:.2f} meters.
The speed difference is {row['Speed_Diff']:.2f} m/s, and the estimated time to collision is {row['TTC']:.2f} seconds.
Over the next second, the projected acceleration is {row['1s_Acc']:.2f} m/s².
        """
        descriptions.append(paragraph.strip())

    return "\n\n".join(descriptions)


