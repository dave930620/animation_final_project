import csv
from bvh import Bvh

# Read the BVH file
with open('deep-motion-editing/retargeting/datasets/Mixamo/Aj/Running.bvh', 'r') as file:
    mocap = Bvh(file.read())

# Open a CSV file to write data
with open('train_data_2leg.csv', 'a', newline='') as csvfile:
    # Create a CSV writer
    csvwriter = csv.writer(csvfile)

    # Traverse each frame
    for frame_number in range(mocap.nframes):
        # Initialize a row of data
        row = []

        # Get the position of each joint and add it to the row data
        for joint in mocap.get_joints():
            joint_name = joint.name
            # Retrieve the joint channels and identify the indices for position channels
            channels = mocap.joint_channels(joint_name)
            # Initialize position values
            x_pos, y_pos, z_pos = 0, 0, 0
            # Retrieve position data if available
            if 'Xposition' in channels:
                x_index = channels.index('Xposition')
                x_pos = float(mocap.frame_joint_channel(frame_number,joint_name,'Xposition'))
            if 'Yposition' in channels:
                y_index = channels.index('Yposition')
                y_pos = float(mocap.frame_joint_channel(frame_number,joint_name,'Yposition'))
            if 'Zposition' in channels:
                z_index = channels.index('Zposition')
                z_pos = float(mocap.frame_joint_channel(frame_number,joint_name,'Zposition'))
            
            row.extend([x_pos, y_pos, z_pos])

        # Add output data
        row.append(1)  # Assumed output value

        # Write the row to the CSV file
        csvwriter.writerow(row)

print("CSV file has been generated, containing all frames' joint positions and a fixed output value.")
