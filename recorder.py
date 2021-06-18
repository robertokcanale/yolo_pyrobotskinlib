
import pyrobotskinlib as rsl 
import time

# FIXME: it works but when the programs end, it throws "pure virtual method called"

# Load skin object
S = rsl.RobotSkin("calibration_files/collaborate_handle_1_ale.json")

# Read skind data from shared memory
u = rsl.SkinUpdaterFromShMem(S)

# Start the updater
u.start_robot_skin_updater()

# Define the recorder and the filename
R = rsl.SkinRecorder("recorded_data/hand_record.txt", S)

# Record data from th updater u
R.attach_robot_skin_updater(u)

# Start recording
R.start()

print("Recording skin data for 10 seconds")

# Wait 2 second in the main thread
time.sleep(10)

# Stop recording
R.stop()
